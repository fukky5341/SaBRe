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
Threshold: 0.1285293


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0567755, 0.0436238, -0.0567755, 0.0436238, -0.1003993, 0.1003993)
1: (-0.0519022, 0.0497940, -0.0519022, 0.0497940, -0.1016962, 0.1016962)
2: (-0.0556083, 0.1421330, -0.0556083, 0.1421330, -0.1977413, 0.1977413)
3: (-0.0335104, 0.0596944, -0.0335104, 0.0596944, -0.0932048, 0.0932048)
4: (-0.0639282, 0.0638309, -0.0639282, 0.0638309, -0.1277591, 0.1277591)
5: (-0.0390747, 0.0742419, -0.0390747, 0.0742419, -0.1133166, 0.1133166)
6: (-0.1072323, 0.0786328, -0.1072323, 0.0786328, -0.1858651, 0.1858651)
7: (0.8322183, 1.0115921, 0.8322183, 1.0115921, -0.1793738, 0.1793738)
8: (-0.0925941, 0.1017981, -0.0925941, 0.1017981, -0.1943922, 0.1943922)
9: (-0.0820671, 0.1014012, -0.0820671, 0.1014012, -0.1834683, 0.1834683)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.78 + 2.02 = 3.80 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.89 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.87 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.97 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.97
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.97
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0486525, 0.0394308, -0.0553262, 0.0428782, -0.0915307, 0.0947570
1: -0.0437831, 0.0431827, -0.0504544, 0.0486183, -0.0924015, 0.0936371
2: -0.0467831, 0.1321014, -0.0540305, 0.1403491, -0.1871322, 0.1861319
3: -0.0258959, 0.0528966, -0.0321393, 0.0584856, -0.0843815, 0.0850359
4: -0.0558123, 0.0537354, -0.0624850, 0.0620202, -0.1178325, 0.1162204
5: -0.0308037, 0.0651902, -0.0376040, 0.0726128, -0.1034166, 0.1027941
6: -0.0978664, 0.0696649, -0.1055519, 0.0770381, -0.1749045, 0.1752168
7: 0.8451583, 1.0068395, 0.8345195, 1.0107408, -0.1655824, 0.1723200
8: -0.0838874, 0.0911671, -0.0910458, 0.0998803, -0.1837676, 0.1822129
9: -0.0755280, 0.0848021, -0.0809011, 0.0984494, -0.1739774, 0.1657032

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.00 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.95 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0989675, 0.0653305, -0.0519412, 0.0411367, -0.1401041, 0.1172717
1: -0.0940512, 0.0840198, -0.0470728, 0.0458724, -0.1399237, 0.1310927
2: -0.1015411, 0.1940653, -0.0503454, 0.1361827, -0.2377238, 0.2444108
3: -0.0734262, 0.0948860, -0.0289369, 0.0556623, -0.1290885, 0.1238229
4: -0.1059434, 0.1165478, -0.0591142, 0.0577908, -0.1637342, 0.1756620
5: -0.0818929, 0.1216681, -0.0341687, 0.0688079, -0.1507009, 0.1558368
6: -0.1561526, 0.1250587, -0.1016271, 0.0733134, -0.2294660, 0.2266859
7: 0.7652297, 1.0363698, 0.8398939, 1.0087531, -0.2435234, 0.1964759
8: -0.1376683, 0.1576294, -0.0874296, 0.0954010, -0.2330693, 0.2450590
9: -0.1160112, 0.1873330, -0.0781779, 0.0915552, -0.2075665, 0.2655109

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 153

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.89 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.88 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.61 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.61
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.61
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.61
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.61
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0477044, 0.0390412, -0.0513211, 0.0408177, -0.0885221, 0.0903623
1: -0.0429169, 0.0423618, -0.0464534, 0.0453694, -0.0882863, 0.0888152
2: -0.0457935, 0.1309852, -0.0496704, 0.1354194, -0.1812129, 0.1806556
3: -0.0253093, 0.0520526, -0.0283503, 0.0551451, -0.0804544, 0.0804029
4: -0.0550926, 0.0526779, -0.0584967, 0.0570160, -0.1121086, 0.1111746
5: -0.0298592, 0.0643113, -0.0335394, 0.0681109, -0.0979701, 0.0978508
6: -0.0968912, 0.0686750, -0.1009082, 0.0726311, -0.1695223, 0.1695831
7: 0.8465894, 1.0063249, 0.8408783, 1.0083888, -0.1617994, 0.1654466
8: -0.0831261, 0.0901916, -0.0867671, 0.0945805, -0.1777065, 0.1769587
9: -0.0747555, 0.0833557, -0.0776790, 0.0902924, -0.1650479, 0.1610348

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.94 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.84 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0461878, 0.0384344, -0.0727348, 0.0518345, -0.0980223, 0.1111692
1: -0.0415427, 0.0410486, -0.0678453, 0.0627401, -0.1042828, 0.1088939
2: -0.0442105, 0.1292160, -0.0729826, 0.1617766, -0.2059871, 0.2021986
3: -0.0243892, 0.0507023, -0.0486087, 0.0730058, -0.0973950, 0.0993110
4: -0.0539778, 0.0509864, -0.0798206, 0.0837714, -0.1377492, 0.1308071
5: -0.0283586, 0.0629054, -0.0552709, 0.0921811, -0.1205398, 0.1181764
6: -0.0953313, 0.0671069, -0.1257367, 0.0961936, -0.1915249, 0.1928436
7: 0.8488567, 1.0055016, 0.8068796, 1.0209643, -0.1721076, 0.1986220
8: -0.0819488, 0.0886310, -0.1096436, 0.1229166, -0.2048654, 0.1982746
9: -0.0735199, 0.0811201, -0.0949067, 0.1339053, -0.2074252, 0.1760268

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 153

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.87 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0979411, 0.0648024, -0.0480360, 0.0391738, -0.1371149, 0.1128385
1: -0.0930259, 0.0831872, -0.0432173, 0.0426489, -0.1356749, 0.1264046
2: -0.1004237, 0.1928020, -0.0461397, 0.1313720, -0.2317957, 0.2389417
3: -0.0724552, 0.0940300, -0.0255105, 0.0523478, -0.1248030, 0.1195404
4: -0.1049213, 0.1152654, -0.0553363, 0.0530478, -0.1579691, 0.1706017
5: -0.0808513, 0.1205144, -0.0301872, 0.0646188, -0.1454700, 0.1507017
6: -0.1549626, 0.1239293, -0.0972323, 0.0690178, -0.2239803, 0.2211617
7: 0.7668594, 1.0357672, 0.8460938, 1.0065050, -0.2396456, 0.1896734
8: -0.1365718, 0.1562713, -0.0833834, 0.0905327, -0.2271045, 0.2396547
9: -0.1151855, 0.1852426, -0.0750257, 0.0838445, -0.1990300, 0.2602683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 153

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.91 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.96 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0961098, 0.0638603, -0.0692407, 0.0500368, -0.1461467, 0.1331010
1: -0.0911966, 0.0817017, -0.0643548, 0.0599057, -0.1511023, 0.1460565
2: -0.0984301, 0.1905480, -0.0691787, 0.1574759, -0.2559060, 0.2597267
3: -0.0707227, 0.0925026, -0.0453031, 0.0700915, -0.1408142, 0.1378057
4: -0.1030978, 0.1129774, -0.0763412, 0.0794057, -0.1825034, 0.1893186
5: -0.0789929, 0.1184560, -0.0517250, 0.0882535, -0.1672464, 0.1701810
6: -0.1528394, 0.1219143, -0.1216854, 0.0923489, -0.2451882, 0.2435997
7: 0.7697667, 1.0346919, 0.8124272, 1.0189124, -0.2491457, 0.2222648
8: -0.1346154, 0.1538480, -0.1059108, 0.1182928, -0.2529083, 0.2597589
9: -0.1137122, 0.1815130, -0.0920956, 0.1267890, -0.2405013, 0.2736085

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 75

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.57 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.84 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.17 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 4.17
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 4.17
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.17
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.17
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 4.17
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 4.17
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 4.17
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 4.17
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0432748, 0.0372691, -0.0644991, 0.0475974, -0.0908722, 0.1017682
1: -0.0389033, 0.0385263, -0.0596179, 0.0560593, -0.0949626, 0.0981442
2: -0.0411699, 0.1258180, -0.0640167, 0.1516396, -0.1928095, 0.1898347
3: -0.0226220, 0.0481088, -0.0408173, 0.0661366, -0.0887585, 0.0889261
4: -0.0518367, 0.0477375, -0.0716194, 0.0734812, -0.1253180, 0.1193569
5: -0.0254765, 0.0602052, -0.0469130, 0.0829237, -0.1084002, 0.1071182
6: -0.0923351, 0.0640953, -0.1161876, 0.0871315, -0.1794666, 0.1802829
7: 0.8532109, 1.0039209, 0.8199555, 1.0161277, -0.1629168, 0.1839654
8: -0.0796878, 0.0856337, -0.1008453, 0.1120185, -0.1917062, 0.1864790
9: -0.0711467, 0.0768262, -0.0882808, 0.1171317, -0.1882784, 0.1651070

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.89 seconds

## Relational analysis of IS_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0477044, 0.0390412, -0.0495427, 0.0399027, -0.0876071, 0.0885838
1: -0.0429169, 0.0423618, -0.0446767, 0.0439268, -0.0868437, 0.0870385
2: -0.0457935, 0.1309852, -0.0477342, 0.1332304, -0.1790239, 0.1787194
3: -0.0253093, 0.0520526, -0.0266677, 0.0536617, -0.0789710, 0.0787203
4: -0.0550926, 0.0526779, -0.0567256, 0.0547939, -0.1098865, 0.1094036
5: -0.0298592, 0.0643113, -0.0317346, 0.0661118, -0.0959709, 0.0960459
6: -0.0968912, 0.0686750, -0.0988461, 0.0706741, -0.1675654, 0.1675210
7: 0.8465894, 1.0063249, 0.8437020, 1.0073446, -0.1607552, 0.1626229
8: -0.0831261, 0.0901916, -0.0848672, 0.0922271, -0.1753532, 0.1750588
9: -0.0747555, 0.0833557, -0.0762482, 0.0866701, -0.1614257, 0.1596039

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.88 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0589733, 0.0435493, -0.0679050, 0.0493497, -0.1083230, 0.1114543
1: -0.0531275, 0.0521194, -0.0630204, 0.0588222, -0.1119496, 0.1151398
2: -0.0575560, 0.1441305, -0.0677246, 0.1558319, -0.2133878, 0.2118551
3: -0.0321458, 0.0620855, -0.0440395, 0.0689775, -0.1011233, 0.1061250
4: -0.0633753, 0.0652467, -0.0750111, 0.0777368, -0.1411121, 0.1402578
5: -0.0410086, 0.0747574, -0.0503695, 0.0867521, -0.1277607, 0.1251269
6: -0.1084820, 0.0803253, -0.1201367, 0.0908792, -0.1993612, 0.2004620
7: 0.8297446, 1.0124412, 0.8145479, 1.0181279, -0.1883833, 0.1978933
8: -0.0918730, 0.1017869, -0.1044839, 0.1165254, -0.2083984, 0.2062708
9: -0.0839365, 0.0999669, -0.0910210, 0.1240686, -0.2080050, 0.1909879

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0445697, 0.0377871, -0.0727348, 0.0518345, -0.0964042, 0.1105219
1: -0.0400766, 0.0396475, -0.0678453, 0.0627401, -0.1028166, 0.1074928
2: -0.0425215, 0.1273285, -0.0729826, 0.1617766, -0.2042981, 0.2003111
3: -0.0234076, 0.0492617, -0.0486087, 0.0730058, -0.0964134, 0.0978704
4: -0.0527885, 0.0491818, -0.0798206, 0.0837714, -0.1365599, 0.1290024
5: -0.0267578, 0.0614055, -0.0552709, 0.0921811, -0.1189389, 0.1166764
6: -0.0936670, 0.0654341, -0.1257367, 0.0961936, -0.1898606, 0.1911708
7: 0.8512753, 1.0046235, 0.8068796, 1.0209643, -0.1696890, 0.1977439
8: -0.0806929, 0.0869661, -0.1096436, 0.1229166, -0.2036095, 0.1966097
9: -0.0722017, 0.0787350, -0.0949067, 0.1339053, -0.2061070, 0.1736416

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 153

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0929701, 0.0622450, -0.0605090, 0.0441636, -0.1371338, 0.1227540
1: -0.0880600, 0.0791548, -0.0545189, 0.0534491, -0.1415091, 0.1336737
2: -0.0950119, 0.1866834, -0.0591589, 0.1459219, -0.2409338, 0.2458423
3: -0.0677524, 0.0898837, -0.0330775, 0.0634528, -0.1312051, 0.1229612
4: -0.0999711, 0.1090545, -0.0645041, 0.0669595, -0.1669306, 0.1735586
5: -0.0758066, 0.1149267, -0.0425280, 0.0761810, -0.1519876, 0.1574547
6: -0.1491989, 0.1184595, -0.1100615, 0.0819130, -0.2311119, 0.2285210
7: 0.7747519, 1.0328478, 0.8274490, 1.0132746, -0.2385226, 0.2053988
8: -0.1312611, 0.1496933, -0.0930651, 0.1033670, -0.2346282, 0.2427584
9: -0.1111862, 0.1751182, -0.0851876, 0.1022306, -0.2134168, 0.2603058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.97 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0979411, 0.0648024, -0.0463869, 0.0385141, -0.1364551, 0.1111893
1: -0.0930259, 0.0831872, -0.0417231, 0.0412210, -0.1342469, 0.1249103
2: -0.1004237, 0.1928020, -0.0444183, 0.1294483, -0.2298720, 0.2372203
3: -0.0724552, 0.0940300, -0.0245100, 0.0508795, -0.1233348, 0.1185400
4: -0.1049213, 0.1152654, -0.0541242, 0.0512085, -0.1561299, 0.1693896
5: -0.0808513, 0.1205144, -0.0285556, 0.0630900, -0.1439413, 0.1490701
6: -0.1549626, 0.1239293, -0.0955361, 0.0673128, -0.2222753, 0.2194654
7: 0.7668594, 1.0357672, 0.8485589, 1.0056100, -0.2387506, 0.1872083
8: -0.1365718, 0.1562713, -0.0821034, 0.0888359, -0.2254077, 0.2383747
9: -0.1151855, 0.1852426, -0.0736822, 0.0814136, -0.1965992, 0.2589248

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 153

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.07 seconds

## Relational analysis of IS_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0911194, 0.0612928, -0.0831686, 0.0572024, -0.1483217, 0.1444614
1: -0.0862111, 0.0776535, -0.0782685, 0.0712039, -0.1574150, 0.1559219
2: -0.0929970, 0.1844054, -0.0843415, 0.1746192, -0.2676162, 0.2687468
3: -0.0660015, 0.0883400, -0.0584796, 0.0817085, -0.1477100, 0.1468196
4: -0.0981282, 0.1067420, -0.0902107, 0.0968079, -0.1949361, 0.1969527
5: -0.0739283, 0.1128464, -0.0658596, 0.1039093, -0.1778376, 0.1787060
6: -0.1470530, 0.1164230, -0.1378344, 0.1076744, -0.2547274, 0.2542573
7: 0.7776904, 1.0317609, 0.7903138, 1.0270920, -0.2494016, 0.2414471
8: -0.1292840, 0.1472441, -0.1207902, 0.1367233, -0.2660073, 0.2680343
9: -0.1096973, 0.1713489, -0.1033008, 0.1551557, -0.2648530, 0.2746496

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 75

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 2.42 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0961098, 0.0638603, -0.0674974, 0.0491399, -0.1452498, 0.1313577
1: -0.0911966, 0.0817017, -0.0626132, 0.0584915, -0.1496881, 0.1443149
2: -0.0984301, 0.1905480, -0.0672809, 0.1553301, -0.2537602, 0.2578289
3: -0.0707227, 0.0925026, -0.0436538, 0.0686374, -0.1393601, 0.1361564
4: -0.1030978, 0.1129774, -0.0746052, 0.0772275, -0.1803252, 0.1875826
5: -0.0789929, 0.1184560, -0.0499557, 0.0862939, -0.1652868, 0.1684117
6: -0.1528394, 0.1219143, -0.1196640, 0.0904306, -0.2432700, 0.2415783
7: 0.7697667, 1.0346919, 0.8151951, 1.0178885, -0.2481219, 0.2194968
8: -0.1346154, 0.1538480, -0.1040484, 0.1159860, -0.2506014, 0.2578964
9: -0.1137122, 0.1815130, -0.0906931, 0.1232383, -0.2369505, 0.2722060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 75

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_B2_B1

### Relational analysis result of IS_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.99 seconds

## Relational analysis of IS_A2_B2_B2_B2

### Relational analysis result of IS_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.95 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.88 seconds
IS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207

## BFS IS instance: IS_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0338278, 0.0325315, -0.0556953, 0.0430681, -0.0768958, 0.0882268
1: -0.0324252, 0.0282721, -0.0508231, 0.0489177, -0.0813429, 0.0790952
2: -0.0288832, 0.1134331, -0.0544324, 0.1408034, -0.1696866, 0.1678655
3: -0.0163998, 0.0426651, -0.0324885, 0.0587935, -0.0751933, 0.0751535
4: -0.0435976, 0.0380635, -0.0628525, 0.0624813, -0.1060789, 0.1009160
5: -0.0186620, 0.0492274, -0.0379785, 0.0730277, -0.0916897, 0.0872059
6: -0.0816640, 0.0530311, -0.1059799, 0.0774442, -0.1591081, 0.1590110
7: 0.8708140, 0.9986646, 0.8339335, 1.0109577, -0.1401437, 0.1647311
8: -0.0707415, 0.0749915, -0.0914401, 0.1003687, -0.1711102, 0.1664316
9: -0.0623411, 0.0624024, -0.0811981, 0.0992012, -0.1615423, 0.1436005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_B1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.00 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0373301, 0.0344208, -0.0624719, 0.0465545, -0.0838846, 0.0968927
1: -0.0347024, 0.0323612, -0.0575928, 0.0544149, -0.0891173, 0.0899541
2: -0.0337823, 0.1181115, -0.0618098, 0.1491445, -0.1829267, 0.1799213
3: -0.0187057, 0.0447943, -0.0388995, 0.0644457, -0.0831514, 0.0836938
4: -0.0468794, 0.0412772, -0.0696007, 0.0709484, -0.1178277, 0.1108779
5: -0.0207935, 0.0536051, -0.0448557, 0.0806450, -0.1014385, 0.0984608
6: -0.0856443, 0.0574337, -0.1138371, 0.0849008, -0.1705452, 0.1712708
7: 0.8638123, 1.0007511, 0.8231742, 1.0149374, -0.1511251, 0.1775770
8: -0.0743070, 0.0789542, -0.0986796, 0.1093359, -0.1836430, 0.1776338
9: -0.0658457, 0.0676017, -0.0866500, 0.1130030, -0.1788487, 0.1542516

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_B1_A2_B1

### Relational analysis result of IS_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.91 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2

### Relational analysis result of IS_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0369349, 0.0342076, -0.0420590, 0.0367464, -0.0736812, 0.0762666
1: -0.0344454, 0.0318998, -0.0379675, 0.0373949, -0.0718403, 0.0698673
2: -0.0332294, 0.1175835, -0.0398130, 0.1242938, -0.1575232, 0.1573965
3: -0.0184455, 0.0445540, -0.0218293, 0.0474154, -0.0658609, 0.0663833
4: -0.0465090, 0.0409145, -0.0509192, 0.0462802, -0.0927892, 0.0918337
5: -0.0205529, 0.0531111, -0.0243465, 0.0589939, -0.0795469, 0.0774576
6: -0.0851952, 0.0569369, -0.0909912, 0.0628531, -0.1480483, 0.1479281
7: 0.8646024, 1.0005157, 0.8551641, 1.0033196, -0.1387172, 0.1453516
8: -0.0739047, 0.0785070, -0.0786962, 0.0842893, -0.1581940, 0.1572032
9: -0.0654502, 0.0670149, -0.0701598, 0.0749002, -0.1403504, 0.1371748

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.93 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0407223, 0.0361620, -0.0475543, 0.0389811, -0.0797034, 0.0837163
1: -0.0369829, 0.0361300, -0.0427809, 0.0422318, -0.0792147, 0.0789109
2: -0.0382975, 0.1225898, -0.0456368, 0.1308101, -0.1691076, 0.1682266
3: -0.0209431, 0.0467567, -0.0252182, 0.0519190, -0.0728620, 0.0719750
4: -0.0499040, 0.0446509, -0.0549822, 0.0525105, -0.1024146, 0.0996331
5: -0.0231234, 0.0576398, -0.0297107, 0.0641721, -0.0872955, 0.0873505
6: -0.0894887, 0.0614913, -0.0967369, 0.0685197, -0.1580084, 0.1582282
7: 0.8573478, 1.0026742, 0.8468139, 1.0062435, -0.1488957, 0.1558603
8: -0.0775933, 0.0827862, -0.0830095, 0.0900371, -0.1676304, 0.1657957
9: -0.0690757, 0.0727468, -0.0746333, 0.0831345, -0.1522102, 0.1473801

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.00 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.96 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0589733, 0.0435493, -0.0612493, 0.0459254, -0.1048987, 0.1047985
1: -0.0531275, 0.0521194, -0.0563714, 0.0534231, -0.1065505, 0.1084908
2: -0.0575560, 0.1441305, -0.0604787, 0.1476395, -0.2051955, 0.2046092
3: -0.0321458, 0.0620855, -0.0377428, 0.0634260, -0.0955718, 0.0998283
4: -0.0633753, 0.0652467, -0.0683832, 0.0694207, -0.1327960, 0.1336299
5: -0.0410086, 0.0747574, -0.0436149, 0.0792707, -0.1202793, 0.1183723
6: -0.1084820, 0.0803253, -0.1124195, 0.0835555, -0.1920375, 0.1927448
7: 0.8297446, 1.0124412, 0.8251156, 1.0142194, -0.1844748, 0.1873256
8: -0.0918730, 0.1017869, -0.0973735, 0.1077181, -0.1995911, 0.1991603
9: -0.0839365, 0.0999669, -0.0856663, 0.1105128, -0.1944493, 0.1856332

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0589733, 0.0435493, -0.1104498, 0.0712379, -0.1302112, 0.1539991
1: -0.0531275, 0.0521194, -0.1055220, 0.0933342, -0.1464617, 0.1576414
2: -0.0575560, 0.1441305, -0.1140415, 0.2081986, -0.2657546, 0.2581719
3: -0.0321458, 0.0620855, -0.0842892, 0.1044634, -0.1366092, 0.1463747
4: -0.0633753, 0.0652467, -0.1173777, 0.1308945, -0.1942699, 0.1826244
5: -0.0410086, 0.0747574, -0.0935457, 0.1345750, -0.1755835, 0.1683032
6: -0.1084820, 0.0803253, -0.1694661, 0.1376933, -0.2461753, 0.2497914
7: 0.8297446, 1.0124412, 0.7469992, 1.0431129, -0.2133683, 0.2654420
8: -0.0918730, 0.1017869, -0.1499350, 0.1728237, -0.2646967, 0.2517219
9: -0.0839365, 0.0999669, -0.1252490, 0.2107190, -0.2946555, 0.2252159

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 75

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0445697, 0.0377871, -0.0660039, 0.0483716, -0.0929414, 0.1037910
1: -0.0400766, 0.0396475, -0.0611213, 0.0572800, -0.0973566, 0.1007688
2: -0.0425215, 0.1273285, -0.0656549, 0.1534919, -0.1960134, 0.1929834
3: -0.0234076, 0.0492617, -0.0422409, 0.0673917, -0.0907993, 0.0915026
4: -0.0527885, 0.0491818, -0.0731180, 0.0753614, -0.1281500, 0.1222997
5: -0.0267578, 0.0614055, -0.0484401, 0.0846152, -0.1113729, 0.1098456
6: -0.0936670, 0.0654341, -0.1179324, 0.0887873, -0.1824543, 0.1833664
7: 0.8512753, 1.0046235, 0.8175664, 1.0170116, -0.1657364, 0.1870571
8: -0.0806929, 0.0869661, -0.1024529, 0.1140097, -0.1947026, 0.1894191
9: -0.0722017, 0.0787350, -0.0894915, 0.1201966, -0.1923983, 0.1682265

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 153

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0445697, 0.0377871, -0.1155549, 0.0738643, -0.1184340, 0.1533420
1: -0.0400766, 0.0396475, -0.1106219, 0.0974754, -0.1375520, 0.1502694
2: -0.0425215, 0.1273285, -0.1195991, 0.2144822, -0.2570036, 0.2469276
3: -0.0234076, 0.0492617, -0.0891188, 0.1087214, -0.1321289, 0.1383805
4: -0.0527885, 0.0491818, -0.1224614, 0.1372731, -0.1900616, 0.1716431
5: -0.0267578, 0.0614055, -0.0987266, 0.1403133, -0.1670711, 0.1601321
6: -0.0936670, 0.0654341, -0.1753853, 0.1433106, -0.2369776, 0.2408194
7: 0.8512753, 1.0046235, 0.7388937, 1.0461111, -0.1948358, 0.2657298
8: -0.0806929, 0.0869661, -0.1553888, 0.1795790, -0.2602719, 0.2423549
9: -0.0722017, 0.0787350, -0.1293561, 0.2211164, -0.2933181, 0.2080911

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0780608, 0.0545745, -0.0522522, 0.0408605, -0.1189213, 0.1068267
1: -0.0731658, 0.0670604, -0.0470376, 0.0462997, -0.1194655, 0.1140980
2: -0.0787808, 0.1683321, -0.0505405, 0.1362902, -0.2150710, 0.2188726
3: -0.0536473, 0.0774481, -0.0280683, 0.0561015, -0.1097489, 0.1055164
4: -0.0851242, 0.0904259, -0.0584352, 0.0577503, -0.1428746, 0.1488611
5: -0.0606759, 0.0981678, -0.0343587, 0.0685271, -0.1292030, 0.1325265
6: -0.1319120, 0.1020540, -0.1015690, 0.0733766, -0.2052886, 0.2036230
7: 0.7984235, 1.0240922, 0.8397914, 1.0087932, -0.2103697, 0.1843008
8: -0.1153333, 0.1299642, -0.0866560, 0.0948711, -0.2102045, 0.2166203
9: -0.0991914, 0.1447525, -0.0784607, 0.0900595, -0.1892509, 0.2232132

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 153

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.05 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0850904, 0.0581911, -0.0586194, 0.0434076, -0.1284981, 0.1168104
1: -0.0801884, 0.0727628, -0.0528067, 0.0518129, -0.1320013, 0.1255695
2: -0.0864336, 0.1769846, -0.0571865, 0.1437175, -0.2301512, 0.2341711
3: -0.0602978, 0.0833115, -0.0319311, 0.0617703, -0.1220681, 0.1152426
4: -0.0921245, 0.0992091, -0.0631151, 0.0648518, -0.1569763, 0.1623242
5: -0.0678099, 0.1060695, -0.0406584, 0.0744293, -0.1422392, 0.1467278
6: -0.1400626, 0.1097891, -0.1081179, 0.0799593, -0.2200219, 0.2179070
7: 0.7872626, 1.0282203, 0.8302738, 1.0122490, -0.2249864, 0.1979465
8: -0.1228433, 0.1392663, -0.0915983, 0.1014226, -0.2242658, 0.2308646
9: -0.1048469, 0.1590697, -0.0836481, 0.0994451, -0.2042920, 0.2427178

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 153

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.16 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0888218, 0.0601108, -0.0360098, 0.0337086, -0.1225303, 0.0961206
1: -0.0839158, 0.0757897, -0.0338439, 0.0308197, -0.1147355, 0.1096336
2: -0.0904958, 0.1815774, -0.0319354, 0.1163478, -0.2068435, 0.2135128
3: -0.0638278, 0.0864237, -0.0178364, 0.0439916, -0.1078194, 0.1042601
4: -0.0958402, 0.1038712, -0.0456422, 0.0400657, -0.1359059, 0.1495135
5: -0.0715966, 0.1102637, -0.0199900, 0.0519548, -0.1235514, 0.1302537
6: -0.1443890, 0.1138948, -0.0841438, 0.0557740, -0.2001630, 0.1980386
7: 0.7813383, 1.0304117, 0.8664518, 0.9999646, -0.2186263, 0.1639599
8: -0.1268294, 0.1442039, -0.0729629, 0.0774603, -0.2042898, 0.2171668
9: -0.1078488, 0.1666694, -0.0645245, 0.0656417, -0.1734905, 0.2311939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 153

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_B1_B1

### Relational analysis result of IS_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.12 seconds

## Relational analysis of IS_A2_B1_B2_B1_B2

### Relational analysis result of IS_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0958463, 0.0637248, -0.0396248, 0.0356585, -0.1315049, 0.1033495
1: -0.0909333, 0.0814879, -0.0361944, 0.0350403, -0.1259736, 0.1176824
2: -0.0981432, 0.1902237, -0.0369920, 0.1211766, -0.2193198, 0.2272157
3: -0.0704734, 0.0922827, -0.0202165, 0.0461893, -0.1166628, 0.1124992
4: -0.1028353, 0.1126481, -0.0490296, 0.0433827, -0.1462181, 0.1616777
5: -0.0787255, 0.1181598, -0.0221900, 0.0564732, -0.1351987, 0.1403497
6: -0.1525338, 0.1216243, -0.0882522, 0.0603181, -0.2128519, 0.2098765
7: 0.7701852, 1.0345371, 0.8592249, 1.0021183, -0.2319331, 0.1753122
8: -0.1343339, 0.1534993, -0.0766431, 0.0815504, -0.2158843, 0.2301424
9: -0.1135002, 0.1809763, -0.0681418, 0.0710080, -0.1845082, 0.2491181

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 153

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.85 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0762336, 0.0536345, -0.0744001, 0.0526912, -0.1289248, 0.1280346
1: -0.0713405, 0.0655782, -0.0695088, 0.0640909, -0.1354314, 0.1350871
2: -0.0767916, 0.1660830, -0.0747955, 0.1638264, -0.2406180, 0.2408785
3: -0.0519188, 0.0759241, -0.0501842, 0.0743948, -0.1263136, 0.1261083
4: -0.0833047, 0.0881429, -0.0814789, 0.0858521, -0.1691568, 0.1696218
5: -0.0588216, 0.0961140, -0.0569609, 0.0940530, -0.1528745, 0.1530749
6: -0.1297934, 0.1000435, -0.1276675, 0.0980260, -0.2278193, 0.2277109
7: 0.8013245, 1.0230191, 0.8042357, 1.0219425, -0.2206180, 0.2187834
8: -0.1133813, 0.1275464, -0.1114227, 0.1251202, -0.2385015, 0.2389690
9: -0.0977214, 0.1410311, -0.0962464, 0.1372969, -0.2350183, 0.2372775

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 75

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_B1_A1_A1

### Relational analysis result of IS_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.98 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2

### Relational analysis result of IS_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0832499, 0.0572442, -0.0811453, 0.0561614, -0.1394113, 0.1383895
1: -0.0783497, 0.0712698, -0.0762472, 0.0695626, -0.1479123, 0.1475170
2: -0.0844299, 0.1747192, -0.0821387, 0.1721288, -0.2565587, 0.2568578
3: -0.0585565, 0.0817762, -0.0565655, 0.0800208, -0.1385774, 0.1383417
4: -0.0902916, 0.0969095, -0.0881959, 0.0942798, -0.1845715, 0.1851054
5: -0.0659420, 0.1040007, -0.0638062, 0.1016350, -0.1675770, 0.1678069
6: -0.1379285, 0.1077638, -0.1354884, 0.1054480, -0.2433766, 0.2432522
7: 0.7901847, 1.0271395, 0.7935264, 1.0259036, -0.2357188, 0.2336132
8: -0.1208769, 0.1368308, -0.1186286, 0.1340458, -0.2549227, 0.2554594
9: -0.1033661, 0.1553212, -0.1016730, 0.1510348, -0.2544009, 0.2569942

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 75

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_B1_A2_A1

### Relational analysis result of IS_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.96 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2

### Relational analysis result of IS_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0869877, 0.0591672, -0.0541969, 0.0422972, -0.1292849, 0.1133641
1: -0.0820837, 0.0743019, -0.0493262, 0.0477023, -0.1297860, 0.1236281
2: -0.0884991, 0.1793199, -0.0528011, 0.1389591, -0.2274582, 0.2321211
3: -0.0620927, 0.0848940, -0.0310709, 0.0575437, -0.1196364, 0.1159649
4: -0.0940139, 0.1015797, -0.0613604, 0.0606092, -0.1546230, 0.1629402
5: -0.0697354, 0.1082022, -0.0364579, 0.0713434, -0.1410788, 0.1446601
6: -0.1422625, 0.1118768, -0.1042426, 0.0757954, -0.2180579, 0.2161193
7: 0.7842501, 1.0293345, 0.8363126, 1.0100776, -0.2258275, 0.1930220
8: -0.1248701, 0.1417769, -0.0898394, 0.0983859, -0.2232561, 0.2316163
9: -0.1063733, 0.1629340, -0.0799926, 0.0961494, -0.2025228, 0.2429267

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 75

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.90 seconds

## Relational analysis of IS_A2_B2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.18 seconds

## BFS IS instance: IS_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0940149, 0.0627825, -0.0596541, 0.0451048, -0.1391196, 0.1224366
1: -0.0891037, 0.0800023, -0.0547779, 0.0521291, -0.1412328, 0.1347802
2: -0.0961493, 0.1879694, -0.0587422, 0.1456761, -0.2418254, 0.2467116
3: -0.0687408, 0.0907551, -0.0362337, 0.0620955, -0.1308362, 0.1269888
4: -0.1010115, 0.1103598, -0.0667948, 0.0674276, -0.1684391, 0.1771546
5: -0.0768669, 0.1161011, -0.0419961, 0.0774776, -0.1543445, 0.1580971
6: -0.1504102, 0.1196091, -0.1105699, 0.0818003, -0.2322105, 0.2301790
7: 0.7730931, 1.0334615, 0.8276480, 1.0132827, -0.2401896, 0.2058134
8: -0.1323773, 0.1510758, -0.0956694, 0.1056072, -0.2379845, 0.2467452
9: -0.1120267, 0.1772461, -0.0843830, 0.1072640, -0.2192908, 0.2616291

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 75

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.82 seconds

## Relational analysis of IS_A2_B2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.07 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.83 seconds
IS_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.83
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.83
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.83
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.83
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.83
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.83
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.83
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.83
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.83
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.83
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.83
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.83
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.83
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.83
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.83
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.83
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.83
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.83
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.83
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.83
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.83
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.83
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.83
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.83
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.83
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.83
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.83
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.83
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.83
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.83
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.83
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.83
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207

## BFS IS instance: IS_A1_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0338278, 0.0325315, -0.0494377, 0.0398487, -0.0736765, 0.0819692
1: -0.0324252, 0.0282721, -0.0445719, 0.0438416, -0.0762668, 0.0728440
2: -0.0288832, 0.1134331, -0.0476200, 0.1331012, -0.1619844, 0.1610531
3: -0.0163998, 0.0426651, -0.0265684, 0.0535741, -0.0699739, 0.0692335
4: -0.0435976, 0.0380635, -0.0566211, 0.0546627, -0.0982603, 0.0946846
5: -0.0186620, 0.0492274, -0.0316280, 0.0659938, -0.0846558, 0.0808555
6: -0.0816640, 0.0530311, -0.0987244, 0.0705586, -0.1522226, 0.1517555
7: 0.8708140, 0.9986646, 0.8438687, 1.0072826, -0.1364686, 0.1547958
8: -0.0707415, 0.0749915, -0.0847550, 0.0920882, -0.1628297, 0.1597465
9: -0.0623411, 0.0624024, -0.0761638, 0.0864564, -0.1487974, 0.1385662

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.97 seconds

## Relational analysis of IS_A1_B1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.19 seconds

## BFS IS instance: IS_A1_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0338278, 0.0325315, -0.0971629, 0.0644021, -0.0982299, 0.1296945
1: -0.0324252, 0.0282721, -0.0922486, 0.0825559, -0.1149811, 0.1205207
2: -0.0288832, 0.1134331, -0.0995765, 0.1918442, -0.2207274, 0.2130096
3: -0.0163998, 0.0426651, -0.0717190, 0.0933809, -0.1097807, 0.1143841
4: -0.0435976, 0.0380635, -0.1041464, 0.1142932, -0.1578908, 0.1422099
5: -0.0186620, 0.0492274, -0.0800616, 0.1196397, -0.1383017, 0.1292890
6: -0.0816640, 0.0530311, -0.1540604, 0.1230730, -0.2047370, 0.2070915
7: 0.8708140, 0.9986646, 0.7680950, 1.0353100, -0.1644960, 0.2305696
8: -0.0707415, 0.0749915, -0.1357404, 0.1552415, -0.2259830, 0.2107320
9: -0.0623411, 0.0624024, -0.1145594, 0.1836577, -0.2459988, 0.1769618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.02 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.18 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0373301, 0.0344208, -0.0562212, 0.0433386, -0.0806687, 0.0906420
1: -0.0347024, 0.0323612, -0.0513485, 0.0493444, -0.0840468, 0.0837097
2: -0.0337823, 0.1181115, -0.0550049, 0.1414507, -0.1752329, 0.1731164
3: -0.0187057, 0.0447943, -0.0329860, 0.0592321, -0.0779378, 0.0777803
4: -0.0468794, 0.0412772, -0.0633762, 0.0631384, -0.1100178, 0.1046534
5: -0.0207935, 0.0536051, -0.0385122, 0.0736189, -0.0944124, 0.0921173
6: -0.0856443, 0.0574337, -0.1065896, 0.0780229, -0.1636672, 0.1640233
7: 0.8638123, 1.0007511, 0.8330984, 1.0112665, -0.1474542, 0.1676527
8: -0.0743070, 0.0789542, -0.0920020, 0.1010646, -0.1753716, 0.1709562
9: -0.0658457, 0.0676017, -0.0816212, 0.1002724, -0.1661181, 0.1492229

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.99 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.45 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0373301, 0.0344208, -0.1059940, 0.0689454, -0.1062755, 0.1404148
1: -0.0347024, 0.0323612, -0.1010707, 0.0897196, -0.1244221, 0.1334319
2: -0.0337823, 0.1181115, -0.1091906, 0.2027140, -0.2364963, 0.2273021
3: -0.0187057, 0.0447943, -0.0800737, 0.1007467, -0.1194525, 0.1248680
4: -0.0468794, 0.0412772, -0.1129406, 0.1253272, -0.1722065, 0.1542177
5: -0.0207935, 0.0536051, -0.0890237, 0.1295663, -0.1503598, 0.1426288
6: -0.0856443, 0.0574337, -0.1642996, 0.1327903, -0.2184346, 0.2217333
7: 0.8638123, 1.0007511, 0.7540736, 1.0404965, -0.1766842, 0.2466776
8: -0.0743070, 0.0789542, -0.1451748, 0.1669274, -0.2412345, 0.2241290
9: -0.0658457, 0.0676017, -0.1216642, 0.2016439, -0.2674896, 0.1892659

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=54, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 75

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.95 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0369349, 0.0342076, -0.0370343, 0.0342612, -0.0711960, 0.0712419
1: -0.0344454, 0.0318998, -0.0345101, 0.0320159, -0.0664613, 0.0664099
2: -0.0332294, 0.1175835, -0.0333685, 0.1177163, -0.1509457, 0.1509521
3: -0.0184455, 0.0445540, -0.0185110, 0.0446145, -0.0630600, 0.0630650
4: -0.0465090, 0.0409145, -0.0466022, 0.0410058, -0.0875148, 0.0875167
5: -0.0205529, 0.0531111, -0.0206135, 0.0532354, -0.0737883, 0.0737246
6: -0.0851952, 0.0569369, -0.0853081, 0.0570618, -0.1422570, 0.1422450
7: 0.8646024, 1.0005157, 0.8644037, 1.0005748, -0.1359724, 0.1361120
8: -0.0739047, 0.0785070, -0.0740059, 0.0786195, -0.1525242, 0.1525129
9: -0.0654502, 0.0670149, -0.0655497, 0.0671625, -0.1326127, 0.1325647

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.89 seconds

## Relational analysis of IS_A1_B1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0369349, 0.0342076, -0.0772365, 0.0521253, -0.0890602, 0.1114441
1: -0.0344454, 0.0318998, -0.0638791, 0.0706818, -0.1051273, 0.0957789
2: -0.0332294, 0.1175835, -0.0796931, 0.1691374, -0.2023667, 0.1972766
3: -0.0184455, 0.0445540, -0.0451513, 0.0647481, -0.0831936, 0.0897053
4: -0.0465090, 0.0409145, -0.0776341, 0.0891567, -0.1356657, 0.1185486
5: -0.0205529, 0.0531111, -0.0565333, 0.0946296, -0.1151826, 0.1096444
6: -0.0851952, 0.0569369, -0.1305317, 0.0986913, -0.1838865, 0.1874685
7: 0.8646024, 1.0005157, 0.7976997, 1.0203046, -0.1557022, 0.2028159
8: -0.0739047, 0.0785070, -0.1077210, 0.1238451, -0.1977499, 0.1862280
9: -0.0654502, 0.0670149, -0.0986885, 0.1315671, -0.1970173, 0.1657034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 2.00 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0407223, 0.0361620, -0.0414809, 0.0364936, -0.0772159, 0.0776428
1: -0.0369829, 0.0361300, -0.0375417, 0.0368478, -0.0738307, 0.0736717
2: -0.0382975, 0.1225898, -0.0391575, 0.1235569, -0.1618544, 0.1617473
3: -0.0209431, 0.0467567, -0.0214460, 0.0471305, -0.0680736, 0.0682027
4: -0.0499040, 0.0446509, -0.0504802, 0.0455755, -0.0954795, 0.0951310
5: -0.0231234, 0.0576398, -0.0238175, 0.0584083, -0.0815316, 0.0814573
6: -0.0894887, 0.0614913, -0.0903414, 0.0622641, -0.1517528, 0.1518327
7: 0.8573478, 1.0026742, 0.8561085, 1.0030406, -0.1456928, 0.1465657
8: -0.0775933, 0.0827862, -0.0782192, 0.0836391, -0.1612324, 0.1610053
9: -0.0690757, 0.0727468, -0.0696909, 0.0739688, -0.1430446, 0.1424377

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.95 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0407223, 0.0361620, -0.0877964, 0.0550800, -0.0958022, 0.1239584
1: -0.0369829, 0.0361300, -0.0792435, 0.0770770, -0.1140598, 0.1153735
2: -0.0382975, 0.1225898, -0.0876415, 0.1777528, -0.2160503, 0.2102313
3: -0.0209431, 0.0467567, -0.0496319, 0.0877473, -0.1086903, 0.0963887
4: -0.0499040, 0.0446509, -0.0845607, 0.0973942, -0.1472982, 0.1292115
5: -0.0231234, 0.0576398, -0.0695260, 0.1014760, -0.1245994, 0.1271658
6: -0.0894887, 0.0614913, -0.1381282, 0.1101242, -0.1996128, 0.1996195
7: 0.8573478, 1.0026742, 0.7866596, 1.0280844, -0.1707366, 0.2160146
8: -0.0775933, 0.0827862, -0.1142457, 0.1314447, -0.2090379, 0.1970319
9: -0.0690757, 0.0727468, -0.1074190, 0.1424540, -0.2115298, 0.1801658

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.96 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0506979, 0.0402387, -0.0480974, 0.0391983, -0.0898962, 0.0883361
1: -0.0456292, 0.0449538, -0.0432729, 0.0427020, -0.0883312, 0.0882268
2: -0.0489181, 0.1344771, -0.0462037, 0.1314435, -0.1803616, 0.1806808
3: -0.0271253, 0.0547177, -0.0255477, 0.0524024, -0.0795277, 0.0802654
4: -0.0572928, 0.0560167, -0.0553814, 0.0531163, -0.1104091, 0.1113981
5: -0.0328209, 0.0670862, -0.0302480, 0.0646756, -0.0974965, 0.0973342
6: -0.0999702, 0.0717697, -0.0972954, 0.0690811, -0.1690513, 0.1690651
7: 0.8421148, 1.0079495, 0.8460020, 1.0065380, -0.1644232, 0.1619474
8: -0.0854496, 0.0932717, -0.0834310, 0.0905959, -0.1760454, 0.1767028
9: -0.0771944, 0.0877683, -0.0750757, 0.0839350, -0.1611294, 0.1628440

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_B2

### Relational analysis result of IS_A1_B2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0570648, 0.0427858, -0.0533357, 0.0418541, -0.0989189, 0.0961215
1: -0.0513982, 0.0504669, -0.0484659, 0.0470036, -0.0984018, 0.0989328
2: -0.0555639, 0.1419041, -0.0518636, 0.1378990, -0.1934629, 0.1937677
3: -0.0309880, 0.0603863, -0.0302561, 0.0568254, -0.0878133, 0.0906424
4: -0.0619725, 0.0631180, -0.0605028, 0.0595331, -0.1215057, 0.1236207
5: -0.0391203, 0.0729883, -0.0355839, 0.0703754, -0.1094957, 0.1085722
6: -0.1065189, 0.0783522, -0.1032440, 0.0748478, -0.1813667, 0.1815962
7: 0.8325975, 1.0114051, 0.8376800, 1.0095721, -0.1769747, 0.1737251
8: -0.0903916, 0.0998230, -0.0889193, 0.0972463, -0.1876379, 0.1887423
9: -0.0823816, 0.0971536, -0.0792997, 0.0943954, -0.1767770, 0.1764533

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2_B2

### Relational analysis result of IS_A1_B2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0466524, 0.0386203, -0.1015167, 0.0666420, -0.1132944, 0.1401370
1: -0.0419637, 0.0414509, -0.0965979, 0.0860877, -0.1280514, 0.1380489
2: -0.0446955, 0.1297580, -0.1043163, 0.1972031, -0.2418986, 0.2340743
3: -0.0246711, 0.0511160, -0.0758379, 0.0970123, -0.1216834, 0.1269539
4: -0.0543193, 0.0515047, -0.1084819, 0.1197330, -0.1740523, 0.1599866
5: -0.0288184, 0.0633361, -0.0844800, 0.1245336, -0.1533519, 0.1478162
6: -0.0958092, 0.0675873, -0.1591084, 0.1278637, -0.2236729, 0.2266957
7: 0.8481620, 1.0057539, 0.7611824, 1.0378670, -0.1897050, 0.2445714
8: -0.0823095, 0.0891091, -0.1403916, 0.1610028, -0.2433123, 0.2295007
9: -0.0738985, 0.0818051, -0.1180621, 0.1925250, -0.2664234, 0.1998672

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 75

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0517496, 0.0406594, -0.1083760, 0.0701709, -0.1219204, 0.1490354
1: -0.0465821, 0.0458644, -0.1034502, 0.0916519, -0.1382340, 0.1493146
2: -0.0500158, 0.1357039, -0.1117836, 0.2056458, -0.2556616, 0.2474875
3: -0.0277634, 0.0556540, -0.0823271, 0.1027334, -0.1304968, 0.1379811
4: -0.0580658, 0.0571897, -0.1153124, 0.1283033, -0.1863691, 0.1725021
5: -0.0338614, 0.0680611, -0.0914410, 0.1322438, -0.1661052, 0.1595021
6: -0.1010519, 0.0728570, -0.1670614, 0.1354112, -0.2364632, 0.2399184
7: 0.8405428, 1.0085205, 0.7502919, 1.0418952, -0.2013524, 0.2582285
8: -0.0862659, 0.0943539, -0.1477194, 0.1700793, -0.2563452, 0.2420732
9: -0.0780512, 0.0893186, -0.1235805, 0.2064950, -0.2845462, 0.2128990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 75

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0346038, 0.0329501, -0.0577739, 0.0441374, -0.0787412, 0.0907240
1: -0.0329297, 0.0291781, -0.0528996, 0.0506039, -0.0835336, 0.0820777
2: -0.0299686, 0.1144697, -0.0566953, 0.1433618, -0.1733304, 0.1711650
3: -0.0169107, 0.0431368, -0.0344549, 0.0605272, -0.0774379, 0.0775917
4: -0.0443248, 0.0387755, -0.0649224, 0.0650784, -0.1094031, 0.1036979
5: -0.0191343, 0.0501974, -0.0400879, 0.0753642, -0.0944985, 0.0902853
6: -0.0825459, 0.0540066, -0.1083899, 0.0797313, -0.1622772, 0.1623965
7: 0.8692628, 0.9991268, 0.8306333, 1.0121783, -0.1429155, 0.1684935
8: -0.0715315, 0.0758695, -0.0936606, 0.1031192, -0.1746507, 0.1695301
9: -0.0631176, 0.0635544, -0.0828703, 0.1034345, -0.1665521, 0.1464247

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0382337, 0.0349082, -0.0638731, 0.0472754, -0.0855090, 0.0987813
1: -0.0352899, 0.0334162, -0.0589926, 0.0555515, -0.0908414, 0.0924088
2: -0.0350461, 0.1193184, -0.0633352, 0.1508691, -0.1859152, 0.1826536
3: -0.0193006, 0.0453436, -0.0402251, 0.0656145, -0.0849151, 0.0855687
4: -0.0477261, 0.0421063, -0.0709961, 0.0726991, -0.1204252, 0.1131023
5: -0.0213434, 0.0547345, -0.0462777, 0.0822200, -0.1035634, 0.1010122
6: -0.0866712, 0.0585695, -0.1154618, 0.0864426, -0.1731138, 0.1740313
7: 0.8620061, 1.0012894, 0.8209495, 1.0157602, -0.1537541, 0.1803399
8: -0.0752269, 0.0799765, -0.1001766, 0.1111901, -0.1864171, 0.1801530
9: -0.0667498, 0.0689430, -0.0877773, 0.1158568, -0.1826066, 0.1567203

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 153

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0346038, 0.0329501, -0.1065076, 0.0692097, -0.1038134, 0.1394577
1: -0.0329297, 0.0291781, -0.1015837, 0.0901363, -0.1230660, 0.1307618
2: -0.0299686, 0.1144697, -0.1097496, 0.2033462, -0.2333149, 0.2242193
3: -0.0169107, 0.0431368, -0.0805596, 0.1011751, -0.1180858, 0.1236964
4: -0.0443248, 0.0387755, -0.1134520, 0.1259689, -0.1702937, 0.1522275
5: -0.0191343, 0.0501974, -0.0895450, 0.1301437, -0.1492779, 0.1397423
6: -0.0825459, 0.0540066, -0.1648952, 0.1333555, -0.2159013, 0.2189018
7: 0.8692628, 0.9991268, 0.7532583, 1.0407979, -0.1715352, 0.2458685
8: -0.0715315, 0.0758695, -0.1457234, 0.1676070, -0.2391386, 0.2215928
9: -0.0631176, 0.0635544, -0.1220774, 0.2026898, -0.2658074, 0.1856318

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0382337, 0.0349082, -0.1134579, 0.0727854, -0.1110191, 0.1483661
1: -0.0352899, 0.0334162, -0.1085269, 0.0957743, -0.1310642, 0.1419431
2: -0.0350461, 0.1193184, -0.1173161, 0.2119010, -0.2469471, 0.2366345
3: -0.0193006, 0.0453436, -0.0871348, 0.1069722, -0.1262729, 0.1324785
4: -0.0477261, 0.0421063, -0.1203731, 0.1346529, -0.1823789, 0.1624794
5: -0.0213434, 0.0547345, -0.0965984, 0.1379562, -0.1592995, 0.1513329
6: -0.0866712, 0.0585695, -0.1729538, 0.1410031, -0.2276744, 0.2315233
7: 0.8620061, 1.0012894, 0.7422233, 1.0448796, -0.1828735, 0.2590661
8: -0.0752269, 0.0799765, -0.1531484, 0.1768041, -0.2520310, 0.2331249
9: -0.0667498, 0.0689430, -0.1276690, 0.2168453, -0.2835952, 0.1966120

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 75

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0760827, 0.0535569, -0.0463823, 0.0385122, -0.1145950, 0.0999392
1: -0.0711899, 0.0654559, -0.0417190, 0.0412170, -0.1124069, 0.1071748
2: -0.0766274, 0.1658975, -0.0444135, 0.1294430, -0.2060704, 0.2103109
3: -0.0517761, 0.0757983, -0.0245072, 0.0508755, -0.1026515, 0.1003055
4: -0.0831546, 0.0879545, -0.0541208, 0.0512034, -0.1343580, 0.1420753
5: -0.0586685, 0.0959444, -0.0285512, 0.0630857, -0.1217542, 0.1244956
6: -0.1296185, 0.0998775, -0.0955314, 0.0673081, -0.1969266, 0.1954089
7: 0.8015640, 1.0229304, 0.8485657, 1.0056076, -0.2040436, 0.1743647
8: -0.1132203, 0.1273467, -0.0820998, 0.0888312, -0.2020515, 0.2094465
9: -0.0976001, 0.1407242, -0.0736784, 0.0814069, -0.1790070, 0.2144026

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_B1_A1_B1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.96 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_B2

### Relational analysis result of IS_A2_B1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0743655, 0.0526734, -0.0575690, 0.0429875, -0.1173530, 0.1102424
1: -0.0694743, 0.0640629, -0.0518550, 0.0509034, -0.1203777, 0.1159179
2: -0.0747579, 0.1637838, -0.0560901, 0.1424923, -0.2172502, 0.2198739
3: -0.0501515, 0.0743659, -0.0312939, 0.0608352, -0.1109866, 0.1056598
4: -0.0814445, 0.0858089, -0.0623431, 0.0636804, -0.1451249, 0.1481520
5: -0.0569258, 0.0940141, -0.0396192, 0.0734556, -0.1303815, 0.1336333
6: -0.1276275, 0.0979879, -0.1070376, 0.0788734, -0.2065009, 0.2050255
7: 0.8042907, 1.0219220, 0.8318438, 1.0116788, -0.2073882, 0.1900782
8: -0.1113857, 0.1250744, -0.0907830, 0.1003419, -0.2117276, 0.2158574
9: -0.0962186, 0.1372265, -0.0827923, 0.0978968, -0.1941154, 0.2200188

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_B1_A1_B2_B1

### Relational analysis result of IS_A2_B1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.98 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2_B2

### Relational analysis result of IS_A2_B1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0829300, 0.0570796, -0.0522820, 0.0408724, -0.1238024, 0.1093616
1: -0.0780301, 0.0710103, -0.0470645, 0.0463254, -0.1243555, 0.1180748
2: -0.0840817, 0.1743255, -0.0505715, 0.1363249, -0.2204065, 0.2248970
3: -0.0582539, 0.0815094, -0.0280863, 0.0561280, -0.1143819, 0.1095958
4: -0.0899732, 0.0965098, -0.0584571, 0.0577835, -0.1477566, 0.1549669
5: -0.0656174, 0.1036411, -0.0343881, 0.0685546, -0.1341721, 0.1380292
6: -0.1375577, 0.1074119, -0.1015995, 0.0734074, -0.2109651, 0.2090113
7: 0.7906926, 1.0269517, 0.8397470, 1.0088093, -0.2181168, 0.1872047
8: -0.1205353, 0.1364076, -0.0866791, 0.0949017, -0.2154369, 0.2230867
9: -0.1031088, 0.1546698, -0.0784849, 0.0901033, -0.1932121, 0.2331547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 153

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.97 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_B2

### Relational analysis result of IS_A2_B1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0809565, 0.0560643, -0.0627942, 0.0450778, -0.1260343, 0.1188585
1: -0.0760586, 0.0694094, -0.0565894, 0.0554278, -0.1314864, 0.1259989
2: -0.0819332, 0.1718963, -0.0615442, 0.1485875, -0.2305207, 0.2334405
3: -0.0563868, 0.0798634, -0.0344638, 0.0654873, -0.1218741, 0.1143272
4: -0.0880079, 0.0940439, -0.0661837, 0.0695083, -0.1575161, 0.1602277
5: -0.0636146, 0.1014227, -0.0447889, 0.0782993, -0.1419140, 0.1462116
6: -0.1352694, 0.1052403, -0.1124120, 0.0842755, -0.2195449, 0.2176523
7: 0.7938261, 1.0257926, 0.8240331, 1.0145149, -0.2206888, 0.2017595
8: -0.1184269, 0.1337960, -0.0948388, 0.1057184, -0.2241453, 0.2286348
9: -0.1015211, 0.1506503, -0.0870494, 0.1055992, -0.2071202, 0.2376997

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 153

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_B1_A2_B2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.97 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0866704, 0.0590040, -0.0317805, 0.0314272, -0.1180976, 0.0907845
1: -0.0817667, 0.0740445, -0.0310940, 0.0258818, -0.1076485, 0.1051385
2: -0.0881537, 0.1789294, -0.0260195, 0.1106984, -0.1988521, 0.2049489
3: -0.0617926, 0.0846293, -0.0150519, 0.0414204, -0.1032130, 0.0996812
4: -0.0936979, 0.1011833, -0.0416792, 0.0361850, -0.1298828, 0.1428625
5: -0.0694134, 0.1078455, -0.0174161, 0.0466685, -0.1160819, 0.1252616
6: -0.1418946, 0.1115276, -0.0793373, 0.0504577, -0.1923522, 0.1908649
7: 0.7847539, 1.0291483, 0.8749068, 0.9974450, -0.2126911, 0.1542416
8: -0.1245312, 0.1413572, -0.0686573, 0.0726751, -0.1972064, 0.2100144
9: -0.1061181, 0.1622878, -0.0602925, 0.0593633, -0.1654813, 0.2225803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 153

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_B2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.99 seconds

## Relational analysis of IS_A2_B1_B2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.99 seconds

## BFS IS instance: IS_A2_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0846710, 0.0579754, -0.0398620, 0.0357865, -0.1204575, 0.0978373
1: -0.0797694, 0.0724227, -0.0363486, 0.0353173, -0.1150867, 0.1087713
2: -0.0859771, 0.1764685, -0.0373238, 0.1214935, -0.2074706, 0.2137923
3: -0.0599010, 0.0829616, -0.0203727, 0.0463335, -0.1062346, 0.1033344
4: -0.0917069, 0.0986852, -0.0492518, 0.0436004, -0.1353073, 0.1479370
5: -0.0673843, 0.1055981, -0.0223343, 0.0567698, -0.1241541, 0.1279324
6: -0.1395764, 0.1093277, -0.0885218, 0.0606163, -0.2001927, 0.1978495
7: 0.7879282, 1.0279741, 0.8587507, 1.0022595, -0.2143313, 0.1692234
8: -0.1223952, 0.1387114, -0.0768846, 0.0818188, -0.2042140, 0.2155960
9: -0.1045095, 0.1582157, -0.0683792, 0.0713602, -0.1758697, 0.2265949

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 153

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_B2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.97 seconds

## Relational analysis of IS_A2_B1_B2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.13 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1104169, 0.0712209, -0.0396248, 0.0356585, -0.1460755, 0.1108457
1: -0.1054891, 0.0933075, -0.0361944, 0.0350403, -0.1405294, 0.1295019
2: -0.1140055, 0.2081579, -0.0369920, 0.1211766, -0.2351821, 0.2451499
3: -0.0842580, 0.1044358, -0.0202165, 0.0461893, -0.1304473, 0.1246523
4: -0.1173449, 0.1308533, -0.0490296, 0.0433827, -0.1607276, 0.1798829
5: -0.0935123, 0.1345380, -0.0221900, 0.0564732, -0.1499855, 0.1567279
6: -0.1694279, 0.1376570, -0.0882522, 0.0603181, -0.2297461, 0.2259092
7: 0.7470515, 1.0430937, 0.8592249, 1.0021183, -0.2550669, 0.1838688
8: -0.1498998, 0.1727801, -0.0766431, 0.0815504, -0.2314502, 0.2494232
9: -0.1252225, 0.2106519, -0.0681418, 0.0710080, -0.1962305, 0.2787937

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 75

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_B2_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.92 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0940937, 0.0628231, -0.0396248, 0.0356585, -0.1297522, 0.1024478
1: -0.0891825, 0.0800662, -0.0361944, 0.0350403, -0.1242228, 0.1162607
2: -0.0962352, 0.1880664, -0.0369920, 0.1211766, -0.2174118, 0.2250584
3: -0.0688154, 0.0908209, -0.0202165, 0.0461893, -0.1150047, 0.1110374
4: -0.1010901, 0.1104583, -0.0490296, 0.0433827, -0.1444728, 0.1594879
5: -0.0769468, 0.1161898, -0.0221900, 0.0564732, -0.1334201, 0.1383797
6: -0.1505017, 0.1196958, -0.0882522, 0.0603181, -0.2108198, 0.2079480
7: 0.7729679, 1.0335077, 0.8592249, 1.0021183, -0.2291504, 0.1742828
8: -0.1324616, 0.1511801, -0.0766431, 0.0815504, -0.2140120, 0.2278233
9: -0.1120902, 0.1774067, -0.0681418, 0.0710080, -0.1830983, 0.2455485

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 153

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_B2_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.04 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0697073, 0.0502769, -0.0725220, 0.0517250, -0.1214323, 0.1227989
1: -0.0648209, 0.0602842, -0.0676327, 0.0625674, -0.1273883, 0.1279169
2: -0.0696867, 0.1580502, -0.0727509, 0.1615147, -0.2312014, 0.2308011
3: -0.0457446, 0.0704806, -0.0484074, 0.0728283, -0.1185729, 0.1188880
4: -0.0768059, 0.0799887, -0.0796087, 0.0835055, -0.1603113, 0.1595974
5: -0.0521985, 0.0887780, -0.0550549, 0.0919419, -0.1441404, 0.1438330
6: -0.1222264, 0.0928623, -0.1254899, 0.0959594, -0.2181858, 0.2183522
7: 0.8116863, 1.0191865, 0.8072176, 1.0208393, -0.2091530, 0.2119689
8: -0.1064093, 0.1189104, -0.1094162, 0.1226349, -0.2290443, 0.2283266
9: -0.0924710, 0.1277393, -0.0947354, 0.1334718, -0.2259428, 0.2224747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_B1_A1_A1_A1

### Relational analysis result of IS_A2_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.99 seconds

## Relational analysis of IS_A2_B2_B1_A1_A1_A2

### Relational analysis result of IS_A2_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0803279, 0.0557409, -0.0708029, 0.0508405, -0.1311684, 0.1265438
1: -0.0754306, 0.0688995, -0.0659153, 0.0611729, -0.1366036, 0.1348148
2: -0.0812488, 0.1711226, -0.0708794, 0.1593987, -0.2406476, 0.2420020
3: -0.0557922, 0.0793391, -0.0467810, 0.0713945, -0.1271867, 0.1261202
4: -0.0873819, 0.0932586, -0.0778968, 0.0813575, -0.1687394, 0.1711554
5: -0.0629767, 0.1007161, -0.0533103, 0.0900095, -0.1529862, 0.1540265
6: -0.1345406, 0.1045487, -0.1234967, 0.0940678, -0.2286084, 0.2280453
7: 0.7948239, 1.0254234, 0.8099469, 1.0198298, -0.2250058, 0.2154765
8: -0.1177554, 0.1329643, -0.1075798, 0.1203601, -0.2381155, 0.2405441
9: -0.1010154, 0.1493700, -0.0933524, 0.1299706, -0.2309860, 0.2427224

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_B1_A1_A2_A1

### Relational analysis result of IS_A2_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.96 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2_A2

### Relational analysis result of IS_A2_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 2.17 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0761687, 0.0536011, -0.0791226, 0.0551208, -0.1312895, 0.1327237
1: -0.0712757, 0.0655256, -0.0742266, 0.0679219, -0.1391976, 0.1397522
2: -0.0767209, 0.1660032, -0.0799369, 0.1696392, -0.2463601, 0.2459401
3: -0.0518574, 0.0758699, -0.0546520, 0.0783339, -0.1301912, 0.1305219
4: -0.0832402, 0.0880618, -0.0861817, 0.0917527, -0.1749929, 0.1742436
5: -0.0587558, 0.0960410, -0.0617536, 0.0993614, -0.1581171, 0.1577946
6: -0.1297181, 0.0999721, -0.1331431, 0.1032225, -0.2329406, 0.2331152
7: 0.8014275, 1.0229809, 0.7967376, 1.0247157, -0.2232881, 0.2262434
8: -0.1133121, 0.1274604, -0.1164678, 0.1313694, -0.2446815, 0.2439283
9: -0.0976692, 0.1408991, -0.1000458, 0.1469153, -0.2445845, 0.2409449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_B1_A2_A1_A1

### Relational analysis result of IS_A2_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.93 seconds

## Relational analysis of IS_A2_B2_B1_A2_A1_A2

### Relational analysis result of IS_A2_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.96 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0862156, 0.0587700, -0.0770778, 0.0540688, -0.1402844, 0.1358477
1: -0.0813124, 0.0736756, -0.0721838, 0.0662631, -0.1475755, 0.1458595
2: -0.0876587, 0.1783697, -0.0777106, 0.1671222, -0.2547809, 0.2560803
3: -0.0613623, 0.0842500, -0.0527174, 0.0766283, -0.1379906, 0.1369674
4: -0.0932450, 0.1006150, -0.0841455, 0.0891977, -0.1824427, 0.1847605
5: -0.0689519, 0.1073343, -0.0596784, 0.0970628, -0.1660147, 0.1670126
6: -0.1413673, 0.1110272, -0.1307722, 0.1009724, -0.2423397, 0.2417994
7: 0.7854759, 1.0288812, 0.7999843, 1.0235150, -0.2380391, 0.2288969
8: -0.1240453, 0.1407554, -0.1142833, 0.1286635, -0.2527088, 0.2550386
9: -0.1057522, 0.1613615, -0.0984007, 0.1427506, -0.2485029, 0.2597622

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_B1_A2_A2_A1

### Relational analysis result of IS_A2_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.96 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2_A2

### Relational analysis result of IS_A2_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.08 seconds

## BFS IS instance: IS_A2_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1017453, 0.0667596, -0.0541969, 0.0422972, -0.1440425, 0.1209565
1: -0.0968262, 0.0862731, -0.0493262, 0.0477023, -0.1445285, 0.1355994
2: -0.1045651, 0.1974844, -0.0528011, 0.1389591, -0.2435242, 0.2502855
3: -0.0760541, 0.0972029, -0.0310709, 0.0575437, -0.1335978, 0.1282738
4: -0.1087096, 0.1200186, -0.0613604, 0.0606092, -0.1693187, 0.1813791
5: -0.0847120, 0.1247905, -0.0364579, 0.0713434, -0.1560554, 0.1612484
6: -0.1593734, 0.1281152, -0.1042426, 0.0757954, -0.2351688, 0.2323578
7: 0.7608195, 1.0380011, 0.8363126, 1.0100776, -0.2492581, 0.2016885
8: -0.1406358, 0.1613052, -0.0898394, 0.0983859, -0.2390217, 0.2511446
9: -0.1182460, 0.1929905, -0.0799926, 0.0961494, -0.2143954, 0.2729831

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 75

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_B2_B1_A1_B1

### Relational analysis result of IS_A2_B2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.97 seconds

## Relational analysis of IS_A2_B2_B2_B1_A1_B2

### Relational analysis result of IS_A2_B2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0852520, 0.0582742, -0.0541969, 0.0422972, -0.1275492, 0.1124711
1: -0.0803497, 0.0728939, -0.0493262, 0.0477023, -0.1280520, 0.1222202
2: -0.0866095, 0.1771835, -0.0528011, 0.1389591, -0.2255686, 0.2299846
3: -0.0604507, 0.0834462, -0.0310709, 0.0575437, -0.1179944, 0.1145171
4: -0.0922854, 0.0994110, -0.0613604, 0.0606092, -0.1528946, 0.1607714
5: -0.0679739, 0.1062511, -0.0364579, 0.0713434, -0.1393173, 0.1427090
6: -0.1402499, 0.1099669, -0.1042426, 0.0757954, -0.2160453, 0.2142095
7: 0.7870061, 1.0283153, 0.8363126, 1.0100776, -0.2230715, 0.1920027
8: -0.1230159, 0.1394801, -0.0898394, 0.0983859, -0.2214018, 0.2293195
9: -0.1049769, 0.1593989, -0.0799926, 0.0961494, -0.2011263, 0.2393915

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 75

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_B2_B1_A2_B1

### Relational analysis result of IS_A2_B2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.95 seconds

## Relational analysis of IS_A2_B2_B2_B1_A2_B2

### Relational analysis result of IS_A2_B2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1084586, 0.0702134, -0.0596541, 0.0451048, -0.1535634, 0.1298675
1: -0.1035328, 0.0917189, -0.0547779, 0.0521291, -0.1556619, 0.1464968
2: -0.1118737, 0.2057476, -0.0587422, 0.1456761, -0.2575498, 0.2644898
3: -0.0824053, 0.1028024, -0.0362337, 0.0620955, -0.1445007, 0.1390361
4: -0.1153948, 0.1284066, -0.0667948, 0.0674276, -0.1828224, 0.1952013
5: -0.0915250, 0.1323368, -0.0419961, 0.0774776, -0.1690026, 0.1743328
6: -0.1671573, 0.1355023, -0.1105699, 0.0818003, -0.2489576, 0.2460722
7: 0.7501606, 1.0419438, 0.8276480, 1.0132827, -0.2631221, 0.2142957
8: -0.1478078, 0.1701887, -0.0956694, 0.1056072, -0.2534150, 0.2658581
9: -0.1236470, 0.2066636, -0.0843830, 0.1072640, -0.2309110, 0.2910466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 75

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_B2_B2_A1_B1

### Relational analysis result of IS_A2_B2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.85 seconds

## Relational analysis of IS_A2_B2_B2_B2_A1_B2

### Relational analysis result of IS_A2_B2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0922607, 0.0618800, -0.0596541, 0.0451048, -0.1373655, 0.1215341
1: -0.0873513, 0.0785793, -0.0547779, 0.0521291, -0.1394804, 0.1333572
2: -0.0942397, 0.1858103, -0.0587422, 0.1456761, -0.2399157, 0.2445525
3: -0.0670813, 0.0892920, -0.0362337, 0.0620955, -0.1291767, 0.1255257
4: -0.0992647, 0.1081681, -0.0667948, 0.0674276, -0.1666923, 0.1749629
5: -0.0750866, 0.1141294, -0.0419961, 0.0774776, -0.1525643, 0.1561254
6: -0.1483764, 0.1176789, -0.1105699, 0.0818003, -0.2301767, 0.2282488
7: 0.7758781, 1.0324312, 0.8276480, 1.0132827, -0.2374046, 0.2047832
8: -0.1305033, 0.1487546, -0.0956694, 0.1056072, -0.2361105, 0.2444240
9: -0.1106155, 0.1736735, -0.0843830, 0.1072640, -0.2178795, 0.2580565

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 75

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_B2_B2_A2_B1

### Relational analysis result of IS_A2_B2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.90 seconds

## Relational analysis of IS_A2_B2_B2_B2_A2_B2

### Relational analysis result of IS_A2_B2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.93 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.81 seconds
IS_A1_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207

## BFS IS instance: IS_A1_B1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0294198, 0.0301538, -0.0476308, 0.0390117, -0.0684315, 0.0777845
1: -0.0295590, 0.0231255, -0.0428502, 0.0422980, -0.0718571, 0.0659757
2: -0.0227173, 0.1075449, -0.0457166, 0.1308993, -0.1536165, 0.1532615
3: -0.0134976, 0.0399852, -0.0252646, 0.0519870, -0.0654846, 0.0652498
4: -0.0394671, 0.0340187, -0.0550384, 0.0525959, -0.0920630, 0.0890571
5: -0.0159794, 0.0437177, -0.0297863, 0.0642431, -0.0802225, 0.0735041
6: -0.0766543, 0.0474901, -0.0968155, 0.0685988, -0.1452531, 0.1443057
7: 0.8796264, 0.9960386, 0.8466997, 1.0062850, -0.1266586, 0.1493388
8: -0.0662539, 0.0700041, -0.0830689, 0.0901158, -0.1563697, 0.1530730
9: -0.0579302, 0.0558587, -0.0746955, 0.0832471, -0.1411774, 0.1305543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_B1_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.12 seconds

## Relational analysis of IS_A1_B1_B1_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0375094, 0.0345175, -0.0461581, 0.0384225, -0.0759319, 0.0806756
1: -0.0348191, 0.0325706, -0.0415157, 0.0410228, -0.0758419, 0.0740864
2: -0.0340331, 0.1183510, -0.0441794, 0.1291814, -0.1632145, 0.1625304
3: -0.0188238, 0.0449034, -0.0243712, 0.0506758, -0.0694996, 0.0692745
4: -0.0470474, 0.0414417, -0.0539560, 0.0509532, -0.0980006, 0.0953977
5: -0.0209026, 0.0538292, -0.0283292, 0.0628779, -0.0837805, 0.0821584
6: -0.0858482, 0.0576591, -0.0953007, 0.0670762, -0.1529244, 0.1529599
7: 0.8634538, 1.0008579, 0.8489010, 1.0054857, -0.1420318, 0.1519570
8: -0.0744896, 0.0791571, -0.0819257, 0.0886004, -0.1630901, 0.1610828
9: -0.0660252, 0.0678678, -0.0734957, 0.0810763, -0.1471014, 0.1413636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_B1_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.98 seconds

## Relational analysis of IS_A1_B1_B1_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0294198, 0.0301538, -0.0952486, 0.0634172, -0.0928370, 0.1254024
1: -0.0295590, 0.0231255, -0.0903362, 0.0810031, -0.1105621, 0.1134617
2: -0.0227173, 0.1075449, -0.0974924, 0.1894879, -0.2122052, 0.2050374
3: -0.0134976, 0.0399852, -0.0699079, 0.0917842, -0.1052818, 0.1098932
4: -0.0394671, 0.0340187, -0.1022402, 0.1119013, -0.1513685, 0.1362589
5: -0.0159794, 0.0437177, -0.0781189, 0.1174879, -0.1334673, 0.1218366
6: -0.0766543, 0.0474901, -0.1518407, 0.1209666, -0.1976209, 0.1993308
7: 0.8796264, 0.9960386, 0.7711343, 1.0341859, -0.1545595, 0.2249043
8: -0.0662539, 0.0700041, -0.1336954, 0.1527083, -0.2189622, 0.2036994
9: -0.0579302, 0.0558587, -0.1130193, 0.1797588, -0.2376891, 0.1688781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_A1

### Relational analysis result of IS_A1_B1_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.97 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_A2

### Relational analysis result of IS_A1_B1_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0375094, 0.0345175, -0.0936636, 0.0626018, -0.1001112, 0.1281811
1: -0.0348191, 0.0325706, -0.0887529, 0.0797174, -0.1145364, 0.1213235
2: -0.0340331, 0.1183510, -0.0957669, 0.1875371, -0.2215702, 0.2141179
3: -0.0188238, 0.0449034, -0.0684085, 0.0904622, -0.1092860, 0.1133118
4: -0.0470474, 0.0414417, -0.1006618, 0.1099210, -0.1569684, 0.1421035
5: -0.0209026, 0.0538292, -0.0765104, 0.1157063, -0.1366089, 0.1303396
6: -0.0858482, 0.0576591, -0.1500030, 0.1192226, -0.2050707, 0.2076621
7: 0.8634538, 1.0008579, 0.7736508, 1.0332551, -0.1698013, 0.2272072
8: -0.0744896, 0.0791571, -0.1320021, 0.1506110, -0.2251006, 0.2111592
9: -0.0660252, 0.0678678, -0.1117442, 0.1765308, -0.2425559, 0.1796121

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_A1

### Relational analysis result of IS_A1_B1_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.16 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_A2

### Relational analysis result of IS_A1_B1_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0218520, 0.0235853, -0.0441933, 0.0376365, -0.0594885, 0.0677786
1: -0.0232744, 0.0132145, -0.0397355, 0.0393216, -0.0625959, 0.0529500
2: -0.0098144, 0.0912792, -0.0421286, 0.1268895, -0.1367039, 0.1334079
3: -0.0085099, 0.0325823, -0.0231792, 0.0489266, -0.0574364, 0.0557615
4: -0.0290609, 0.0270755, -0.0525119, 0.0487619, -0.0778228, 0.0795874
5: -0.0102054, 0.0356282, -0.0263853, 0.0610565, -0.0712619, 0.0620135
6: -0.0649410, 0.0345573, -0.0932799, 0.0650450, -0.1299859, 0.1278372
7: 0.9020013, 0.9903364, 0.8518379, 1.0044192, -0.1024179, 0.1384984
8: -0.0551115, 0.0563749, -0.0804007, 0.0865788, -0.1416903, 0.1367756
9: -0.0462236, 0.0433772, -0.0718950, 0.0781801, -0.1244037, 0.1152722

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A2_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.08 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_A1_A2

### Relational analysis result of IS_A1_B1_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0302481, 0.0306006, -0.0534971, 0.0419372, -0.0721852, 0.0840977
1: -0.0300976, 0.0240927, -0.0486271, 0.0471345, -0.0772322, 0.0727198
2: -0.0238759, 0.1086514, -0.0520392, 0.1380976, -0.1619735, 0.1606907
3: -0.0140430, 0.0404888, -0.0304088, 0.0569600, -0.0710029, 0.0708976
4: -0.0402433, 0.0347788, -0.0606635, 0.0597347, -0.0999780, 0.0954423
5: -0.0164835, 0.0447531, -0.0357476, 0.0705568, -0.0870403, 0.0805007
6: -0.0775957, 0.0485314, -0.1034310, 0.0750253, -0.1526210, 0.1519624
7: 0.8779705, 0.9965320, 0.8374237, 1.0096666, -0.1316961, 0.1591083
8: -0.0670972, 0.0709414, -0.0890918, 0.0974598, -0.1645570, 0.1600331
9: -0.0587591, 0.0570884, -0.0794295, 0.0947241, -0.1534832, 0.1365179

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A2_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.91 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_A2_A2

### Relational analysis result of IS_A1_B1_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0329732, 0.0320705, -0.1039060, 0.0678712, -0.1008444, 0.1359765
1: -0.0318695, 0.0272743, -0.0989847, 0.0880259, -0.1198954, 0.1262590
2: -0.0276878, 0.1122915, -0.1069174, 0.2001439, -0.2278317, 0.2192089
3: -0.0158371, 0.0421455, -0.0780982, 0.0990051, -0.1148422, 0.1202437
4: -0.0427968, 0.0372793, -0.1108612, 0.1227183, -0.1655151, 0.1481405
5: -0.0181419, 0.0481592, -0.0869047, 0.1272192, -0.1453612, 0.1350639
6: -0.0806927, 0.0519569, -0.1618787, 0.1304927, -0.2111855, 0.2138356
7: 0.8725226, 0.9981555, 0.7573889, 1.0392700, -0.1667475, 0.2407666
8: -0.0698715, 0.0740246, -0.1429441, 0.1641644, -0.2340359, 0.2169687
9: -0.0614860, 0.0611338, -0.1199843, 0.1973911, -0.2588771, 0.1811181

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=54, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_A1

### Relational analysis result of IS_A1_B1_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.05 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_A2

### Relational analysis result of IS_A1_B1_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0410427, 0.0364234, -0.1020784, 0.0669310, -0.1079737, 0.1385019
1: -0.0371164, 0.0366958, -0.0971590, 0.0865434, -0.1236598, 0.1338549
2: -0.0389754, 0.1230707, -0.1049278, 0.1978945, -0.2368699, 0.2279985
3: -0.0211501, 0.0470514, -0.0763693, 0.0974808, -0.1186309, 0.1234207
4: -0.0503582, 0.0446838, -0.1090413, 0.1204348, -0.1707930, 0.1537252
5: -0.0230529, 0.0582456, -0.0850500, 0.1251650, -0.1482179, 0.1432956
6: -0.0898637, 0.0621005, -0.1597597, 0.1284819, -0.2183455, 0.2218602
7: 0.8563903, 1.0029631, 0.7602904, 1.0381968, -0.1818065, 0.2426727
8: -0.0780866, 0.0831547, -0.1409917, 0.1617461, -0.2398327, 0.2241464
9: -0.0695607, 0.0731130, -0.1185140, 0.1936690, -0.2632297, 0.1916270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=54, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_A1

### Relational analysis result of IS_A1_B1_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.59 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_A2

### Relational analysis result of IS_A1_B1_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0467484, 0.0395011, -0.0370343, 0.0342612, -0.0810096, 0.0765355
1: -0.0408262, 0.0433575, -0.0345101, 0.0320159, -0.0728421, 0.0778676
2: -0.0469565, 0.1306922, -0.0333685, 0.1177163, -0.1646729, 0.1640607
3: -0.0249066, 0.0505202, -0.0185110, 0.0446145, -0.0695211, 0.0690311
4: -0.0557046, 0.0499193, -0.0466022, 0.0410058, -0.0967104, 0.0965215
5: -0.0265252, 0.0653773, -0.0206135, 0.0532354, -0.0797606, 0.0859907
6: -0.0963481, 0.0692727, -0.0853081, 0.0570618, -0.1534099, 0.1545808
7: 0.8449838, 1.0063621, 0.8644037, 1.0005748, -0.1555911, 0.1419584
8: -0.0838953, 0.0896103, -0.0740059, 0.0786195, -0.1625148, 0.1636162
9: -0.0752701, 0.0815831, -0.0655497, 0.0671625, -0.1424326, 0.1471328

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.91 seconds

## Relational analysis of IS_A1_B1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0357911, 0.0335906, -0.0370343, 0.0342612, -0.0700523, 0.0706249
1: -0.0337018, 0.0305644, -0.0345101, 0.0320159, -0.0657176, 0.0650745
2: -0.0316295, 0.1160557, -0.0333685, 0.1177163, -0.1493458, 0.1494243
3: -0.0176925, 0.0438587, -0.0185110, 0.0446145, -0.0623070, 0.0623697
4: -0.0454373, 0.0398650, -0.0466022, 0.0410058, -0.0864431, 0.0864673
5: -0.0198569, 0.0516815, -0.0206135, 0.0532354, -0.0730923, 0.0722950
6: -0.0838953, 0.0554992, -0.0853081, 0.0570618, -0.1409571, 0.1408073
7: 0.8668889, 0.9998344, 0.8644037, 1.0005748, -0.1336859, 0.1354306
8: -0.0727403, 0.0772129, -0.0740059, 0.0786195, -0.1513598, 0.1512188
9: -0.0643057, 0.0653170, -0.0655497, 0.0671625, -0.1314683, 0.1308668

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_B2_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.89 seconds

## Relational analysis of IS_A1_B1_B2_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0467484, 0.0395011, -0.0772365, 0.0521253, -0.0988737, 0.1167377
1: -0.0408262, 0.0433575, -0.0638791, 0.0706818, -0.1115080, 0.1072366
2: -0.0469565, 0.1306922, -0.0796931, 0.1691374, -0.2160939, 0.2103853
3: -0.0249066, 0.0505202, -0.0451513, 0.0647481, -0.0896548, 0.0956714
4: -0.0557046, 0.0499193, -0.0776341, 0.0891567, -0.1448613, 0.1275534
5: -0.0265252, 0.0653773, -0.0565333, 0.0946296, -0.1211548, 0.1219106
6: -0.0963481, 0.0692727, -0.1305317, 0.0986913, -0.1950393, 0.1998044
7: 0.8449838, 1.0063621, 0.7976997, 1.0203046, -0.1753208, 0.2086623
8: -0.0838953, 0.0896103, -0.1077210, 0.1238451, -0.2077405, 0.1973313
9: -0.0752701, 0.0815831, -0.0986885, 0.1315671, -0.2068372, 0.1802716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_A1_B2_A1_A1

### Relational analysis result of IS_A1_B1_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.03 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2_A1_A2

### Relational analysis result of IS_A1_B1_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0357911, 0.0335906, -0.0772365, 0.0521253, -0.0879164, 0.1108272
1: -0.0337018, 0.0305644, -0.0638791, 0.0706818, -0.1043836, 0.0944435
2: -0.0316295, 0.1160557, -0.0796931, 0.1691374, -0.2007669, 0.1957488
3: -0.0176925, 0.0438587, -0.0451513, 0.0647481, -0.0824406, 0.0890100
4: -0.0454373, 0.0398650, -0.0776341, 0.0891567, -0.1345940, 0.1174992
5: -0.0198569, 0.0516815, -0.0565333, 0.0946296, -0.1144865, 0.1082148
6: -0.0838953, 0.0554992, -0.1305317, 0.0986913, -0.1825866, 0.1860308
7: 0.8668889, 0.9998344, 0.7976997, 1.0203046, -0.1534156, 0.2021346
8: -0.0727403, 0.0772129, -0.1077210, 0.1238451, -0.1965854, 0.1849339
9: -0.0643057, 0.0653170, -0.0986885, 0.1315671, -0.1958728, 0.1640055

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_A1_B2_A2_A1

### Relational analysis result of IS_A1_B1_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.01 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2_A2_A2

### Relational analysis result of IS_A1_B1_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0523262, 0.0412350, -0.0414809, 0.0364936, -0.0888198, 0.0827159
1: -0.0455303, 0.0471103, -0.0375417, 0.0368478, -0.0823781, 0.0846519
2: -0.0514526, 0.1373822, -0.0391575, 0.1235569, -0.1750095, 0.1765397
3: -0.0286362, 0.0524742, -0.0214460, 0.0471305, -0.0757668, 0.0739203
4: -0.0587164, 0.0587944, -0.0504802, 0.0455755, -0.1042919, 0.1092746
5: -0.0337407, 0.0693948, -0.0238175, 0.0584083, -0.0921490, 0.0932123
6: -0.1025318, 0.0733131, -0.0903414, 0.0622641, -0.1647959, 0.1636545
7: 0.8383921, 1.0082769, 0.8561085, 1.0030406, -0.1646485, 0.1521684
8: -0.0871676, 0.0958343, -0.0782192, 0.0836391, -0.1708067, 0.1740535
9: -0.0784864, 0.0914394, -0.0696909, 0.0739688, -0.1524552, 0.1611303

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.94 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0393704, 0.0355213, -0.0414809, 0.0364936, -0.0758640, 0.0770022
1: -0.0360290, 0.0347434, -0.0375417, 0.0368478, -0.0728768, 0.0722850
2: -0.0366361, 0.1208368, -0.0391575, 0.1235569, -0.1601930, 0.1599943
3: -0.0200490, 0.0460347, -0.0214460, 0.0471305, -0.0671796, 0.0674807
4: -0.0487912, 0.0431493, -0.0504802, 0.0455755, -0.0943667, 0.0936295
5: -0.0220351, 0.0561553, -0.0238175, 0.0584083, -0.0804434, 0.0799728
6: -0.0879631, 0.0599984, -0.0903414, 0.0622641, -0.1502272, 0.1503398
7: 0.8597335, 1.0019666, 0.8561085, 1.0030406, -0.1433071, 0.1458580
8: -0.0763842, 0.0812626, -0.0782192, 0.0836391, -0.1600233, 0.1594817
9: -0.0678873, 0.0706304, -0.0696909, 0.0739688, -0.1418561, 0.1403213

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_B2_A2_B1_A2_A1

### Relational analysis result of IS_A1_B1_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.93 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_A2_A2

### Relational analysis result of IS_A1_B1_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0523262, 0.0412350, -0.0877964, 0.0550800, -0.1074061, 0.1290314
1: -0.0455303, 0.0471103, -0.0792435, 0.0770770, -0.1226072, 0.1263538
2: -0.0514526, 0.1373822, -0.0876415, 0.1777528, -0.2292054, 0.2250237
3: -0.0286362, 0.0524742, -0.0496319, 0.0877473, -0.1163835, 0.1021062
4: -0.0587164, 0.0587944, -0.0845607, 0.0973942, -0.1561106, 0.1433551
5: -0.0337407, 0.0693948, -0.0695260, 0.1014760, -0.1352168, 0.1389208
6: -0.1025318, 0.0733131, -0.1381282, 0.1101242, -0.2126560, 0.2114413
7: 0.8383921, 1.0082769, 0.7866596, 1.0280844, -0.1896923, 0.2216173
8: -0.0871676, 0.0958343, -0.1142457, 0.1314447, -0.2186123, 0.2100800
9: -0.0784864, 0.0914394, -0.1074190, 0.1424540, -0.2209404, 0.1988584

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_A2_B2_A1_A1

### Relational analysis result of IS_A1_B1_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.95 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_A1_A2

### Relational analysis result of IS_A1_B1_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0393704, 0.0355213, -0.0877964, 0.0550800, -0.0944503, 0.1233177
1: -0.0360290, 0.0347434, -0.0792435, 0.0770770, -0.1131060, 0.1139869
2: -0.0366361, 0.1208368, -0.0876415, 0.1777528, -0.2143889, 0.2084783
3: -0.0200490, 0.0460347, -0.0496319, 0.0877473, -0.1077963, 0.0956666
4: -0.0487912, 0.0431493, -0.0845607, 0.0973942, -0.1461854, 0.1277100
5: -0.0220351, 0.0561553, -0.0695260, 0.1014760, -0.1235111, 0.1256813
6: -0.0879631, 0.0599984, -0.1381282, 0.1101242, -0.1980873, 0.1981266
7: 0.8597335, 1.0019666, 0.7866596, 1.0280844, -0.1683509, 0.2153070
8: -0.0763842, 0.0812626, -0.1142457, 0.1314447, -0.2078288, 0.1955083
9: -0.0678873, 0.0706304, -0.1074190, 0.1424540, -0.2103414, 0.1780494

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_B2_A2_B2_A2_A1

### Relational analysis result of IS_A1_B1_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.94 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_A2_A2

### Relational analysis result of IS_A1_B1_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0489081, 0.0395227, -0.0424515, 0.0369180, -0.0858260, 0.0819742
1: -0.0440075, 0.0434040, -0.0382567, 0.0377663, -0.0817738, 0.0816607
2: -0.0470499, 0.1323892, -0.0402579, 0.1247942, -0.1718441, 0.1726472
3: -0.0260395, 0.0531242, -0.0220895, 0.0476088, -0.0736483, 0.0752137
4: -0.0559772, 0.0540205, -0.0512173, 0.0467586, -0.1027358, 0.1052378
5: -0.0310501, 0.0654271, -0.0247056, 0.0593916, -0.0904417, 0.0901327
6: -0.0981293, 0.0699193, -0.0914324, 0.0632530, -0.1613823, 0.1613517
7: 0.8447902, 1.0069783, 0.8545228, 1.0035092, -0.1587189, 0.1524554
8: -0.0840603, 0.0914301, -0.0790200, 0.0847306, -0.1687908, 0.1704501
9: -0.0757362, 0.0851300, -0.0704781, 0.0755324, -0.1512686, 0.1556081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B1_B1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.16 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_B1_B2

### Relational analysis result of IS_A1_B2_A1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0474402, 0.0389355, -0.0531622, 0.0412245, -0.0886648, 0.0920977
1: -0.0426775, 0.0421331, -0.0478620, 0.0470876, -0.0897652, 0.0899951
2: -0.0455178, 0.1306770, -0.0514903, 0.1373517, -0.1828695, 0.1821673
3: -0.0251490, 0.0518174, -0.0286204, 0.0569117, -0.0820607, 0.0804378
4: -0.0548984, 0.0523833, -0.0591041, 0.0587652, -0.1136636, 0.1114874
5: -0.0295978, 0.0640665, -0.0352590, 0.0693706, -0.0989684, 0.0993255
6: -0.0966195, 0.0684018, -0.1025049, 0.0743174, -0.1709370, 0.1709067
7: 0.8469844, 1.0061816, 0.8384311, 1.0092872, -0.1623029, 0.1677505
8: -0.0829210, 0.0899198, -0.0873624, 0.0958074, -0.1787284, 0.1772822
9: -0.0745403, 0.0829663, -0.0792021, 0.0914009, -0.1659412, 0.1621684

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B1_B1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_B2_B2

### Relational analysis result of IS_A1_B2_A1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0551322, 0.0420127, -0.0473810, 0.0389117, -0.0940440, 0.0893937
1: -0.0496471, 0.0487935, -0.0426239, 0.0420817, -0.0917288, 0.0914173
2: -0.0535467, 0.1396498, -0.0454559, 0.1306079, -0.1841546, 0.1851057
3: -0.0298155, 0.0586657, -0.0251131, 0.0517646, -0.0815801, 0.0837788
4: -0.0605521, 0.0609626, -0.0548548, 0.0523173, -0.1128694, 0.1158174
5: -0.0372082, 0.0711968, -0.0295392, 0.0640115, -0.1012198, 0.1007360
6: -0.1045312, 0.0763541, -0.0965586, 0.0683405, -0.1728718, 0.1729127
7: 0.8354862, 1.0103563, 0.8470730, 1.0061494, -0.1706632, 0.1632833
8: -0.0888916, 0.0978346, -0.0828750, 0.0898587, -0.1787503, 0.1807096
9: -0.0808071, 0.0943049, -0.0744920, 0.0828790, -0.1636862, 0.1687969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B1_B2_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2_B1_B2

### Relational analysis result of IS_A1_B2_A1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0533525, 0.0413007, -0.0587692, 0.0446495, -0.0980021, 0.1000699
1: -0.0480346, 0.0472524, -0.0538939, 0.0514113, -0.0994459, 0.1011463
2: -0.0516890, 0.1375737, -0.0577788, 0.1445870, -0.1962760, 0.1953526
3: -0.0287358, 0.0570812, -0.0353966, 0.0613574, -0.0900932, 0.0924778
4: -0.0592440, 0.0589776, -0.0659136, 0.0663221, -0.1255661, 0.1248912
5: -0.0354474, 0.0695471, -0.0410981, 0.0764830, -0.1119304, 0.1106451
6: -0.1027007, 0.0745142, -0.1095440, 0.0808266, -0.1835273, 0.1840582
7: 0.8381466, 1.0093904, 0.8290530, 1.0127629, -0.1746163, 0.1803373
8: -0.0875101, 0.0960033, -0.0947241, 0.1044364, -0.1919464, 0.1907273
9: -0.0793571, 0.0916815, -0.0836711, 0.1054619, -0.1848190, 0.1753526

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B1_B2_B2_B1

### Relational analysis result of IS_A1_B2_A1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.06 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2_B2_B2

### Relational analysis result of IS_A1_B2_A1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0408999, 0.0362396, -0.0995194, 0.0656144, -0.1065142, 0.1357590
1: -0.0371137, 0.0362980, -0.0946026, 0.0844675, -0.1215812, 0.1309006
2: -0.0384988, 0.1228162, -0.1021419, 0.1947447, -0.2332435, 0.2249581
3: -0.0210608, 0.0468442, -0.0739483, 0.0953464, -0.1164071, 0.1207926
4: -0.0500389, 0.0448673, -0.1064930, 0.1172374, -0.1672764, 0.1513603
5: -0.0232858, 0.0578197, -0.0824530, 0.1222885, -0.1455743, 0.1402727
6: -0.0896883, 0.0616722, -0.1567926, 0.1256659, -0.2153542, 0.2184648
7: 0.8570576, 1.0027598, 0.7643535, 1.0366939, -0.1796364, 0.2384063
8: -0.0777398, 0.0829858, -0.1382578, 0.1583597, -0.2360995, 0.2212436
9: -0.0692198, 0.0730329, -0.1164552, 0.1884571, -0.2576769, 0.1894881

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B2_A1_A1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_A1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0518903, 0.0407157, -0.0974832, 0.0645668, -0.1164571, 0.1381989
1: -0.0467096, 0.0459863, -0.0925685, 0.0828158, -0.1295254, 0.1385548
2: -0.0501627, 0.1358681, -0.0999251, 0.1922383, -0.2424011, 0.2357932
3: -0.0278488, 0.0557794, -0.0720219, 0.0936480, -0.1214968, 0.1278013
4: -0.0581692, 0.0573467, -0.1044653, 0.1146932, -0.1728625, 0.1618120
5: -0.0340007, 0.0681916, -0.0803866, 0.1199997, -0.1540003, 0.1485782
6: -0.1011967, 0.0730025, -0.1544316, 0.1234254, -0.2246221, 0.2274341
7: 0.8403323, 1.0085968, 0.7675865, 1.0354984, -0.1951661, 0.2410103
8: -0.0863751, 0.0944987, -0.1360825, 0.1556652, -0.2420404, 0.2305812
9: -0.0781658, 0.0895260, -0.1148171, 0.1843099, -0.2624758, 0.2043431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B2_A1_A2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.25 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_A2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0452640, 0.0380649, -0.1063969, 0.0691528, -0.1144168, 0.1444618
1: -0.0407057, 0.0402487, -0.1014732, 0.0900465, -0.1307522, 0.1417219
2: -0.0432463, 0.1281384, -0.1096293, 0.2032100, -0.2464563, 0.2377677
3: -0.0238288, 0.0498798, -0.0804549, 0.1010828, -0.1249116, 0.1303347
4: -0.0532988, 0.0499561, -0.1133418, 0.1258307, -0.1791295, 0.1632979
5: -0.0274447, 0.0620491, -0.0894327, 0.1300193, -0.1574640, 0.1514818
6: -0.0943812, 0.0661519, -0.1647669, 0.1332337, -0.2276148, 0.2309189
7: 0.8502374, 1.0050005, 0.7534339, 1.0407327, -0.1904953, 0.2515666
8: -0.0812318, 0.0876805, -0.1456053, 0.1674607, -0.2486924, 0.2332858
9: -0.0727673, 0.0797585, -0.1219884, 0.2024647, -0.2752320, 0.2017468

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B2_A2_A1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.14 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_A1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0563013, 0.0424803, -0.1043963, 0.0681235, -0.1244248, 0.1468767
1: -0.0507064, 0.0498057, -0.0994746, 0.0884237, -0.1391301, 0.1492804
2: -0.0547669, 0.1410135, -0.1074513, 0.2007475, -0.2555145, 0.2484648
3: -0.0305248, 0.0597066, -0.0785622, 0.0994142, -0.1299390, 0.1382688
4: -0.0614114, 0.0622664, -0.1113496, 0.1233310, -0.1847424, 0.1736160
5: -0.0383649, 0.0722805, -0.0874024, 0.1277705, -0.1661354, 0.1596829
6: -0.1057337, 0.0775628, -0.1624473, 0.1310324, -0.2367660, 0.2400101
7: 0.8337388, 1.0109909, 0.7566103, 1.0395581, -0.2058192, 0.2543806
8: -0.0897990, 0.0990375, -0.1434680, 0.1648133, -0.2546123, 0.2425055
9: -0.0817596, 0.0960281, -0.1203789, 0.1983900, -0.2801496, 0.2164070

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B2_A2_A2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_A2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0346038, 0.0329501, -0.0715080, 0.0512033, -0.0858071, 0.1044582
1: -0.0329297, 0.0291781, -0.0666198, 0.0617449, -0.0946746, 0.0957979
2: -0.0299686, 0.1144697, -0.0716470, 0.1602666, -0.1902352, 0.1861167
3: -0.0169107, 0.0431368, -0.0474481, 0.0719826, -0.0888933, 0.0905850
4: -0.0443248, 0.0387755, -0.0785990, 0.0822386, -0.1265633, 0.1173746
5: -0.0191343, 0.0501974, -0.0540260, 0.0908021, -0.1099364, 0.1042234
6: -0.0825459, 0.0540066, -0.1243142, 0.0948437, -0.1773896, 0.1783209
7: 0.8692628, 0.9991268, 0.8088275, 1.0202439, -0.1509811, 0.1902993
8: -0.0715315, 0.0758695, -0.1083330, 0.1212932, -0.1928247, 0.1842025
9: -0.0631176, 0.0635544, -0.0939196, 0.1314068, -0.1945244, 0.1574741

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 153

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0346038, 0.0329501, -0.0560468, 0.0432490, -0.0778527, 0.0889969
1: -0.0329297, 0.0291781, -0.0511743, 0.0492029, -0.0821326, 0.0803524
2: -0.0299686, 0.1144697, -0.0548151, 0.1412360, -0.1712047, 0.1692848
3: -0.0169107, 0.0431368, -0.0328210, 0.0590867, -0.0759974, 0.0759579
4: -0.0443248, 0.0387755, -0.0632026, 0.0629205, -0.1072453, 0.1019781
5: -0.0191343, 0.0501974, -0.0383353, 0.0734228, -0.0925571, 0.0885327
6: -0.0825459, 0.0540066, -0.1063875, 0.0778310, -0.1603769, 0.1603941
7: 0.8692628, 0.9991268, 0.8333754, 1.0111642, -0.1419014, 0.1657514
8: -0.0715315, 0.0758695, -0.0918157, 0.1008339, -0.1723654, 0.1676851
9: -0.0631176, 0.0635544, -0.0814809, 0.0999172, -0.1630348, 0.1450353

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 153

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0382337, 0.0349082, -0.0782530, 0.0546734, -0.0929071, 0.1131612
1: -0.0352899, 0.0334162, -0.0733579, 0.0672164, -0.1025064, 0.1067741
2: -0.0350461, 0.1193184, -0.0789901, 0.1685688, -0.2036149, 0.1983085
3: -0.0193006, 0.0453436, -0.0538293, 0.0776085, -0.0969091, 0.0991729
4: -0.0477261, 0.0421063, -0.0853157, 0.0906661, -0.1383922, 0.1274220
5: -0.0213434, 0.0547345, -0.0608711, 0.0983839, -0.1197273, 0.1156056
6: -0.0866712, 0.0585695, -0.1321349, 0.1022656, -0.1889368, 0.1907044
7: 0.8620061, 1.0012894, 0.7981182, 1.0242051, -0.1621990, 0.2031711
8: -0.0752269, 0.0799765, -0.1155388, 0.1302187, -0.2054456, 0.1955152
9: -0.0667498, 0.0689430, -0.0993461, 0.1451442, -0.2118940, 0.1682891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0382337, 0.0349082, -0.0621303, 0.0463788, -0.0846125, 0.0970385
1: -0.0352899, 0.0334162, -0.0572516, 0.0541378, -0.0894277, 0.0906678
2: -0.0350461, 0.1193184, -0.0614380, 0.1487240, -0.1837702, 0.1807564
3: -0.0193006, 0.0453436, -0.0385764, 0.0641609, -0.0834615, 0.0839200
4: -0.0477261, 0.0421063, -0.0692606, 0.0705216, -0.1182477, 0.1113669
5: -0.0213434, 0.0547345, -0.0445091, 0.0802610, -0.1016044, 0.0992436
6: -0.0866712, 0.0585695, -0.1134411, 0.0845250, -0.1711962, 0.1720106
7: 0.8620061, 1.0012894, 0.8237165, 1.0147368, -0.1527307, 0.1775728
8: -0.0752269, 0.0799765, -0.0983148, 0.1088841, -0.1841110, 0.1782913
9: -0.0667498, 0.0689430, -0.0863752, 0.1123074, -0.1790573, 0.1553182

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0303272, 0.0306433, -0.1045211, 0.0681877, -0.0985149, 0.1351643
1: -0.0301491, 0.0241850, -0.0995993, 0.0885249, -0.1186739, 0.1237842
2: -0.0239865, 0.1087570, -0.1075869, 0.2009011, -0.2248876, 0.2163440
3: -0.0140950, 0.0405369, -0.0786802, 0.0995182, -0.1136132, 0.1192171
4: -0.0403174, 0.0348514, -0.1114738, 0.1234868, -0.1638042, 0.1463251
5: -0.0165317, 0.0448519, -0.0875290, 0.1279107, -0.1444423, 0.1323809
6: -0.0776855, 0.0486308, -0.1625918, 0.1311696, -0.2088551, 0.2112226
7: 0.8778123, 0.9965792, 0.7564123, 1.0396314, -0.1618191, 0.2401669
8: -0.0671777, 0.0710308, -0.1436012, 0.1649784, -0.2321561, 0.2146320
9: -0.0588382, 0.0572058, -0.1204792, 0.1986440, -0.2574822, 0.1776850

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A1_A1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_A1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0383715, 0.0349825, -0.1024916, 0.0671436, -0.1055150, 0.1374741
1: -0.0353795, 0.0335771, -0.0975718, 0.0868785, -0.1222581, 0.1311489
2: -0.0352389, 0.1195025, -0.1053777, 0.1984031, -0.2336420, 0.2248802
3: -0.0193914, 0.0454274, -0.0767602, 0.0978255, -0.1172168, 0.1221876
4: -0.0478552, 0.0422327, -0.1094528, 0.1209511, -0.1688063, 0.1516855
5: -0.0214272, 0.0549068, -0.0854694, 0.1256295, -0.1470567, 0.1403762
6: -0.0868278, 0.0587427, -0.1602388, 0.1289365, -0.2157643, 0.2189815
7: 0.8617304, 1.0013715, 0.7596345, 1.0384395, -0.1767091, 0.2417370
8: -0.0753672, 0.0801324, -0.1414331, 0.1622928, -0.2376601, 0.2215655
9: -0.0668877, 0.0691476, -0.1188465, 0.1945106, -0.2613983, 0.1879941

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0222971, 0.0241526, -0.1008808, 0.0663148, -0.0886119, 0.1250334
1: -0.0237433, 0.0138740, -0.0959626, 0.0855719, -0.1093151, 0.1098367
2: -0.0107641, 0.0926839, -0.1036240, 0.1964203, -0.2071845, 0.1963079
3: -0.0087936, 0.0332216, -0.0752363, 0.0964819, -0.1052755, 0.1084579
4: -0.0299053, 0.0275141, -0.1078487, 0.1189384, -0.1488437, 0.1353628
5: -0.0106746, 0.0359811, -0.0838346, 0.1238188, -0.1344934, 0.1198157
6: -0.0658650, 0.0355457, -0.1583710, 0.1271640, -0.1930290, 0.1939168
7: 0.9001579, 0.9907899, 0.7621919, 1.0374935, -0.1373356, 0.2285981
8: -0.0560068, 0.0575493, -0.1397122, 0.1601612, -0.2161680, 0.1972615
9: -0.0472087, 0.0442447, -0.1175505, 0.1912298, -0.2384385, 0.1617952

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A2_A1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.14 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_A1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0312078, 0.0311183, -0.1105600, 0.0712945, -0.1025024, 0.1416782
1: -0.0307217, 0.0252132, -0.1056320, 0.0934235, -0.1241452, 0.1308452
2: -0.0252184, 0.1099334, -0.1141613, 0.2083341, -0.2335525, 0.2240947
3: -0.0146748, 0.0410723, -0.0843933, 0.1045551, -0.1192299, 0.1254656
4: -0.0411426, 0.0356594, -0.1174874, 0.1310321, -0.1721747, 0.1531468
5: -0.0170676, 0.0459527, -0.0936575, 0.1346987, -0.1517663, 0.1396102
6: -0.0786864, 0.0497378, -0.1695938, 0.1378145, -0.2165009, 0.2193316
7: 0.8760517, 0.9971038, 0.7468241, 1.0431776, -0.1671259, 0.2502796
8: -0.0680742, 0.0720272, -0.1500527, 0.1729695, -0.2410437, 0.2220799
9: -0.0597195, 0.0585131, -0.1253376, 0.2109434, -0.2706628, 0.1838507

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.99 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0631969, 0.0469275, -0.0267542, 0.0287010, -0.0918979, 0.0736816
1: -0.0583171, 0.0550030, -0.0278080, 0.0200411, -0.0783582, 0.0828109
2: -0.0625991, 0.1500368, -0.0190075, 0.1039475, -0.1665466, 0.1690443
3: -0.0395854, 0.0650504, -0.0117460, 0.0383479, -0.0779333, 0.0767964
4: -0.0703227, 0.0718542, -0.0369436, 0.0316131, -0.1019359, 0.1087978
5: -0.0455915, 0.0814600, -0.0144372, 0.0403515, -0.0859430, 0.0958971
6: -0.1146778, 0.0856986, -0.0736123, 0.0441048, -0.1587826, 0.1593108
7: 0.8220230, 1.0153630, 0.8850104, 0.9945081, -0.1724851, 0.1303526
8: -0.0994542, 0.1102954, -0.0635122, 0.0669658, -0.1664200, 0.1738075
9: -0.0872332, 0.1144796, -0.0552353, 0.0519531, -0.1391864, 0.1697150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 153

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_B1_A1_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.58 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.10 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0731992, 0.0520734, -0.0379657, 0.0347636, -0.1079628, 0.0900391
1: -0.0683093, 0.0631168, -0.0351157, 0.0331034, -0.1014126, 0.0982325
2: -0.0734882, 0.1623483, -0.0346713, 0.1189605, -0.1924487, 0.1970196
3: -0.0490481, 0.0733932, -0.0191242, 0.0451807, -0.0942288, 0.0925175
4: -0.0802831, 0.0843517, -0.0474750, 0.0418604, -0.1221435, 0.1318267
5: -0.0557423, 0.0927032, -0.0211803, 0.0543996, -0.1101418, 0.1138835
6: -0.1262752, 0.0967046, -0.0863667, 0.0582327, -0.1845078, 0.1830713
7: 0.8061424, 1.0212373, 0.8625417, 1.0011299, -0.1949875, 0.1586956
8: -0.1101398, 0.1235311, -0.0749541, 0.0796733, -0.1898131, 0.1984853
9: -0.0952803, 0.1348512, -0.0664817, 0.0685452, -0.1638255, 0.2013330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_B1_A1_B1_B2_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.04 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_B2_B2

### Relational analysis result of IS_A2_B1_B1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0614127, 0.0460095, -0.0351269, 0.0332323, -0.0946450, 0.0811364
1: -0.0565347, 0.0535557, -0.0332699, 0.0297889, -0.0863236, 0.0868255
2: -0.0606567, 0.1478407, -0.0307004, 0.1151684, -0.1758251, 0.1785411
3: -0.0378974, 0.0635622, -0.0172551, 0.0434549, -0.0813523, 0.0808174
4: -0.0685460, 0.0696250, -0.0448149, 0.0392555, -0.1078015, 0.1144399
5: -0.0437808, 0.0794544, -0.0194526, 0.0508512, -0.0946320, 0.0989071
6: -0.1126091, 0.0837354, -0.0831404, 0.0546642, -0.1672732, 0.1668757
7: 0.8248559, 1.0143151, 0.8682170, 0.9994385, -0.1745826, 0.1460981
8: -0.0975481, 0.1079344, -0.0720641, 0.0764614, -0.1740094, 0.1799984
9: -0.0857978, 0.1108458, -0.0636411, 0.0643310, -0.1501288, 0.1744868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_B1_A1_B2_B1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.15 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2_B1_B2

### Relational analysis result of IS_A2_B1_B1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0714800, 0.0511889, -0.0485196, 0.0393673, -0.1108473, 0.0997085
1: -0.0665918, 0.0617222, -0.0436555, 0.0430676, -0.1096594, 0.1053777
2: -0.0716166, 0.1602322, -0.0466444, 0.1319361, -0.2035527, 0.2068766
3: -0.0474216, 0.0719592, -0.0258038, 0.0527783, -0.1002000, 0.0977631
4: -0.0785711, 0.0822036, -0.0556917, 0.0535872, -0.1321583, 0.1378953
5: -0.0539975, 0.0907707, -0.0306657, 0.0650670, -0.1190645, 0.1214364
6: -0.1242818, 0.0948129, -0.0977297, 0.0695177, -0.1937995, 0.1925426
7: 0.8088719, 1.0202274, 0.8453710, 1.0067674, -0.1978955, 0.1748564
8: -0.1083031, 0.1212561, -0.0837588, 0.0910304, -0.1993335, 0.2050149
9: -0.0938972, 0.1313497, -0.0754197, 0.0845574, -0.1784545, 0.2067694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_B1_A1_B2_B2_B1

### Relational analysis result of IS_A2_B1_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.04 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2_B2_B2

### Relational analysis result of IS_A2_B1_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0701172, 0.0504878, -0.0309905, 0.0310011, -0.1011183, 0.0814783
1: -0.0652303, 0.0606166, -0.0305804, 0.0249595, -0.0901898, 0.0911970
2: -0.0701329, 0.1585547, -0.0249144, 0.1096431, -0.1797760, 0.1834692
3: -0.0461323, 0.0708225, -0.0145318, 0.0409402, -0.0870725, 0.0853543
4: -0.0772140, 0.0805008, -0.0409390, 0.0354600, -0.1126740, 0.1214398
5: -0.0526145, 0.0892387, -0.0169353, 0.0456811, -0.0982955, 0.1061741
6: -0.1227016, 0.0933133, -0.0784394, 0.0494646, -0.1721663, 0.1717527
7: 0.8110358, 1.0194272, 0.8764862, 0.9969743, -0.1859386, 0.1429410
8: -0.1068472, 0.1194527, -0.0678530, 0.0717813, -0.1786286, 0.1873057
9: -0.0928008, 0.1285741, -0.0595020, 0.0581905, -0.1509912, 0.1880760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_B1_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.02 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0800875, 0.0556172, -0.0429497, 0.0371358, -0.1172233, 0.0985669
1: -0.0751905, 0.0687045, -0.0386236, 0.0382377, -0.1134282, 0.1073281
2: -0.0809872, 0.1708268, -0.0408227, 0.1254293, -0.2064164, 0.2116495
3: -0.0555647, 0.0791386, -0.0224198, 0.0478542, -0.1034190, 0.1015584
4: -0.0871425, 0.0929582, -0.0515957, 0.0473658, -0.1345083, 0.1445539
5: -0.0627327, 0.1004459, -0.0251614, 0.0598963, -0.1226290, 0.1256074
6: -0.1342619, 0.1042841, -0.0919924, 0.0637606, -0.1980225, 0.1962765
7: 0.7952057, 1.0252824, 0.8537090, 1.0037496, -0.2085440, 0.1715733
8: -0.1174985, 0.1326461, -0.0794311, 0.0852908, -0.2027894, 0.2120772
9: -0.1008220, 0.1488804, -0.0708822, 0.0763350, -0.1771570, 0.2197626

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_B1_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.13 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.05 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0680248, 0.0494113, -0.0389883, 0.0353152, -0.1033400, 0.0883995
1: -0.0631400, 0.0589193, -0.0357806, 0.0342972, -0.0974372, 0.0946999
2: -0.0678550, 0.1559792, -0.0361017, 0.1203263, -0.1881813, 0.1920809
3: -0.0441528, 0.0690772, -0.0197974, 0.0458024, -0.0899551, 0.0888747
4: -0.0751303, 0.0778864, -0.0484331, 0.0427987, -0.1179290, 0.1263195
5: -0.0504909, 0.0868868, -0.0218026, 0.0556777, -0.1061686, 0.1086894
6: -0.1202756, 0.0910109, -0.0875288, 0.0595180, -0.1797936, 0.1785396
7: 0.8143578, 1.0181984, 0.8604974, 1.0017390, -0.1873812, 0.1577010
8: -0.1046118, 0.1166839, -0.0759951, 0.0808302, -0.1854421, 0.1926790
9: -0.0911174, 0.1243125, -0.0675049, 0.0700632, -0.1611806, 0.1918174

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_B1_A2_B2_B1_B1

### Relational analysis result of IS_A2_B1_B1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.91 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_B1_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.30 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0781074, 0.0545985, -0.0537751, 0.0414697, -0.1195771, 0.1083736
1: -0.0732124, 0.0670983, -0.0484175, 0.0476184, -0.1208308, 0.1155157
2: -0.0788315, 0.1683894, -0.0521301, 0.1380668, -0.2168983, 0.2205196
3: -0.0536914, 0.0774870, -0.0289923, 0.0574575, -0.1111489, 0.1064792
4: -0.0851707, 0.0904841, -0.0595547, 0.0594490, -0.1446196, 0.1500388
5: -0.0607232, 0.0982201, -0.0358656, 0.0699388, -0.1306621, 0.1340857
6: -0.1319660, 0.1021053, -0.1031354, 0.0749512, -0.2069172, 0.2052407
7: 0.7983497, 1.0241194, 0.8375149, 1.0096200, -0.2112703, 0.1866045
8: -0.1153832, 0.1300259, -0.0878382, 0.0964382, -0.2118213, 0.2178641
9: -0.0992289, 0.1448475, -0.0797015, 0.0923045, -0.1915334, 0.2245490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_B1_A2_B2_B2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.95 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_B2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0738552, 0.0524109, -0.0201799, 0.0214546, -0.0953097, 0.0725908
1: -0.0689645, 0.0636489, -0.0215130, 0.0107368, -0.0797014, 0.0851619
2: -0.0742023, 0.1631557, -0.0062473, 0.0860029, -0.1602051, 0.1694030
3: -0.0496687, 0.0739403, -0.0074441, 0.0301809, -0.0798495, 0.0813844
4: -0.0809364, 0.0851712, -0.0258893, 0.0254281, -0.1063645, 0.1110605
5: -0.0564080, 0.0934404, -0.0084428, 0.0343026, -0.0907106, 0.1018832
6: -0.1270357, 0.0974264, -0.0614700, 0.0308444, -0.1578801, 0.1588964
7: 0.8051009, 1.0216223, 0.9089249, 0.9886324, -0.1835316, 0.1126974
8: -0.1108405, 0.1243991, -0.0517483, 0.0519638, -0.1628044, 0.1761474
9: -0.0958080, 0.1361872, -0.0425232, 0.0401188, -0.1359268, 0.1787105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=44, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B2_B1_B1_B1_A1

### Relational analysis result of IS_A2_B1_B2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.98 seconds

## Relational analysis of IS_A2_B1_B2_B1_B1_B1_A2

### Relational analysis result of IS_A2_B1_B2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0838242, 0.0575396, -0.0249464, 0.0275284, -0.1113526, 0.0824860
1: -0.0789233, 0.0717357, -0.0265339, 0.0177995, -0.0967229, 0.0982695
2: -0.0850550, 0.1754260, -0.0164157, 0.1010437, -0.1860987, 0.1918417
3: -0.0590998, 0.0822552, -0.0104822, 0.0370263, -0.0961261, 0.0927374
4: -0.0908635, 0.0976269, -0.0349304, 0.0301242, -0.1209877, 0.1325573
5: -0.0665248, 0.1046461, -0.0134672, 0.0380813, -0.1046061, 0.1181133
6: -0.1385944, 0.1083957, -0.0713643, 0.0414284, -0.1800228, 0.1797600
7: 0.7892731, 1.0274765, 0.8891883, 0.9934894, -0.2042164, 0.1382883
8: -0.1214905, 0.1375906, -0.0613353, 0.0645382, -0.1860286, 0.1989259
9: -0.1038282, 0.1564907, -0.0530714, 0.0494073, -0.1532355, 0.2095622

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 153

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B2_B1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.95 seconds

## Relational analysis of IS_A2_B1_B2_B1_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0717691, 0.0513376, -0.0238011, 0.0260691, -0.0978382, 0.0751387
1: -0.0668806, 0.0619566, -0.0253276, 0.0161026, -0.0829832, 0.0872842
2: -0.0719313, 0.1605879, -0.0139726, 0.0974299, -0.1693612, 0.1745606
3: -0.0476951, 0.0722003, -0.0097523, 0.0353816, -0.0830767, 0.0819525
4: -0.0788589, 0.0825647, -0.0327581, 0.0289959, -0.1078548, 0.1153229
5: -0.0542908, 0.0910956, -0.0122600, 0.0371734, -0.0914643, 0.1033555
6: -0.1246170, 0.0951309, -0.0689871, 0.0388854, -0.1635024, 0.1641180
7: 0.8084130, 1.0203974, 0.8939302, 0.9923225, -0.1839095, 0.1264672
8: -0.1086119, 0.1216386, -0.0590319, 0.0615170, -0.1701289, 0.1806705
9: -0.0941297, 0.1319384, -0.0505371, 0.0471756, -0.1413053, 0.1824754

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 153

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B2_B1_B2_B1_A1

### Relational analysis result of IS_A2_B1_B2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.05 seconds

## Relational analysis of IS_A2_B1_B2_B1_B2_B1_A2

### Relational analysis result of IS_A2_B1_B2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.18 seconds

## BFS IS instance: IS_A2_B1_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0818170, 0.0565070, -0.0327427, 0.0319462, -0.1137632, 0.0892497
1: -0.0769183, 0.0701074, -0.0317197, 0.0270052, -0.1039235, 0.1018271
2: -0.0828700, 0.1729555, -0.0273653, 0.1119837, -0.1948537, 0.2003209
3: -0.0572009, 0.0805811, -0.0156854, 0.0420054, -0.0992063, 0.0962666
4: -0.0888647, 0.0951191, -0.0425808, 0.0370678, -0.1259326, 0.1377000
5: -0.0644879, 0.1023900, -0.0180017, 0.0478712, -0.1123591, 0.1203917
6: -0.1362672, 0.1061871, -0.0804308, 0.0516672, -0.1879343, 0.1866179
7: 0.7924597, 1.0262980, 0.8729834, 0.9980182, -0.2055585, 0.1533147
8: -0.1193462, 0.1349347, -0.0696368, 0.0737638, -0.1931100, 0.2045716
9: -0.1022134, 0.1524029, -0.0612553, 0.0607916, -0.1630051, 0.2136582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B2_B1_B2_B2_A1

### Relational analysis result of IS_A2_B1_B2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.97 seconds

## Relational analysis of IS_A2_B1_B2_B1_B2_B2_A2

### Relational analysis result of IS_A2_B1_B2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.16 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1104169, 0.0712209, -0.0373082, 0.0344089, -0.1448258, 0.1085291
1: -0.1054891, 0.0933075, -0.0346882, 0.0323356, -0.1378247, 0.1279957
2: -0.1140055, 0.2081579, -0.0337516, 0.1180821, -0.2320876, 0.2419095
3: -0.0842580, 0.1044358, -0.0186913, 0.0447810, -0.1290389, 0.1231271
4: -0.1173449, 0.1308533, -0.0468588, 0.0412571, -0.1586020, 0.1777122
5: -0.0935123, 0.1345380, -0.0207801, 0.0535777, -0.1470900, 0.1553181
6: -0.1694279, 0.1376570, -0.0856194, 0.0574061, -0.2268341, 0.2232764
7: 0.7470515, 1.0430937, 0.8638561, 1.0007379, -0.2536864, 0.1792375
8: -0.1498998, 0.1727801, -0.0742848, 0.0789293, -0.2288291, 0.2470649
9: -0.1252225, 0.2106519, -0.0658237, 0.0675691, -0.1927916, 0.2764756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 75

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_B2_A1_B1_B1

### Relational analysis result of IS_A2_B1_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.98 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_B1_B2

### Relational analysis result of IS_A2_B1_B2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1104169, 0.0712209, -0.0702452, 0.0521758, -0.1625928, 0.1414661
1: -0.1054891, 0.0933075, -0.0561041, 0.0707912, -0.1762802, 0.1494116
2: -0.1140055, 0.2081579, -0.0798240, 0.1620790, -0.2760845, 0.2879820
3: -0.0842580, 0.1044358, -0.0403769, 0.0648051, -0.1490631, 0.1448126
4: -0.1173449, 0.1308533, -0.0777219, 0.0714799, -0.1888247, 0.2085752
5: -0.0935123, 0.1345380, -0.0408249, 0.0947467, -0.1882590, 0.1753628
6: -0.1694279, 0.1376570, -0.1230521, 0.0988090, -0.2682370, 0.2607092
7: 0.7470515, 1.0430937, 0.7980095, 1.0203602, -0.2733088, 0.2450842
8: -0.1498998, 0.1727801, -0.1078163, 0.1161955, -0.2660953, 0.2805964
9: -0.1252225, 0.2106519, -0.0987821, 0.1164643, -0.2416868, 0.3094340

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 75

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B1_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.24 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B1_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.97 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0940937, 0.0628231, -0.0373082, 0.0344089, -0.1285026, 0.1001312
1: -0.0891825, 0.0800662, -0.0346882, 0.0323356, -0.1215181, 0.1147544
2: -0.0962352, 0.1880664, -0.0337516, 0.1180821, -0.2143174, 0.2218180
3: -0.0688154, 0.0908209, -0.0186913, 0.0447810, -0.1135963, 0.1095122
4: -0.1010901, 0.1104583, -0.0468588, 0.0412571, -0.1423471, 0.1573171
5: -0.0769468, 0.1161898, -0.0207801, 0.0535777, -0.1305245, 0.1369699
6: -0.1505017, 0.1196958, -0.0856194, 0.0574061, -0.2079078, 0.2053152
7: 0.7729679, 1.0335077, 0.8638561, 1.0007379, -0.2277700, 0.1696516
8: -0.1324616, 0.1511801, -0.0742848, 0.0789293, -0.2113909, 0.2254649
9: -0.1120902, 0.1774067, -0.0658237, 0.0675691, -0.1796593, 0.2432304

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 153

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_B2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.98 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0940937, 0.0628231, -0.0702452, 0.0521758, -0.1462696, 0.1330683
1: -0.0891825, 0.0800662, -0.0561041, 0.0707912, -0.1599737, 0.1361704
2: -0.0962352, 0.1880664, -0.0798240, 0.1620790, -0.2583142, 0.2678905
3: -0.0688154, 0.0908209, -0.0403769, 0.0648051, -0.1336204, 0.1311978
4: -0.1010901, 0.1104583, -0.0777219, 0.0714799, -0.1725699, 0.1881802
5: -0.0769468, 0.1161898, -0.0408249, 0.0947467, -0.1716935, 0.1570147
6: -0.1505017, 0.1196958, -0.1230521, 0.0988090, -0.2493107, 0.2427479
7: 0.7729679, 1.0335077, 0.7980095, 1.0203602, -0.2473923, 0.2354982
8: -0.1324616, 0.1511801, -0.1078163, 0.1161955, -0.2486571, 0.2589965
9: -0.1120902, 0.1774067, -0.0987821, 0.1164643, -0.2285545, 0.2761889

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.05 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0417568, 0.0366142, -0.0602581, 0.0454155, -0.0871723, 0.0968723
1: -0.0377449, 0.0371089, -0.0553813, 0.0526191, -0.0903640, 0.0924902
2: -0.0394703, 0.1239086, -0.0593997, 0.1464195, -0.1858899, 0.1833083
3: -0.0216289, 0.0472665, -0.0368051, 0.0625992, -0.0842282, 0.0840716
4: -0.0506897, 0.0459118, -0.0673962, 0.0681823, -0.1188720, 0.1133080
5: -0.0240699, 0.0586878, -0.0426090, 0.0781565, -0.1022265, 0.1012968
6: -0.0906515, 0.0625452, -0.1112703, 0.0824648, -0.1731163, 0.1738155
7: 0.8556577, 1.0031736, 0.8266892, 1.0136372, -0.1579795, 0.1764844
8: -0.0784468, 0.0839494, -0.0963146, 0.1064065, -0.1848533, 0.1802640
9: -0.0699147, 0.0744133, -0.0848690, 0.1084942, -0.1784089, 0.1592823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_B1_A1_A1_A1_B1

### Relational analysis result of IS_A2_B2_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.96 seconds

## Relational analysis of IS_A2_B2_B1_A1_A1_A1_B2

### Relational analysis result of IS_A2_B2_B1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.04 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0592346, 0.0448890, -0.0696584, 0.0502517, -0.1094863, 0.1145473
1: -0.0543589, 0.0517888, -0.0647720, 0.0602445, -0.1146034, 0.1165608
2: -0.0582854, 0.1451598, -0.0696334, 0.1579900, -0.2162754, 0.2147932
3: -0.0358369, 0.0617455, -0.0456982, 0.0704398, -0.1062767, 0.1074438
4: -0.0663770, 0.0669035, -0.0767571, 0.0799275, -0.1463045, 0.1436606
5: -0.0415704, 0.0770061, -0.0521488, 0.0887230, -0.1302934, 0.1291550
6: -0.1100836, 0.0813387, -0.1221696, 0.0928084, -0.2028920, 0.2035083
7: 0.8283140, 1.0130363, 0.8117642, 1.0191576, -0.1908436, 0.2012721
8: -0.0952213, 0.1050522, -0.1063570, 0.1188455, -0.2140668, 0.2114092
9: -0.0840455, 0.1064097, -0.0924316, 0.1276395, -0.2116850, 0.1988413

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_B1_A1_A1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.00 seconds

## Relational analysis of IS_A2_B2_B1_A1_A1_A2_B2

### Relational analysis result of IS_A2_B2_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0518051, 0.0410667, -0.0584315, 0.0444758, -0.0962809, 0.0994981
1: -0.0469369, 0.0457621, -0.0535565, 0.0511373, -0.0980741, 0.0993186
2: -0.0501972, 0.1360151, -0.0574112, 0.1441712, -0.1943685, 0.1934263
3: -0.0288081, 0.0555487, -0.0350770, 0.0610757, -0.0898838, 0.0906257
4: -0.0589786, 0.0576207, -0.0655772, 0.0659001, -0.1248787, 0.1231979
5: -0.0340306, 0.0686549, -0.0407553, 0.0761033, -0.1101339, 0.1094102
6: -0.1014693, 0.0731636, -0.1091524, 0.0804549, -0.1819243, 0.1823160
7: 0.8401100, 1.0086733, 0.8295892, 1.0125647, -0.1724547, 0.1790841
8: -0.0872842, 0.0952209, -0.0943632, 0.1039894, -0.1912736, 0.1895841
9: -0.0780684, 0.0912780, -0.0833994, 0.1047739, -0.1828423, 0.1746775

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_B1_A1_A2_A1_B1

### Relational analysis result of IS_A2_B2_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.43 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2_A1_B2

### Relational analysis result of IS_A2_B2_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.22 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0700738, 0.0504654, -0.0679562, 0.0493760, -0.1194498, 0.1184216
1: -0.0651870, 0.0605814, -0.0630716, 0.0588637, -0.1240506, 0.1236530
2: -0.0700856, 0.1585013, -0.0677803, 0.1558948, -0.2259805, 0.2262816
3: -0.0460912, 0.0707863, -0.0440879, 0.0690201, -0.1151113, 0.1148742
4: -0.0771707, 0.0804465, -0.0750621, 0.0778007, -0.1549715, 0.1555086
5: -0.0525704, 0.0891900, -0.0504214, 0.0868097, -0.1393801, 0.1396114
6: -0.1226513, 0.0932655, -0.1201960, 0.0909355, -0.2135868, 0.2134615
7: 0.8111046, 1.0194017, 0.8144667, 1.0181580, -0.2070534, 0.2049350
8: -0.1068008, 0.1193952, -0.1045386, 0.1165932, -0.2233940, 0.2239339
9: -0.0927658, 0.1284856, -0.0910622, 0.1241729, -0.2169387, 0.2195478

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_B1_A1_A2_A2_B1

### Relational analysis result of IS_A2_B2_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.47 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2_A2_B2

### Relational analysis result of IS_A2_B2_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0476673, 0.0390263, -0.0668797, 0.0488222, -0.0964895, 0.1059060
1: -0.0428833, 0.0423297, -0.0619961, 0.0579905, -0.1008738, 0.1043258
2: -0.0457548, 0.1309419, -0.0666083, 0.1545698, -0.2003246, 0.1975503
3: -0.0252868, 0.0520196, -0.0430695, 0.0681221, -0.0934090, 0.0950891
4: -0.0550653, 0.0526366, -0.0739901, 0.0764557, -0.1315210, 0.1266267
5: -0.0298225, 0.0642770, -0.0493289, 0.0855996, -0.1154221, 0.1136059
6: -0.0968532, 0.0686366, -0.1189478, 0.0897509, -0.1866041, 0.1875844
7: 0.8466448, 1.0063047, 0.8161759, 1.0175259, -0.1708811, 0.1901289
8: -0.0830973, 0.0901535, -0.1033886, 0.1151686, -0.1982659, 0.1935420
9: -0.0747254, 0.0833011, -0.0901961, 0.1219803, -0.1967056, 0.1734972

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_B1_A2_A1_A1_A1

### Relational analysis result of IS_A2_B2_B1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.00 seconds

## Relational analysis of IS_A2_B2_B1_A2_A1_A1_A2

### Relational analysis result of IS_A2_B2_B1_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0658013, 0.0482673, -0.0762538, 0.0536449, -0.1194462, 0.1245211
1: -0.0609188, 0.0571156, -0.0713606, 0.0655946, -0.1265134, 0.1284762
2: -0.0654344, 0.1532424, -0.0768136, 0.1661079, -0.2315423, 0.2300560
3: -0.0420492, 0.0672227, -0.0519378, 0.0759409, -0.1179901, 0.1191605
4: -0.0729162, 0.0751083, -0.0833249, 0.0881682, -0.1610843, 0.1584331
5: -0.0482345, 0.0843874, -0.0588421, 0.0961366, -0.1443710, 0.1432295
6: -0.1176975, 0.0885643, -0.1298168, 0.1000656, -0.2177631, 0.2183811
7: 0.8178881, 1.0168926, 0.8012925, 1.0230310, -0.2051429, 0.2156000
8: -0.1022364, 0.1137417, -0.1134029, 0.1275731, -0.2298095, 0.2271446
9: -0.0893285, 0.1197839, -0.0977377, 0.1410723, -0.2304008, 0.2175215

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_B1_A2_A1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.01 seconds

## Relational analysis of IS_A2_B2_B1_A2_A1_A2_B2

### Relational analysis result of IS_A2_B2_B1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.04 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0575539, 0.0440243, -0.0647340, 0.0477182, -0.1052722, 0.1087583
1: -0.0526799, 0.0504255, -0.0598526, 0.0562498, -0.1089297, 0.1102781
2: -0.0564558, 0.1430912, -0.0642724, 0.1519287, -0.2083845, 0.2073635
3: -0.0342469, 0.0603437, -0.0410395, 0.0663325, -0.1005794, 0.1013833
4: -0.0647034, 0.0648036, -0.0718533, 0.0737747, -0.1384781, 0.1366569
5: -0.0398648, 0.0751169, -0.0471514, 0.0831877, -0.1230524, 0.1222683
6: -0.1081349, 0.0794894, -0.1164600, 0.0873899, -0.1955248, 0.1959493
7: 0.8309824, 1.0120493, 0.8195826, 1.0162657, -0.1852833, 0.1924667
8: -0.0934258, 0.1028282, -0.1010963, 0.1123293, -0.2057551, 0.2039244
9: -0.0826934, 0.1029867, -0.0884699, 0.1176101, -0.2003035, 0.1914566

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_B1_A2_A2_A1_A1

### Relational analysis result of IS_A2_B2_B1_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.10 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2_A1_A2

### Relational analysis result of IS_A2_B2_B1_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0760707, 0.0535507, -0.0742301, 0.0526037, -0.1286744, 0.1277808
1: -0.0711778, 0.0654462, -0.0693390, 0.0639530, -0.1351308, 0.1347852
2: -0.0766144, 0.1658827, -0.0746104, 0.1636170, -0.2402314, 0.2404931
3: -0.0517647, 0.0757882, -0.0500233, 0.0742530, -0.1260177, 0.1258115
4: -0.0831426, 0.0879395, -0.0813096, 0.0856396, -0.1687822, 0.1692491
5: -0.0586564, 0.0959309, -0.0567883, 0.0938619, -0.1525182, 0.1527192
6: -0.1296046, 0.0998643, -0.1274703, 0.0978389, -0.2274435, 0.2273346
7: 0.8015831, 1.0229235, 0.8045056, 1.0218424, -0.2202593, 0.2184178
8: -0.1132074, 0.1273309, -0.1112410, 0.1248952, -0.2381026, 0.2385719
9: -0.0975904, 0.1406997, -0.0961096, 0.1369506, -0.2345410, 0.2368093

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_B1_A2_A2_A2_B1

### Relational analysis result of IS_A2_B2_B1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.61 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2_A2_B2

### Relational analysis result of IS_A2_B2_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.00 seconds

## BFS IS instance: IS_A2_B2_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1017453, 0.0667596, -0.0509671, 0.0406356, -0.1423808, 0.1177267
1: -0.0968262, 0.0862731, -0.0460998, 0.0450823, -0.1419085, 0.1323729
2: -0.1045651, 0.1974844, -0.0492850, 0.1349837, -0.2395488, 0.2467695
3: -0.0760541, 0.0972029, -0.0280153, 0.0548498, -0.1309039, 0.1252183
4: -0.1087096, 0.1200186, -0.0581442, 0.0565737, -0.1652832, 0.1781628
5: -0.0847120, 0.1247905, -0.0331802, 0.0677130, -0.1524249, 0.1579707
6: -0.1593734, 0.1281152, -0.1004977, 0.0722416, -0.2316150, 0.2286129
7: 0.7608195, 1.0380011, 0.8414404, 1.0081810, -0.2473615, 0.1965606
8: -0.1406358, 0.1613052, -0.0863890, 0.0941120, -0.2347478, 0.2476942
9: -0.1182460, 0.1929905, -0.0773942, 0.0895714, -0.2078174, 0.2703847

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 75

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_B2_B1_A1_B1_B1

### Relational analysis result of IS_A2_B2_B2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.97 seconds

## Relational analysis of IS_A2_B2_B2_B1_A1_B1_B2

### Relational analysis result of IS_A2_B2_B2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.98 seconds

## BFS IS instance: IS_A2_B2_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1017453, 0.0667596, -0.0984789, 0.0650791, -0.1668244, 0.1652385
1: -0.0968262, 0.0862731, -0.0935633, 0.0836235, -0.1804497, 0.1798364
2: -0.1045651, 0.1974844, -0.1010092, 0.1934640, -0.2980291, 0.2984937
3: -0.0760541, 0.0972029, -0.0729640, 0.0944785, -0.1705327, 0.1701669
4: -0.1087096, 0.1200186, -0.1054569, 0.1159374, -0.2246470, 0.2254756
5: -0.0847120, 0.1247905, -0.0813972, 0.1211190, -0.2058309, 0.2061877
6: -0.1593734, 0.1281152, -0.1555862, 0.1245212, -0.2838946, 0.2837014
7: 0.7608195, 1.0380011, 0.7660055, 1.0360830, -0.2752635, 0.2719956
8: -0.1406358, 0.1613052, -0.1371464, 0.1569829, -0.2976187, 0.2984516
9: -0.1182460, 0.1929905, -0.1156182, 0.1863380, -0.3045840, 0.3086087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=54, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_B2_B1_A1_B2_B1

### Relational analysis result of IS_A2_B2_B2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.98 seconds

## Relational analysis of IS_A2_B2_B2_B1_A1_B2_B2

### Relational analysis result of IS_A2_B2_B2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.09 seconds

## BFS IS instance: IS_A2_B2_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0852520, 0.0582742, -0.0509671, 0.0406356, -0.1258876, 0.1092414
1: -0.0803497, 0.0728939, -0.0460998, 0.0450823, -0.1254320, 0.1189937
2: -0.0866095, 0.1771835, -0.0492850, 0.1349837, -0.2215932, 0.2264685
3: -0.0604507, 0.0834462, -0.0280153, 0.0548498, -0.1153005, 0.1114615
4: -0.0922854, 0.0994110, -0.0581442, 0.0565737, -0.1488591, 0.1575551
5: -0.0679739, 0.1062511, -0.0331802, 0.0677130, -0.1356868, 0.1394313
6: -0.1402499, 0.1099669, -0.1004977, 0.0722416, -0.2124915, 0.2104646
7: 0.7870061, 1.0283153, 0.8414404, 1.0081810, -0.2211749, 0.1868749
8: -0.1230159, 0.1394801, -0.0863890, 0.0941120, -0.2171279, 0.2258691
9: -0.1049769, 0.1593989, -0.0773942, 0.0895714, -0.1945483, 0.2367931

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 75

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_B2_B1_A2_B1_B1

### Relational analysis result of IS_A2_B2_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.00 seconds

## Relational analysis of IS_A2_B2_B2_B1_A2_B1_B2

### Relational analysis result of IS_A2_B2_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.14 seconds

## BFS IS instance: IS_A2_B2_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0852520, 0.0582742, -0.0984789, 0.0650791, -0.1503311, 0.1567532
1: -0.0803497, 0.0728939, -0.0935633, 0.0836235, -0.1639732, 0.1664572
2: -0.0866095, 0.1771835, -0.1010092, 0.1934640, -0.2800735, 0.2781928
3: -0.0604507, 0.0834462, -0.0729640, 0.0944785, -0.1549292, 0.1564102
4: -0.0922854, 0.0994110, -0.1054569, 0.1159374, -0.2082228, 0.2048679
5: -0.0679739, 0.1062511, -0.0813972, 0.1211190, -0.1890928, 0.1876483
6: -0.1402499, 0.1099669, -0.1555862, 0.1245212, -0.2647711, 0.2655531
7: 0.7870061, 1.0283153, 0.7660055, 1.0360830, -0.2490769, 0.2623098
8: -0.1230159, 0.1394801, -0.1371464, 0.1569829, -0.2799988, 0.2766265
9: -0.1049769, 0.1593989, -0.1156182, 0.1863380, -0.2913149, 0.2750171

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=54, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_B2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.07 seconds

## Relational analysis of IS_A2_B2_B2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1084586, 0.0702134, -0.0563629, 0.0434115, -0.1518702, 0.1265763
1: -0.1035328, 0.0917189, -0.0514900, 0.0494593, -0.1529920, 0.1432089
2: -0.1118737, 0.2057476, -0.0551591, 0.1416251, -0.2534988, 0.2609067
3: -0.0824053, 0.1028024, -0.0331201, 0.0593503, -0.1417556, 0.1359225
4: -0.1153948, 0.1284066, -0.0635173, 0.0633154, -0.1787102, 0.1919239
5: -0.0915250, 0.1323368, -0.0386560, 0.0737781, -0.1653031, 0.1709928
6: -0.1671573, 0.1355023, -0.1067539, 0.0781787, -0.2453361, 0.2422562
7: 0.7501606, 1.0419438, 0.8328735, 1.0113497, -0.2611891, 0.2090703
8: -0.1478078, 0.1701887, -0.0921533, 0.1012521, -0.2490599, 0.2623420
9: -0.1236470, 0.2066636, -0.0817352, 0.1005609, -0.2242079, 0.2883987

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 75

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_B2_B2_A1_B1_B1

### Relational analysis result of IS_A2_B2_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.93 seconds

## Relational analysis of IS_A2_B2_B2_B2_A1_B1_B2

### Relational analysis result of IS_A2_B2_B2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1084586, 0.0702134, -0.1037926, 0.0678129, -0.1762715, 0.1740060
1: -0.1035328, 0.0917189, -0.0988715, 0.0879339, -0.1914667, 0.1905905
2: -0.1118737, 0.2057476, -0.1067940, 0.2000045, -0.3118781, 0.3125416
3: -0.0824053, 0.1028024, -0.0779911, 0.0989106, -0.1813159, 0.1807935
4: -0.1153948, 0.1284066, -0.1107484, 0.1225766, -0.2379714, 0.2391549
5: -0.0915250, 0.1323368, -0.0867897, 0.1270919, -0.2186168, 0.2191264
6: -0.1671573, 0.1355023, -0.1617473, 0.1303681, -0.2975254, 0.2972496
7: 0.7501606, 1.0419438, 0.7575688, 1.0392036, -0.2890431, 0.2843750
8: -0.1478078, 0.1701887, -0.1428230, 0.1640144, -0.3118222, 0.3130118
9: -0.1236470, 0.2066636, -0.1198931, 0.1971603, -0.3208073, 0.3265567

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_B2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.99 seconds

## Relational analysis of IS_A2_B2_B2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0922607, 0.0618800, -0.0563629, 0.0434115, -0.1356722, 0.1182429
1: -0.0873513, 0.0785793, -0.0514900, 0.0494593, -0.1368106, 0.1300693
2: -0.0942397, 0.1858103, -0.0551591, 0.1416251, -0.2358648, 0.2409694
3: -0.0670813, 0.0892920, -0.0331201, 0.0593503, -0.1264315, 0.1224121
4: -0.0992647, 0.1081681, -0.0635173, 0.0633154, -0.1625802, 0.1716854
5: -0.0750866, 0.1141294, -0.0386560, 0.0737781, -0.1488648, 0.1527854
6: -0.1483764, 0.1176789, -0.1067539, 0.0781787, -0.2265551, 0.2244329
7: 0.7758781, 1.0324312, 0.8328735, 1.0113497, -0.2354716, 0.1995577
8: -0.1305033, 0.1487546, -0.0921533, 0.1012521, -0.2317554, 0.2409079
9: -0.1106155, 0.1736735, -0.0817352, 0.1005609, -0.2111764, 0.2554087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 75

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_B2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.00 seconds

## Relational analysis of IS_A2_B2_B2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0922607, 0.0618800, -0.1037926, 0.0678129, -0.1600736, 0.1656726
1: -0.0873513, 0.0785793, -0.0988715, 0.0879339, -0.1752853, 0.1774508
2: -0.0942397, 0.1858103, -0.1067940, 0.2000045, -0.2942442, 0.2926043
3: -0.0670813, 0.0892920, -0.0779911, 0.0989106, -0.1659919, 0.1672831
4: -0.0992647, 0.1081681, -0.1107484, 0.1225766, -0.2218414, 0.2189165
5: -0.0750866, 0.1141294, -0.0867897, 0.1270919, -0.2021785, 0.2009190
6: -0.1483764, 0.1176789, -0.1617473, 0.1303681, -0.2787445, 0.2794262
7: 0.7758781, 1.0324312, 0.7575688, 1.0392036, -0.2633256, 0.2748625
8: -0.1305033, 0.1487546, -0.1428230, 0.1640144, -0.2945178, 0.2915776
9: -0.1106155, 0.1736735, -0.1198931, 0.1971603, -0.3077758, 0.2935666

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_B2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.95 seconds

## Relational analysis of IS_A2_B2_B2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.00 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.87 seconds
IS_A1_B1_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.87
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207

## BFS IS instance: IS_A1_B1_B1_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0194264, 0.0201407, -0.0371849, 0.0343424, -0.0537688, 0.0573256
1: -0.0204092, 0.0096589, -0.0346080, 0.0321917, -0.0526008, 0.0442669
2: -0.0040117, 0.0834458, -0.0335791, 0.1179175, -0.1219292, 0.1170250
3: -0.0069664, 0.0286837, -0.0186101, 0.0447060, -0.0516724, 0.0472938
4: -0.0243897, 0.0243956, -0.0467433, 0.0411439, -0.0655337, 0.0711389
5: -0.0074250, 0.0335198, -0.0207051, 0.0534236, -0.0608486, 0.0542249
6: -0.0594565, 0.0291000, -0.0854793, 0.0572511, -0.1167077, 0.1145793
7: 0.9117100, 0.9877052, 0.8641027, 1.0006646, -0.0889546, 0.1236024
8: -0.0497410, 0.0495993, -0.0741592, 0.0787898, -0.1285308, 0.1237585
9: -0.0405110, 0.0382523, -0.0657003, 0.0673861, -0.1078971, 0.1039526

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 247

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_B1_A1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.05 seconds

## Relational analysis of IS_A1_B1_B1_A1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_B1_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0239419, 0.0262484, -0.0450521, 0.0379801, -0.0619220, 0.0713005
1: -0.0254758, 0.0163112, -0.0405137, 0.0400652, -0.0655409, 0.0568248
2: -0.0142728, 0.0978740, -0.0430250, 0.1278913, -0.1421641, 0.1408990
3: -0.0098420, 0.0355837, -0.0237002, 0.0496911, -0.0595331, 0.0592839
4: -0.0330251, 0.0291345, -0.0531430, 0.0497198, -0.0827448, 0.0822776
5: -0.0124083, 0.0372850, -0.0272350, 0.0618526, -0.0742610, 0.0645200
6: -0.0692792, 0.0391979, -0.0941632, 0.0659328, -0.1352120, 0.1333611
7: 0.8933475, 0.9924660, 0.8505541, 1.0048854, -0.1115379, 0.1419119
8: -0.0593149, 0.0618882, -0.0810672, 0.0874625, -0.1467774, 0.1429555
9: -0.0508485, 0.0474498, -0.0725947, 0.0794461, -0.1302945, 0.1200445

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_B1_A1_B1_A1_A2_A1

### Relational analysis result of IS_A1_B1_B1_A1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.95 seconds

## Relational analysis of IS_A1_B1_B1_A1_B1_A1_A2_A2

### Relational analysis result of IS_A1_B1_B1_A1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_B1_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0227577, 0.0247395, -0.0360122, 0.0337099, -0.0564676, 0.0607517
1: -0.0242284, 0.0145565, -0.0338456, 0.0308226, -0.0550510, 0.0484020
2: -0.0117467, 0.0941372, -0.0319388, 0.1163511, -0.1280978, 0.1260760
3: -0.0090872, 0.0338831, -0.0178381, 0.0439931, -0.0530803, 0.0517211
4: -0.0307789, 0.0279679, -0.0456445, 0.0400679, -0.0708468, 0.0736124
5: -0.0111601, 0.0363462, -0.0199915, 0.0519579, -0.0631180, 0.0563377
6: -0.0668211, 0.0365684, -0.0841466, 0.0557771, -0.1225981, 0.1207150
7: 0.8982508, 0.9912592, 0.8664470, 0.9999660, -0.1017152, 0.1248122
8: -0.0569331, 0.0587643, -0.0729654, 0.0774631, -0.1343962, 0.1317298
9: -0.0482279, 0.0451422, -0.0645270, 0.0656453, -0.1138732, 0.1096692

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 247

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_B1_A1_B1_A2_A1_A1

### Relational analysis result of IS_A1_B1_B1_A1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.93 seconds

## Relational analysis of IS_A1_B1_B1_A1_B1_A2_A1_A2

### Relational analysis result of IS_A1_B1_B1_A1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_B1_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0303733, 0.0306681, -0.0435902, 0.0373952, -0.0677686, 0.0742583
1: -0.0301790, 0.0242388, -0.0391891, 0.0387993, -0.0689783, 0.0634279
2: -0.0240511, 0.1088186, -0.0414991, 0.1261858, -0.1502369, 0.1503178
3: -0.0141254, 0.0405649, -0.0228133, 0.0483895, -0.0625149, 0.0633782
4: -0.0403607, 0.0348937, -0.0520685, 0.0480892, -0.0884498, 0.0869622
5: -0.0165597, 0.0449096, -0.0257885, 0.0604975, -0.0770572, 0.0706981
6: -0.0777380, 0.0486888, -0.0926595, 0.0644213, -0.1421593, 0.1413482
7: 0.8777201, 0.9966066, 0.8527395, 1.0040920, -0.1263719, 0.1438671
8: -0.0672247, 0.0710830, -0.0799325, 0.0859582, -0.1531828, 0.1510155
9: -0.0588844, 0.0572743, -0.0714036, 0.0772911, -0.1361754, 0.1286779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_B1_A1_B1_A2_A2_A1

### Relational analysis result of IS_A1_B1_B1_A1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.11 seconds

## Relational analysis of IS_A1_B1_B1_A1_B1_A2_A2_A2

### Relational analysis result of IS_A1_B1_B1_A1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_B1_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0194264, 0.0201407, -0.0826367, 0.0569287, -0.0763551, 0.1027774
1: -0.0204092, 0.0096589, -0.0777371, 0.0707724, -0.0911816, 0.0873960
2: -0.0040117, 0.0834458, -0.0837624, 0.1739645, -0.1779762, 0.1672082
3: -0.0069664, 0.0286837, -0.0579764, 0.0812648, -0.0882312, 0.0866601
4: -0.0243897, 0.0243956, -0.0896810, 0.0961433, -0.1205331, 0.1140767
5: -0.0074250, 0.0335198, -0.0653198, 0.1033114, -0.1107365, 0.0988395
6: -0.0594565, 0.0291000, -0.1372177, 0.1070891, -0.1665457, 0.1663177
7: 0.9117100, 0.9877052, 0.7911584, 1.0267794, -0.1150694, 0.1965467
8: -0.0497410, 0.0495993, -0.1202219, 0.1360194, -0.1857604, 0.1698212
9: -0.0405110, 0.0382523, -0.1028729, 0.1540724, -0.1945834, 0.1411252

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 153

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.03 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_B1_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0239419, 0.0262484, -0.0924490, 0.0619769, -0.0859188, 0.1186974
1: -0.0254758, 0.0163112, -0.0875394, 0.0787320, -0.1042078, 0.1038506
2: -0.0142728, 0.0978740, -0.0944446, 0.1860420, -0.2003148, 0.1923186
3: -0.0098420, 0.0355837, -0.0672593, 0.0894491, -0.0992910, 0.1028431
4: -0.0330251, 0.0291345, -0.0994522, 0.1084033, -0.1414283, 0.1285867
5: -0.0124083, 0.0372850, -0.0752777, 0.1143409, -0.1267493, 0.1125626
6: -0.0692792, 0.0391979, -0.1485945, 0.1178861, -0.1871653, 0.1877925
7: 0.8933475, 0.9924660, 0.7755793, 1.0325419, -0.1391944, 0.2168867
8: -0.0593149, 0.0618882, -0.1307045, 0.1490037, -0.2083186, 0.1925927
9: -0.0508485, 0.0474498, -0.1107670, 0.1740569, -0.2249054, 0.1582169

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_A2_A1

### Relational analysis result of IS_A1_B1_B1_A1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.95 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_A2_A2

### Relational analysis result of IS_A1_B1_B1_A1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_B1_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0227577, 0.0247395, -0.0809661, 0.0560692, -0.0788269, 0.1057055
1: -0.0242284, 0.0145565, -0.0760682, 0.0694172, -0.0936456, 0.0906247
2: -0.0117467, 0.0941372, -0.0819436, 0.1719081, -0.1836548, 0.1760809
3: -0.0090872, 0.0338831, -0.0563959, 0.0798713, -0.0889585, 0.0902790
4: -0.0307789, 0.0279679, -0.0880174, 0.0940559, -0.1248348, 0.1159853
5: -0.0111601, 0.0363462, -0.0636244, 0.1014335, -0.1125936, 0.0999706
6: -0.0668211, 0.0365684, -0.1352805, 0.1052509, -0.1720719, 0.1718490
7: 0.8982508, 0.9912592, 0.7938107, 1.0257983, -0.1275475, 0.1974485
8: -0.0569331, 0.0587643, -0.1184372, 0.1338087, -0.1907418, 0.1772015
9: -0.0482279, 0.0451422, -0.1015288, 0.1506698, -0.1988977, 0.1466711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 247

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_A1_A1

### Relational analysis result of IS_A1_B1_B1_A1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.12 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_A1_A2

### Relational analysis result of IS_A1_B1_B1_A1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_B1_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0303733, 0.0306681, -0.0908630, 0.0611610, -0.0915343, 0.1215312
1: -0.0301790, 0.0242388, -0.0859551, 0.0774456, -0.1076246, 0.1101939
2: -0.0240511, 0.1088186, -0.0927181, 0.1840899, -0.2081410, 0.2015367
3: -0.0141254, 0.0405649, -0.0657590, 0.0881263, -0.1022517, 0.1063239
4: -0.0403607, 0.0348937, -0.0978730, 0.1064218, -0.1467824, 0.1327666
5: -0.0165597, 0.0449096, -0.0736683, 0.1125582, -0.1291180, 0.1185778
6: -0.0777380, 0.0486888, -0.1467558, 0.1161410, -0.1938790, 0.1954446
7: 0.8777201, 0.9966066, 0.7780973, 1.0316104, -0.1538903, 0.2185094
8: -0.0672247, 0.0710830, -0.1290102, 0.1469051, -0.2141297, 0.2000932
9: -0.0588844, 0.0572743, -0.1094911, 0.1708269, -0.2297113, 0.1667654

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_A2_A1

### Relational analysis result of IS_A1_B1_B1_A1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.01 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_A2_A2

### Relational analysis result of IS_A1_B1_B1_A1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0202497, 0.0215436, -0.0422923, 0.0368484, -0.0570980, 0.0638358
1: -0.0215866, 0.0108403, -0.0381394, 0.0376156, -0.0592022, 0.0489796
2: -0.0063963, 0.0862232, -0.0400774, 0.1245912, -0.1309875, 0.1263005
3: -0.0074886, 0.0302811, -0.0219840, 0.0475303, -0.0550189, 0.0522651
4: -0.0260217, 0.0254969, -0.0510964, 0.0465645, -0.0725861, 0.0765933
5: -0.0085164, 0.0343580, -0.0245599, 0.0592303, -0.0677467, 0.0589179
6: -0.0616149, 0.0309994, -0.0912534, 0.0630908, -0.1247057, 0.1222528
7: 0.9086359, 0.9887036, 0.8547829, 1.0034323, -0.0947964, 0.1339207
8: -0.0518887, 0.0521480, -0.0788887, 0.0845516, -0.1364403, 0.1310367
9: -0.0426778, 0.0402549, -0.0703490, 0.0752759, -0.1179537, 0.1106038

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_B1_A2_B1_A1_A1_A1

### Relational analysis result of IS_A1_B1_B1_A2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.19 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_A1_A1_A2

### Relational analysis result of IS_A1_B1_B1_A2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0239048, 0.0262012, -0.0405744, 0.0360973, -0.0600021, 0.0667755
1: -0.0254367, 0.0162562, -0.0368739, 0.0359900, -0.0614267, 0.0531301
2: -0.0141937, 0.0977569, -0.0381299, 0.1224013, -0.1365950, 0.1358868
3: -0.0098183, 0.0355304, -0.0208450, 0.0466839, -0.0565022, 0.0563754
4: -0.0329547, 0.0290980, -0.0497918, 0.0444706, -0.0774253, 0.0788898
5: -0.0123692, 0.0372556, -0.0229880, 0.0574900, -0.0698592, 0.0602436
6: -0.0692022, 0.0391156, -0.0893225, 0.0613406, -0.1305428, 0.1284381
7: 0.8935011, 0.9924282, 0.8575891, 1.0026027, -0.1091016, 0.1348391
8: -0.0592403, 0.0617904, -0.0774712, 0.0826198, -0.1418602, 0.1392616
9: -0.0507664, 0.0473776, -0.0689558, 0.0725086, -0.1232750, 0.1163334

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_B1_A2_B1_A1_A2_A1

### Relational analysis result of IS_A1_B1_B1_A2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.24 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_A1_A2_A2

### Relational analysis result of IS_A1_B1_B1_A2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0258525, 0.0281912, -0.0514218, 0.0408695, -0.0667220, 0.0796130
1: -0.0271933, 0.0190325, -0.0465540, 0.0454511, -0.0726444, 0.0655865
2: -0.0177765, 0.1026848, -0.0497800, 0.1355433, -0.1533198, 0.1524648
3: -0.0111577, 0.0377733, -0.0284455, 0.0552290, -0.0663867, 0.0662187
4: -0.0360579, 0.0308498, -0.0585969, 0.0571418, -0.0931997, 0.0894467
5: -0.0140154, 0.0391700, -0.0336416, 0.0682240, -0.0822394, 0.0728116
6: -0.0725676, 0.0429166, -0.1010249, 0.0727418, -0.1453095, 0.1439414
7: 0.8869001, 0.9940618, 0.8407185, 1.0084480, -0.1215479, 0.1533433
8: -0.0625498, 0.0659102, -0.0868747, 0.0947137, -0.1572635, 0.1527848
9: -0.0542894, 0.0506964, -0.0777600, 0.0904975, -0.1447869, 0.1284564

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_B1_A2_B1_A2_A1_A1

### Relational analysis result of IS_A1_B1_B1_A2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.98 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_A2_A1_A2

### Relational analysis result of IS_A1_B1_B1_A2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0339394, 0.0325918, -0.0495285, 0.0398954, -0.0738349, 0.0821203
1: -0.0324978, 0.0284025, -0.0446626, 0.0439153, -0.0764130, 0.0730651
2: -0.0290394, 0.1135822, -0.0477188, 0.1332129, -0.1622522, 0.1613010
3: -0.0164733, 0.0427329, -0.0266543, 0.0536498, -0.0701232, 0.0693873
4: -0.0437022, 0.0381659, -0.0567116, 0.0547761, -0.0984783, 0.0948775
5: -0.0187300, 0.0493670, -0.0317202, 0.0660959, -0.0848259, 0.0810872
6: -0.0817909, 0.0531715, -0.0988296, 0.0706586, -0.1524495, 0.1520011
7: 0.8705908, 0.9987312, 0.8437245, 1.0073361, -0.1367453, 0.1550066
8: -0.0708552, 0.0751178, -0.0848521, 0.0922083, -0.1630634, 0.1599699
9: -0.0624528, 0.0625682, -0.0762368, 0.0866413, -0.1490941, 0.1388050

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_B1_A2_B1_A2_A2_A1

### Relational analysis result of IS_A1_B1_B1_A2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.90 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_A2_A2_A2

### Relational analysis result of IS_A1_B1_B1_A2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0202497, 0.0215436, -0.0913330, 0.0614028, -0.0816524, 0.1128766
1: -0.0215866, 0.0108403, -0.0864246, 0.0778268, -0.0994134, 0.0972649
2: -0.0063963, 0.0862232, -0.0932298, 0.1846684, -0.1910647, 0.1794529
3: -0.0074886, 0.0302811, -0.0662036, 0.0885182, -0.0960068, 0.0964848
4: -0.0260217, 0.0254969, -0.0983410, 0.1070090, -0.1330307, 0.1238379
5: -0.0085164, 0.0343580, -0.0741452, 0.1130866, -0.1216030, 0.1085032
6: -0.0616149, 0.0309994, -0.1473008, 0.1166581, -0.1782730, 0.1783001
7: 0.9086359, 0.9887036, 0.7773510, 1.0318863, -0.1232504, 0.2113526
8: -0.0518887, 0.0521480, -0.1295123, 0.1475270, -0.1994158, 0.1816603
9: -0.0426778, 0.0402549, -0.1098692, 0.1717841, -0.2144618, 0.1501240

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=54, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_A1_A1

### Relational analysis result of IS_A1_B1_B1_A2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.02 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_A1_A2

### Relational analysis result of IS_A1_B1_B1_A2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0258525, 0.0281912, -0.1011167, 0.0664362, -0.0922887, 0.1293079
1: -0.0271933, 0.0190325, -0.0961983, 0.0857632, -0.1129565, 0.1152308
2: -0.0177765, 0.1026848, -0.1038808, 0.1967107, -0.2144872, 0.2065656
3: -0.0111577, 0.0377733, -0.0754594, 0.0966787, -0.1078364, 0.1132327
4: -0.0360579, 0.0308498, -0.1080836, 0.1192332, -0.1552910, 0.1389334
5: -0.0140154, 0.0391700, -0.0840740, 0.1240840, -0.1380993, 0.1232440
6: -0.0725676, 0.0429166, -0.1586446, 0.1274235, -0.1999912, 0.2015612
7: 0.8869001, 0.9940618, 0.7618175, 1.0376320, -0.1507319, 0.2322444
8: -0.0625498, 0.0659102, -0.1399643, 0.1604734, -0.2230233, 0.2058744
9: -0.0542894, 0.0506964, -0.1177403, 0.1917103, -0.2459997, 0.1684367

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=54, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_A2_A1

### Relational analysis result of IS_A1_B1_B1_A2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.99 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_A2_A2

### Relational analysis result of IS_A1_B1_B1_A2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0239048, 0.0262012, -0.0894115, 0.0604141, -0.0843189, 0.1156126
1: -0.0254367, 0.0162562, -0.0845050, 0.0762680, -0.1017047, 0.1007612
2: -0.0141937, 0.0977569, -0.0911378, 0.1823032, -0.1964969, 0.1888947
3: -0.0098183, 0.0355304, -0.0643857, 0.0869155, -0.0967338, 0.0999161
4: -0.0329547, 0.0290980, -0.0964274, 0.1046081, -0.1375628, 0.1255254
5: -0.0123692, 0.0372556, -0.0721951, 0.1109266, -0.1232959, 0.1094506
6: -0.0692022, 0.0391156, -0.1450727, 0.1145437, -0.1837459, 0.1841883
7: 0.8935011, 0.9924282, 0.7804019, 1.0307579, -0.1372568, 0.2120262
8: -0.0592403, 0.0617904, -0.1274594, 0.1449842, -0.2042245, 0.1892498
9: -0.0507664, 0.0473776, -0.1083232, 0.1678705, -0.2186368, 0.1557008

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=54, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_A1_A1

### Relational analysis result of IS_A1_B1_B1_A2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.00 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_A1_A2

### Relational analysis result of IS_A1_B1_B1_A2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0339394, 0.0325918, -0.0992918, 0.0654973, -0.0994367, 0.1318835
1: -0.0324978, 0.0284025, -0.0943752, 0.0842829, -0.1167806, 0.1227777
2: -0.0290394, 0.1135822, -0.1018941, 0.1944645, -0.2235038, 0.2154763
3: -0.0164733, 0.0427329, -0.0737329, 0.0951565, -0.1116298, 0.1164659
4: -0.0437022, 0.0381659, -0.1062663, 0.1169530, -0.1606553, 0.1444322
5: -0.0187300, 0.0493670, -0.0822220, 0.1220326, -0.1407626, 0.1315890
6: -0.0817909, 0.0531715, -0.1565287, 0.1254155, -0.2072063, 0.2097002
7: 0.8705908, 0.9987312, 0.7647149, 1.0365604, -0.1659696, 0.2340163
8: -0.0708552, 0.0751178, -0.1380146, 0.1580585, -0.2289137, 0.2131324
9: -0.0624528, 0.0625682, -0.1162721, 0.1879933, -0.2504461, 0.1788403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=54, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_A2_A1

### Relational analysis result of IS_A1_B1_B1_A2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.00 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_A2_A2

### Relational analysis result of IS_A1_B1_B1_A2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0453374, 0.0387401, -0.0327934, 0.0319736, -0.0773110, 0.0715335
1: -0.0399089, 0.0417102, -0.0317526, 0.0270645, -0.0669734, 0.0734628
2: -0.0449830, 0.1288076, -0.0274364, 0.1120514, -0.1570344, 0.1562439
3: -0.0239777, 0.0496624, -0.0157188, 0.0420363, -0.0660140, 0.0653812
4: -0.0543825, 0.0486247, -0.0426284, 0.0371144, -0.0914969, 0.0912531
5: -0.0256666, 0.0636137, -0.0180326, 0.0479346, -0.0736012, 0.0816463
6: -0.0947446, 0.0674992, -0.0804884, 0.0517310, -0.1464756, 0.1479876
7: 0.8478043, 1.0055215, 0.8728819, 0.9980483, -0.1502440, 0.1326396
8: -0.0824589, 0.0880140, -0.0696885, 0.0738212, -0.1562802, 0.1577024
9: -0.0738582, 0.0794886, -0.0613061, 0.0608669, -0.1347252, 0.1407947

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_B2_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_B2_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.89 seconds

## Relational analysis of IS_A1_B1_B2_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0441554, 0.0381024, -0.0408202, 0.0363034, -0.0804587, 0.0789226
1: -0.0391403, 0.0403300, -0.0369717, 0.0364361, -0.0755763, 0.0773017
2: -0.0433294, 0.1272285, -0.0386642, 0.1227735, -0.1661029, 0.1658927
3: -0.0231994, 0.0489437, -0.0210036, 0.0469161, -0.0701155, 0.0699473
4: -0.0532749, 0.0475400, -0.0501497, 0.0444797, -0.0977545, 0.0976897
5: -0.0249472, 0.0621362, -0.0229175, 0.0579675, -0.0829146, 0.0850536
6: -0.0934012, 0.0660132, -0.0896108, 0.0618209, -0.1552221, 0.1556241
7: 0.8501676, 1.0048172, 0.8568351, 1.0028303, -0.1526626, 0.1479822
8: -0.0812555, 0.0866764, -0.0778602, 0.0829030, -0.1641585, 0.1645366
9: -0.0726754, 0.0777337, -0.0693381, 0.0727827, -0.1454581, 0.1470718

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_B2_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_B2_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.01 seconds

## Relational analysis of IS_A1_B1_B2_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_B2_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0215371, 0.0231841, -0.0282554, 0.0295257, -0.0510628, 0.0514395
1: -0.0229427, 0.0127479, -0.0288020, 0.0217661, -0.0447089, 0.0415499
2: -0.0091428, 0.0902857, -0.0210886, 0.1059897, -0.1151325, 0.1113743
3: -0.0083092, 0.0321301, -0.0127310, 0.0392774, -0.0475866, 0.0448611
4: -0.0284637, 0.0267653, -0.0383762, 0.0329504, -0.0614140, 0.0651415
5: -0.0098735, 0.0353786, -0.0152709, 0.0422624, -0.0521359, 0.0506495
6: -0.0642874, 0.0338581, -0.0753311, 0.0460266, -0.1103140, 0.1091892
7: 0.9033049, 0.9900156, 0.8819540, 0.9953449, -0.0920400, 0.1080616
8: -0.0544782, 0.0555443, -0.0650686, 0.0686868, -0.1231650, 0.1206129
9: -0.0455268, 0.0427637, -0.0567651, 0.0541303, -0.0996571, 0.0995288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 153

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_A1_B1_A2_A1_A1

### Relational analysis result of IS_A1_B1_B2_A1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.97 seconds

## Relational analysis of IS_A1_B1_B2_A1_B1_A2_A1_A2

### Relational analysis result of IS_A1_B1_B2_A1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0286158, 0.0297201, -0.0350764, 0.0332051, -0.0618209, 0.0647965
1: -0.0290363, 0.0221869, -0.0332370, 0.0297299, -0.0587662, 0.0554239
2: -0.0215927, 0.1064710, -0.0306297, 0.1151009, -0.1366936, 0.1371008
3: -0.0129683, 0.0394964, -0.0172219, 0.0434242, -0.0563924, 0.0567183
4: -0.0387138, 0.0332810, -0.0447676, 0.0392092, -0.0779230, 0.0780486
5: -0.0154901, 0.0427128, -0.0194219, 0.0507881, -0.0662783, 0.0621347
6: -0.0757406, 0.0464796, -0.0830830, 0.0546007, -0.1303412, 0.1295626
7: 0.8812336, 0.9955596, 0.8683179, 0.9994085, -0.1181750, 0.1272417
8: -0.0654355, 0.0690945, -0.0720126, 0.0764042, -0.1418397, 0.1411071
9: -0.0571257, 0.0546652, -0.0635905, 0.0642560, -0.1213818, 0.1182558

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_A1_B1_A2_A2_A1

### Relational analysis result of IS_A1_B1_B2_A1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.13 seconds

## Relational analysis of IS_A1_B1_B2_A1_B1_A2_A2_A2

### Relational analysis result of IS_A1_B1_B2_A1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0420560, 0.0369700, -0.0756640, 0.0514378, -0.0934938, 0.1126340
1: -0.0377752, 0.0378790, -0.0627208, 0.0691938, -0.1069690, 0.1005998
2: -0.0403929, 0.1244242, -0.0779102, 0.1671327, -0.2075256, 0.2023345
3: -0.0218172, 0.0476674, -0.0441087, 0.0639733, -0.0857905, 0.0917761
4: -0.0513077, 0.0456136, -0.0764399, 0.0872399, -0.1385477, 0.1220535
5: -0.0236696, 0.0595122, -0.0550944, 0.0930366, -0.1167062, 0.1146066
6: -0.0910153, 0.0633743, -0.1287640, 0.0970892, -0.1881045, 0.1921384
7: 0.8543646, 1.0035664, 0.8002685, 1.0195453, -0.1651807, 0.2032979
8: -0.0791183, 0.0843012, -0.1064235, 0.1220769, -0.2011951, 0.1907247
9: -0.0705747, 0.0746173, -0.0974131, 0.1290339, -0.1996086, 0.1720304

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_B2_A1_B2_A1_A1_A1

### Relational analysis result of IS_A1_B1_B2_A1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.99 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2_A1_A1_A2

### Relational analysis result of IS_A1_B1_B2_A1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0502554, 0.0413929, -0.0743482, 0.0508625, -0.1011180, 0.1157411
1: -0.0431066, 0.0474522, -0.0617515, 0.0679486, -0.1110552, 0.1092037
2: -0.0518622, 0.1353769, -0.0764184, 0.1654553, -0.2173175, 0.2117954
3: -0.0272157, 0.0526523, -0.0432363, 0.0633249, -0.0905406, 0.0958886
4: -0.0589908, 0.0531374, -0.0754406, 0.0856361, -0.1446269, 0.1285780
5: -0.0286595, 0.0697609, -0.0538905, 0.0917036, -0.1203632, 0.1236514
6: -0.1003339, 0.0736812, -0.1272851, 0.0957486, -0.1960825, 0.2009663
7: 0.8379725, 1.0084513, 0.8024181, 1.0189099, -0.1809374, 0.2060333
8: -0.0874657, 0.0935783, -0.1053378, 0.1205972, -0.2080629, 0.1989161
9: -0.0787794, 0.0867893, -0.0963460, 0.1269142, -0.2056937, 0.1831354

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_B2_A1_B2_A1_A2_A1

### Relational analysis result of IS_A1_B1_B2_A1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.01 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2_A1_A2_A2

### Relational analysis result of IS_A1_B1_B2_A1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0313896, 0.0312163, -0.0756640, 0.0514378, -0.0828274, 0.1068803
1: -0.0308398, 0.0254254, -0.0627208, 0.0691938, -0.1000336, 0.0881462
2: -0.0254726, 0.1101761, -0.0779102, 0.1671327, -0.1926054, 0.1880864
3: -0.0147945, 0.0411827, -0.0441087, 0.0639733, -0.0787678, 0.0852914
4: -0.0413129, 0.0358262, -0.0764399, 0.0872399, -0.1285529, 0.1122661
5: -0.0171782, 0.0461799, -0.0550944, 0.0930366, -0.1102148, 0.1012743
6: -0.0788930, 0.0499662, -0.1287640, 0.0970892, -0.1759822, 0.1787303
7: 0.8756884, 0.9972120, 0.8002685, 1.0195453, -0.1438569, 0.1969435
8: -0.0682593, 0.0722328, -0.1064235, 0.1220769, -0.1903362, 0.1786563
9: -0.0599013, 0.0587829, -0.0974131, 0.1290339, -0.1889352, 0.1561961

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 153

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_B2_A1_B2_A2_A1_A1

### Relational analysis result of IS_A1_B1_B2_A1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.01 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2_A2_A1_A2

### Relational analysis result of IS_A1_B1_B2_A1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0394732, 0.0355768, -0.0743482, 0.0508625, -0.0903357, 0.1099249
1: -0.0360959, 0.0348634, -0.0617515, 0.0679486, -0.1040445, 0.0966149
2: -0.0367800, 0.1209742, -0.0764184, 0.1654553, -0.2022353, 0.1973926
3: -0.0201167, 0.0460972, -0.0432363, 0.0633249, -0.0834416, 0.0893335
4: -0.0488875, 0.0432437, -0.0754406, 0.0856361, -0.1345236, 0.1186843
5: -0.0220977, 0.0562838, -0.0538905, 0.0917036, -0.1138013, 0.1101743
6: -0.0880799, 0.0601276, -0.1272851, 0.0957486, -0.1838285, 0.1874126
7: 0.8595281, 1.0020278, 0.8024181, 1.0189099, -0.1593819, 0.1996097
8: -0.0764888, 0.0813789, -0.1053378, 0.1205972, -0.1970860, 0.1867166
9: -0.0679901, 0.0707830, -0.0963460, 0.1269142, -0.1949044, 0.1671291

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_B2_A1_B2_A2_A2_A1

### Relational analysis result of IS_A1_B1_B2_A1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.99 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2_A2_A2_A2

### Relational analysis result of IS_A1_B1_B2_A1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 1.19 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0415824, 0.0365380, -0.0233151, 0.0254497, -0.0670321, 0.0598531
1: -0.0376165, 0.0369439, -0.0248156, 0.0153824, -0.0529989, 0.0617595
2: -0.0392726, 0.1236863, -0.0129357, 0.0958962, -0.1351688, 0.1366220
3: -0.0215133, 0.0471806, -0.0094424, 0.0346836, -0.0561969, 0.0566230
4: -0.0505573, 0.0456993, -0.0318362, 0.0285170, -0.0790743, 0.0775355
5: -0.0239104, 0.0585111, -0.0117476, 0.0367881, -0.0606985, 0.0702588
6: -0.0904555, 0.0623676, -0.0679782, 0.0378061, -0.1282617, 0.1303458
7: 0.8559425, 1.0030895, 0.8959429, 0.9918272, -0.1358847, 0.1071467
8: -0.0783030, 0.0837533, -0.0580543, 0.0602347, -0.1385377, 0.1418076
9: -0.0697733, 0.0741324, -0.0494614, 0.0462284, -0.1160018, 0.1235939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=46, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_A2_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.89 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
time: 0.96 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.99 seconds
IS_A1_B1_B1_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B1_A2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B1_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A1_B2_A2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B1_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207
IS_A2_B2_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.1437207, upper bound: 0.1437207

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.80 + 596.81 = 600.61 seconds
