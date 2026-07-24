## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 3.518859945


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398)
1: (-1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865)
2: (-1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576)
3: (-1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979)
4: (-1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516)
5: (-1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349)
6: (-1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487)
7: (-1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090)
8: (-2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340)
9: (-1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.61 + 3.75 = 5.36 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -3.7040631, upper bound: 3.7040631

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 137

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7037028, upper bound: 3.7038323
time: 1.72 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7036607, upper bound: 3.7036607
time: 1.72 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 3.61 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 3.61
Output dim: 8, lower bound: -3.7037028, upper bound: 3.7038323
IS_A2, status: Status.UNKNOWN, split count: 1, time: 3.61
Output dim: 8, lower bound: -3.7036607, upper bound: 3.7036607

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -1.2718190, 1.0070211, -1.8503866, 1.3866532, -2.6584721, 2.8574076
1: -0.9931253, 0.9572994, -1.4591178, 1.3697687, -2.3628941, 2.4164171
2: -1.0603840, 1.6191735, -1.7938340, 1.9670236, -3.0274076, 3.4130075
3: -1.1187407, 0.9134134, -1.8323436, 1.2170542, -2.3357949, 2.7457571
4: -1.2598392, 1.0215979, -1.9638085, 1.4680431, -2.7278824, 2.9854064
5: -1.0370084, 1.0621163, -1.5762671, 1.4573678, -2.4943762, 2.6383834
6: -0.9544353, 1.1605603, -1.4516947, 1.6586541, -2.6130896, 2.6122551
7: -1.2069880, 1.1785308, -1.8006461, 1.7152629, -2.9222507, 2.9791770
8: -1.5936482, 1.5946894, -2.4382658, 1.6120684, -3.2057166, 4.0329552
9: -0.9599805, 1.2171302, -1.4212788, 1.6854837, -2.6454642, 2.6384091

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=134, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7036607, upper bound: 3.7036607
time: 1.65 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7036607, upper bound: 3.7036607
time: 1.90 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -1.2410746, 0.9897738, -1.7186071, 1.2966546, -2.5377293, 2.7083807
1: -0.9676212, 0.9382080, -1.3526791, 1.2740853, -2.2417064, 2.2908871
2: -1.0260965, 1.6039276, -1.6240987, 1.8876131, -2.9137096, 3.2280264
3: -1.0788692, 0.8986558, -1.6678380, 1.1464310, -2.2253003, 2.5664937
4: -1.2206101, 0.9985654, -1.8033099, 1.3649038, -2.5855141, 2.8018754
5: -1.0081573, 1.0435973, -1.4518367, 1.3659167, -2.3740740, 2.4954340
6: -0.9327656, 1.1345145, -1.3368192, 1.5420322, -2.4747977, 2.4713335
7: -1.1748067, 1.1500624, -1.6612500, 1.5905693, -2.7653761, 2.8113124
8: -1.5578110, 1.6340172, -2.2425489, 1.6077530, -3.1655641, 3.8765662
9: -0.9374577, 1.1966113, -1.3084379, 1.5777006, -2.5151582, 2.5050492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=35, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=135, inp2_unstable=151, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7032032, upper bound: 3.7033663
time: 1.75 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7036277, upper bound: 3.7036277
time: 2.20 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 5.53 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.53
Output dim: 8, lower bound: -3.7036607, upper bound: 3.7036607
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.53
Output dim: 8, lower bound: -3.7036607, upper bound: 3.7036607
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.53
Output dim: 8, lower bound: -3.7032032, upper bound: 3.7033663
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 5.53
Output dim: 8, lower bound: -3.7036277, upper bound: 3.7036277

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -1.2718190, 1.0070211, -1.2718190, 1.0070211, -2.2788401, 2.2788401
1: -0.9931253, 0.9572994, -0.9931253, 0.9572994, -1.9504247, 1.9504247
2: -1.0603840, 1.6191735, -1.0603840, 1.6191735, -2.6795576, 2.6795576
3: -1.1187407, 0.9134134, -1.1187407, 0.9134134, -2.0321541, 2.0321541
4: -1.2598392, 1.0215979, -1.2598392, 1.0215979, -2.2814369, 2.2814369
5: -1.0370084, 1.0621163, -1.0370084, 1.0621163, -2.0991247, 2.0991247
6: -0.9544353, 1.1605603, -0.9544353, 1.1605603, -2.1149955, 2.1149955
7: -1.2069880, 1.1785308, -1.2069880, 1.1785308, -2.3855188, 2.3855188
8: -1.5936482, 1.5946894, -1.5936482, 1.5946894, -3.1883376, 3.1883376
9: -0.9599805, 1.2171302, -0.9599805, 1.2171302, -2.1771107, 2.1771107

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=134, inp2_unstable=134, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 137

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7035704, upper bound: 3.7035534
time: 1.86 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7036775, upper bound: 3.7037884
time: 1.82 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -1.2718190, 1.0070211, -1.2410746, 0.9897738, -2.2615929, 2.2480955
1: -0.9931253, 0.9572994, -0.9676212, 0.9382080, -1.9313333, 1.9249206
2: -1.0603840, 1.6191735, -1.0260965, 1.6039276, -2.6643116, 2.6452699
3: -1.1187407, 0.9134134, -1.0788692, 0.8986558, -2.0173965, 1.9922826
4: -1.2598392, 1.0215979, -1.2206101, 0.9985654, -2.2584047, 2.2422080
5: -1.0370084, 1.0621163, -1.0081573, 1.0435973, -2.0806057, 2.0702734
6: -0.9544353, 1.1605603, -0.9327656, 1.1345145, -2.0889497, 2.0933259
7: -1.2069880, 1.1785308, -1.1748067, 1.1500624, -2.3570504, 2.3533375
8: -1.5936482, 1.5946894, -1.5578110, 1.6340172, -3.2276652, 3.1525002
9: -0.9599805, 1.2171302, -0.9374577, 1.1966113, -2.1565919, 2.1545877

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=134, inp2_unstable=135, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 137

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7035704, upper bound: 3.7035534
time: 1.77 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7036775, upper bound: 3.7037884
time: 1.81 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -1.1954240, 0.9606574, -1.6678703, 1.2610711, -2.4564950, 2.6285276
1: -0.9315046, 0.9067385, -1.3128372, 1.2345343, -2.1660390, 2.2195759
2: -0.9714409, 1.5780686, -1.5623306, 1.8709507, -2.8423915, 3.1403992
3: -1.0235716, 0.8750880, -1.6041765, 1.1183331, -2.1419048, 2.4792645
4: -1.1643655, 0.9639504, -1.7400206, 1.3252621, -2.4896276, 2.7039709
5: -0.9668267, 1.0120473, -1.4030454, 1.3291075, -2.2959342, 2.4150927
6: -0.8934535, 1.0959954, -1.2853040, 1.4984332, -2.3918867, 2.3812995
7: -1.1291574, 1.1087347, -1.6104244, 1.5436842, -2.6728415, 2.7191591
8: -1.4914572, 1.6251546, -2.1621251, 1.5617703, -3.0532274, 3.7872796
9: -0.9029644, 1.1597379, -1.2677286, 1.5320884, -2.4350529, 2.4274664

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=129, inp2_unstable=148, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 137

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7031207, upper bound: 3.7031207
time: 1.92 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7031207, upper bound: 3.7033663
time: 1.70 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -1.2410746, 0.9897738, -1.6280794, 1.2357630, -2.4768376, 2.6178532
1: -0.9676212, 0.9382080, -1.2801648, 1.2081786, -2.1757998, 2.2183728
2: -1.0260965, 1.6039276, -1.5085274, 1.8368547, -2.8629513, 3.1124549
3: -1.0788692, 0.8986558, -1.5562381, 1.0979208, -2.1767900, 2.4548938
4: -1.2206101, 0.9985654, -1.6929722, 1.2947203, -2.5153303, 2.6915376
5: -1.0081573, 1.0435973, -1.3664560, 1.3034432, -2.3116004, 2.4100533
6: -0.9327656, 1.1345145, -1.2554197, 1.4642329, -2.3969984, 2.3899341
7: -1.1748067, 1.1500624, -1.5691915, 1.5067245, -2.6815312, 2.7192540
8: -1.5578110, 1.6340172, -2.1064878, 1.5843378, -3.1421490, 3.7405050
9: -0.9374577, 1.1966113, -1.2370646, 1.5016688, -2.4391265, 2.4336758

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=135, inp2_unstable=148, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 137

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7033663, upper bound: 3.7032032
time: 1.85 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7033663, upper bound: 3.7036277
time: 1.66 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 5.11 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.11
Output dim: 8, lower bound: -3.7035704, upper bound: 3.7035534
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.11
Output dim: 8, lower bound: -3.7036775, upper bound: 3.7037884
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.11
Output dim: 8, lower bound: -3.7035704, upper bound: 3.7035534
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.11
Output dim: 8, lower bound: -3.7036775, upper bound: 3.7037884
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.11
Output dim: 8, lower bound: -3.7031207, upper bound: 3.7031207
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.11
Output dim: 8, lower bound: -3.7031207, upper bound: 3.7033663
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.11
Output dim: 8, lower bound: -3.7033663, upper bound: 3.7032032
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.11
Output dim: 8, lower bound: -3.7033663, upper bound: 3.7036277

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -1.2215023, 0.9734242, -1.2257935, 0.9775702, -2.1990724, 2.1992178
1: -0.9535491, 0.9201962, -0.9562844, 0.9256201, -1.8791692, 1.8764806
2: -1.0033677, 1.6031204, -1.0050900, 1.5934522, -2.5968199, 2.6082106
3: -1.0558460, 0.8875699, -1.0623252, 0.8900506, -1.9458966, 1.9498951
4: -1.1962926, 0.9837828, -1.2032766, 0.9868163, -2.1831088, 2.1870594
5: -0.9916061, 1.0255891, -0.9954468, 1.0305057, -2.0221119, 2.0210359
6: -0.9078528, 1.1178652, -0.9152491, 1.1216054, -2.0294580, 2.0331144
7: -1.1573603, 1.1332127, -1.1608573, 1.1369524, -2.2943127, 2.2940700
8: -1.5236320, 1.5489385, -1.5276970, 1.5861120, -3.1097436, 3.0766354
9: -0.9211030, 1.1734855, -0.9249562, 1.1801653, -2.1012683, 2.0984416

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7037790, upper bound: 3.7037790
time: 1.36 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7037790, upper bound: 3.7037830
time: 1.42 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1.1790082, 0.9477135, -1.2718190, 1.0070211, -2.1860294, 2.2195325
1: -0.9194198, 0.8935461, -0.9931253, 0.9572994, -1.8767191, 1.8866714
2: -0.9493456, 1.5668555, -1.0603840, 1.6191735, -2.5685191, 2.6272397
3: -1.0061703, 0.8660859, -1.1187407, 0.9134134, -1.9195837, 1.9848266
4: -1.1457578, 0.9516307, -1.2598392, 1.0215979, -2.1673555, 2.2114699
5: -0.9535775, 0.9979627, -1.0370084, 1.0621163, -2.0156937, 2.0349712
6: -0.8753743, 1.0821372, -0.9544353, 1.1605603, -2.0359344, 2.0365725
7: -1.1143459, 1.0947887, -1.2069880, 1.1785308, -2.2928767, 2.3017766
8: -1.4587713, 1.5714149, -1.5936482, 1.5946894, -3.0534606, 3.1650627
9: -0.8899143, 1.1417463, -0.9599805, 1.2171302, -2.1070445, 2.1017268

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=35, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=129, inp2_unstable=134, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7037830, upper bound: 3.7039194
time: 1.90 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7037830, upper bound: 3.7039879
time: 2.07 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1.2215023, 0.9734242, -1.1954240, 0.9606574, -2.1821597, 2.1688480
1: -0.9535491, 0.9201962, -0.9315046, 0.9067385, -1.8602877, 1.8517008
2: -1.0033677, 1.6031204, -0.9714409, 1.5780686, -2.5814362, 2.5745614
3: -1.0558460, 0.8875699, -1.0235716, 0.8750880, -1.9309340, 1.9111415
4: -1.1962926, 0.9837828, -1.1643655, 0.9639504, -2.1602430, 2.1481483
5: -0.9916061, 1.0255891, -0.9668267, 1.0120473, -2.0036535, 1.9924158
6: -0.9078528, 1.1178652, -0.8934535, 1.0959954, -2.0038481, 2.0113187
7: -1.1573603, 1.1332127, -1.1291574, 1.1087347, -2.2660952, 2.2623701
8: -1.5236320, 1.5489385, -1.4914572, 1.6251546, -3.1487861, 3.0403957
9: -0.9211030, 1.1734855, -0.9029644, 1.1597379, -2.0808411, 2.0764499

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=129, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 137

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7032800, upper bound: 3.7034548
time: 1.58 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7032800, upper bound: 3.7035534
time: 1.52 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1.1790082, 0.9477135, -1.2410746, 0.9897738, -2.1687820, 2.1887879
1: -0.9194198, 0.8935461, -0.9676212, 0.9382080, -1.8576277, 1.8611673
2: -0.9493456, 1.5668555, -1.0260965, 1.6039276, -2.5532732, 2.5929520
3: -1.0061703, 0.8660859, -1.0788692, 0.8986558, -1.9048260, 1.9449551
4: -1.1457578, 0.9516307, -1.2206101, 0.9985654, -2.1443233, 2.1722407
5: -0.9535775, 0.9979627, -1.0081573, 1.0435973, -1.9971749, 2.0061200
6: -0.8753743, 1.0821372, -0.9327656, 1.1345145, -2.0098886, 2.0149028
7: -1.1143459, 1.0947887, -1.1748067, 1.1500624, -2.2644083, 2.2695954
8: -1.4587713, 1.5714149, -1.5578110, 1.6340172, -3.0927882, 3.1292257
9: -0.8899143, 1.1417463, -0.9374577, 1.1966113, -2.0865254, 2.0792041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=129, inp2_unstable=135, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 137

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7032958, upper bound: 3.7036319
time: 1.78 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7032958, upper bound: 3.7037884
time: 1.78 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -1.1934845, 0.9576916, -1.6678703, 1.2610711, -2.4545555, 2.6255617
1: -0.9307069, 0.9027662, -1.3128372, 1.2345343, -2.1652412, 2.2156034
2: -0.9715015, 1.5880483, -1.5623306, 1.8709507, -2.8424520, 3.1503789
3: -1.0202091, 0.8736806, -1.6041765, 1.1183331, -2.1385422, 2.4778571
4: -1.1604913, 0.9625165, -1.7400206, 1.3252621, -2.4857535, 2.7025371
5: -0.9652208, 1.0083866, -1.4030454, 1.3291075, -2.2943282, 2.4114320
6: -0.8874640, 1.0943329, -1.2853040, 1.4984332, -2.3858972, 2.3796368
7: -1.1280476, 1.1069928, -1.6104244, 1.5436842, -2.6717319, 2.7174172
8: -1.4906927, 1.5830584, -2.1621251, 1.5617703, -3.0524631, 3.7451835
9: -0.9008090, 1.1545565, -1.2677286, 1.5320884, -2.4328976, 2.4222851

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=129, inp2_unstable=148, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7031207, upper bound: 3.7031207
time: 1.64 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7031207, upper bound: 3.7031207
time: 1.84 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1.1462979, 0.9297220, -1.6678703, 1.2610711, -2.4073691, 2.5975924
1: -0.8925265, 0.8735266, -1.3128372, 1.2345343, -2.1270609, 2.1863637
2: -0.9128386, 1.5498278, -1.5623306, 1.8709507, -2.7837892, 3.1121583
3: -0.9655280, 0.8501917, -1.6041765, 1.1183331, -2.0838611, 2.4543681
4: -1.1045774, 0.9267977, -1.7400206, 1.3252621, -2.4298396, 2.6668184
5: -0.9233721, 0.9775290, -1.4030454, 1.3291075, -2.2524796, 2.3805745
6: -0.8512925, 1.0551265, -1.2853040, 1.4984332, -2.3497257, 2.3404305
7: -1.0800151, 1.0644822, -1.6104244, 1.5436842, -2.6236992, 2.6749067
8: -1.4183735, 1.6091639, -2.1621251, 1.5617703, -2.9801435, 3.7712889
9: -0.8666105, 1.1191626, -1.2677286, 1.5320884, -2.3986988, 2.3868911

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=127, inp2_unstable=148, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7031207, upper bound: 3.7033663
time: 1.63 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7031207, upper bound: 3.7033663
time: 1.60 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1.1934845, 0.9576916, -1.6280794, 1.2357630, -2.4292474, 2.5857711
1: -0.9307069, 0.9027662, -1.2801648, 1.2081786, -2.1388855, 2.1829309
2: -0.9715015, 1.5880483, -1.5085274, 1.8368547, -2.8083563, 3.0965757
3: -1.0202091, 0.8736806, -1.5562381, 1.0979208, -2.1181297, 2.4299188
4: -1.1604913, 0.9625165, -1.6929722, 1.2947203, -2.4552116, 2.6554887
5: -0.9652208, 1.0083866, -1.3664560, 1.3034432, -2.2686639, 2.3748426
6: -0.8874640, 1.0943329, -1.2554197, 1.4642329, -2.3516970, 2.3497527
7: -1.1280476, 1.1069928, -1.5691915, 1.5067245, -2.6347721, 2.6761842
8: -1.4906927, 1.5830584, -2.1064878, 1.5843378, -3.0750299, 3.6895461
9: -0.9008090, 1.1545565, -1.2370646, 1.5016688, -2.4024777, 2.3916211

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=129, inp2_unstable=148, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7031207, upper bound: 3.7032032
time: 1.63 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7031207, upper bound: 3.7032032
time: 1.58 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1.1462979, 0.9297220, -1.6280794, 1.2357630, -2.3820610, 2.5578012
1: -0.8925265, 0.8735266, -1.2801648, 1.2081786, -2.1007051, 2.1536913
2: -0.9128386, 1.5498278, -1.5085274, 1.8368547, -2.7496934, 3.0583553
3: -0.9655280, 0.8501917, -1.5562381, 1.0979208, -2.0634489, 2.4064298
4: -1.1045774, 0.9267977, -1.6929722, 1.2947203, -2.3992977, 2.6197701
5: -0.9233721, 0.9775290, -1.3664560, 1.3034432, -2.2268152, 2.3439851
6: -0.8512925, 1.0551265, -1.2554197, 1.4642329, -2.3155255, 2.3105464
7: -1.0800151, 1.0644822, -1.5691915, 1.5067245, -2.5867395, 2.6336737
8: -1.4183735, 1.6091639, -2.1064878, 1.5843378, -3.0027103, 3.7156515
9: -0.8666105, 1.1191626, -1.2370646, 1.5016688, -2.3682795, 2.3562272

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=127, inp2_unstable=148, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7031207, upper bound: 3.7036277
time: 1.66 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7031207, upper bound: 3.7036277
time: 1.77 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.91 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.91
Output dim: 8, lower bound: -3.7037790, upper bound: 3.7037790
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.91
Output dim: 8, lower bound: -3.7037790, upper bound: 3.7037830
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.91
Output dim: 8, lower bound: -3.7037830, upper bound: 3.7039194
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.91
Output dim: 8, lower bound: -3.7037830, upper bound: 3.7039879
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.91
Output dim: 8, lower bound: -3.7032800, upper bound: 3.7034548
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.91
Output dim: 8, lower bound: -3.7032800, upper bound: 3.7035534
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.91
Output dim: 8, lower bound: -3.7032958, upper bound: 3.7036319
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.91
Output dim: 8, lower bound: -3.7032958, upper bound: 3.7037884
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.91
Output dim: 8, lower bound: -3.7031207, upper bound: 3.7031207
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.91
Output dim: 8, lower bound: -3.7031207, upper bound: 3.7031207
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.91
Output dim: 8, lower bound: -3.7031207, upper bound: 3.7033663
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.91
Output dim: 8, lower bound: -3.7031207, upper bound: 3.7033663
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.91
Output dim: 8, lower bound: -3.7031207, upper bound: 3.7032032
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.91
Output dim: 8, lower bound: -3.7031207, upper bound: 3.7032032
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.91
Output dim: 8, lower bound: -3.7031207, upper bound: 3.7036277
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.91
Output dim: 8, lower bound: -3.7031207, upper bound: 3.7036277

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -1.2215023, 0.9734242, -1.2215023, 0.9734242, -2.1949265, 2.1949265
1: -0.9535491, 0.9201962, -0.9535491, 0.9201962, -1.8737453, 1.8737453
2: -1.0033677, 1.6031204, -1.0033677, 1.6031204, -2.6064882, 2.6064882
3: -1.0558460, 0.8875699, -1.0558460, 0.8875699, -1.9434159, 1.9434159
4: -1.1962926, 0.9837828, -1.1962926, 0.9837828, -2.1800754, 2.1800754
5: -0.9916061, 1.0255891, -0.9916061, 1.0255891, -2.0171952, 2.0171952
6: -0.9078528, 1.1178652, -0.9078528, 1.1178652, -2.0257180, 2.0257180
7: -1.1573603, 1.1332127, -1.1573603, 1.1332127, -2.2905731, 2.2905731
8: -1.5236320, 1.5489385, -1.5236320, 1.5489385, -3.0725706, 3.0725706
9: -0.9211030, 1.1734855, -0.9211030, 1.1734855, -2.0945885, 2.0945885

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 137

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7033540, upper bound: 3.7034503
time: 1.49 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7036106, upper bound: 3.7036106
time: 1.39 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -1.2215023, 0.9734242, -1.1790082, 0.9477135, -2.1692157, 2.1524324
1: -0.9535491, 0.9201962, -0.9194198, 0.8935461, -1.8470953, 1.8396159
2: -1.0033677, 1.6031204, -0.9493456, 1.5668555, -2.5702233, 2.5524659
3: -1.0558460, 0.8875699, -1.0061703, 0.8660859, -1.9219319, 1.8937402
4: -1.1962926, 0.9837828, -1.1457578, 0.9516307, -2.1479235, 2.1295404
5: -0.9916061, 1.0255891, -0.9535775, 0.9979627, -1.9895688, 1.9791666
6: -0.9078528, 1.1178652, -0.8753743, 1.0821372, -1.9899900, 1.9932395
7: -1.1573603, 1.1332127, -1.1143459, 1.0947887, -2.2521491, 2.2475586
8: -1.5236320, 1.5489385, -1.4587713, 1.5714149, -3.0950465, 3.0077100
9: -0.9211030, 1.1734855, -0.8899143, 1.1417463, -2.0628493, 2.0633998

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=129, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 137

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7033540, upper bound: 3.7034850
time: 1.45 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7036106, upper bound: 3.7036260
time: 1.41 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -1.1790082, 0.9477135, -1.2215023, 0.9734242, -2.1524324, 2.1692157
1: -0.9194198, 0.8935461, -0.9535491, 0.9201962, -1.8396159, 1.8470953
2: -0.9493456, 1.5668555, -1.0033677, 1.6031204, -2.5524659, 2.5702233
3: -1.0061703, 0.8660859, -1.0558460, 0.8875699, -1.8937402, 1.9219319
4: -1.1457578, 0.9516307, -1.1962926, 0.9837828, -2.1295404, 2.1479235
5: -0.9535775, 0.9979627, -0.9916061, 1.0255891, -1.9791666, 1.9895688
6: -0.8753743, 1.0821372, -0.9078528, 1.1178652, -1.9932395, 1.9899900
7: -1.1143459, 1.0947887, -1.1573603, 1.1332127, -2.2475586, 2.2521491
8: -1.4587713, 1.5714149, -1.5236320, 1.5489385, -3.0077100, 3.0950465
9: -0.8899143, 1.1417463, -0.9211030, 1.1734855, -2.0633998, 2.0628493

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=129, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7034323, upper bound: 3.7036294
time: 1.39 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7036261, upper bound: 3.7037266
time: 1.38 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -1.1790082, 0.9477135, -1.1790082, 0.9477135, -2.1267219, 2.1267219
1: -0.9194198, 0.8935461, -0.9194198, 0.8935461, -1.8129659, 1.8129659
2: -0.9493456, 1.5668555, -0.9493456, 1.5668555, -2.5162010, 2.5162010
3: -1.0061703, 0.8660859, -1.0061703, 0.8660859, -1.8722562, 1.8722562
4: -1.1457578, 0.9516307, -1.1457578, 0.9516307, -2.0973885, 2.0973885
5: -0.9535775, 0.9979627, -0.9535775, 0.9979627, -1.9515402, 1.9515402
6: -0.8753743, 1.0821372, -0.8753743, 1.0821372, -1.9575115, 1.9575115
7: -1.1143459, 1.0947887, -1.1143459, 1.0947887, -2.2091346, 2.2091346
8: -1.4587713, 1.5714149, -1.4587713, 1.5714149, -3.0301857, 3.0301859
9: -0.8899143, 1.1417463, -0.8899143, 1.1417463, -2.0316606, 2.0316606

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=129, inp2_unstable=129, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7034323, upper bound: 3.7037464
time: 1.40 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7036261, upper bound: 3.7038223
time: 1.83 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -1.2215023, 0.9734242, -1.1934845, 0.9576916, -2.1791940, 2.1669087
1: -0.9535491, 0.9201962, -0.9307069, 0.9027662, -1.8563154, 1.8509030
2: -1.0033677, 1.6031204, -0.9715015, 1.5880483, -2.5914159, 2.5746219
3: -1.0558460, 0.8875699, -1.0202091, 0.8736806, -1.9295266, 1.9077790
4: -1.1962926, 0.9837828, -1.1604913, 0.9625165, -2.1588092, 2.1442742
5: -0.9916061, 1.0255891, -0.9652208, 1.0083866, -1.9999927, 1.9908099
6: -0.9078528, 1.1178652, -0.8874640, 1.0943329, -2.0021858, 2.0053291
7: -1.1573603, 1.1332127, -1.1280476, 1.1069928, -2.2643533, 2.2612603
8: -1.5236320, 1.5489385, -1.4906927, 1.5830584, -3.1066904, 3.0396314
9: -0.9211030, 1.1734855, -0.9008090, 1.1545565, -2.0756595, 2.0742946

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=129, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 137

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7024927, upper bound: 3.7000314
time: 4.08 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7029499, upper bound: 3.7004295
time: 1.94 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1.2215023, 0.9734242, -1.1462979, 0.9297220, -2.1512241, 2.1197221
1: -0.9535491, 0.9201962, -0.8925265, 0.8735266, -1.8270757, 1.8127227
2: -1.0033677, 1.6031204, -0.9128386, 1.5498278, -2.5531955, 2.5159590
3: -1.0558460, 0.8875699, -0.9655280, 0.8501917, -1.9060377, 1.8530979
4: -1.1962926, 0.9837828, -1.1045774, 0.9267977, -2.1230903, 2.0883603
5: -0.9916061, 1.0255891, -0.9233721, 0.9775290, -1.9691352, 1.9489613
6: -0.9078528, 1.1178652, -0.8512925, 1.0551265, -1.9629793, 1.9691577
7: -1.1573603, 1.1332127, -1.0800151, 1.0644822, -2.2218425, 2.2132277
8: -1.5236320, 1.5489385, -1.4183735, 1.6091639, -3.1327953, 2.9673119
9: -0.9211030, 1.1734855, -0.8666105, 1.1191626, -2.0402656, 2.0400960

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=127, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 137

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7024927, upper bound: 3.7027886
time: 1.69 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7029499, upper bound: 3.7030690
time: 1.64 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -1.1790082, 0.9477135, -1.1934845, 0.9576916, -2.1366997, 2.1411982
1: -0.9194198, 0.8935461, -0.9307069, 0.9027662, -1.8221860, 1.8242530
2: -0.9493456, 1.5668555, -0.9715015, 1.5880483, -2.5373940, 2.5383570
3: -1.0061703, 0.8660859, -1.0202091, 0.8736806, -1.8798509, 1.8862950
4: -1.1457578, 0.9516307, -1.1604913, 0.9625165, -2.1082745, 2.1121221
5: -0.9535775, 0.9979627, -0.9652208, 1.0083866, -1.9619641, 1.9631835
6: -0.8753743, 1.0821372, -0.8874640, 1.0943329, -1.9697073, 1.9696013
7: -1.1143459, 1.0947887, -1.1280476, 1.1069928, -2.2213387, 2.2228363
8: -1.4587713, 1.5714149, -1.4906927, 1.5830584, -3.0418296, 3.0621071
9: -0.8899143, 1.1417463, -0.9008090, 1.1545565, -2.0444708, 2.0425553

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=129, inp2_unstable=129, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7026123, upper bound: 3.7002222
time: 3.81 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7029751, upper bound: 3.7006293
time: 1.80 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -1.1790082, 0.9477135, -1.1462979, 0.9297220, -2.1087303, 2.0940113
1: -0.9194198, 0.8935461, -0.8925265, 0.8735266, -1.7929463, 1.7860726
2: -0.9493456, 1.5668555, -0.9128386, 1.5498278, -2.4991734, 2.4796941
3: -1.0061703, 0.8660859, -0.9655280, 0.8501917, -1.8563620, 1.8316139
4: -1.1457578, 0.9516307, -1.1045774, 0.9267977, -2.0725555, 2.0562081
5: -0.9535775, 0.9979627, -0.9233721, 0.9775290, -1.9311066, 1.9213349
6: -0.8753743, 1.0821372, -0.8512925, 1.0551265, -1.9305009, 1.9334297
7: -1.1143459, 1.0947887, -1.0800151, 1.0644822, -2.1788282, 2.1748037
8: -1.4587713, 1.5714149, -1.4183735, 1.6091639, -3.0679352, 2.9897876
9: -0.8899143, 1.1417463, -0.8666105, 1.1191626, -2.0090768, 2.0083568

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=129, inp2_unstable=127, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7026123, upper bound: 3.7031703
time: 1.50 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7029751, upper bound: 3.7034843
time: 1.39 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -1.1934845, 0.9576916, -1.2215023, 0.9734242, -2.1669087, 2.1791940
1: -0.9307069, 0.9027662, -0.9535491, 0.9201962, -1.8509030, 1.8563154
2: -0.9715015, 1.5880483, -1.0033677, 1.6031204, -2.5746219, 2.5914159
3: -1.0202091, 0.8736806, -1.0558460, 0.8875699, -1.9077790, 1.9295266
4: -1.1604913, 0.9625165, -1.1962926, 0.9837828, -2.1442742, 2.1588092
5: -0.9652208, 1.0083866, -0.9916061, 1.0255891, -1.9908099, 1.9999927
6: -0.8874640, 1.0943329, -0.9078528, 1.1178652, -2.0053291, 2.0021858
7: -1.1280476, 1.1069928, -1.1573603, 1.1332127, -2.2612603, 2.2643533
8: -1.4906927, 1.5830584, -1.5236320, 1.5489385, -3.0396314, 3.1066902
9: -0.9008090, 1.1545565, -0.9211030, 1.1734855, -2.0742946, 2.0756595

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=129, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6813741, upper bound: 3.6996222
time: 1.78 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7001756, upper bound: 3.7001756
time: 1.36 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -1.1934845, 0.9576916, -1.1871551, 0.9537766, -2.1472611, 2.1448467
1: -0.9307069, 0.9027662, -0.9257241, 0.8984902, -1.8291970, 1.8284904
2: -0.9715015, 1.5880483, -0.9638585, 1.5842650, -2.5557666, 2.5519068
3: -1.0202091, 0.8736806, -1.0126770, 0.8704512, -1.8906603, 1.8863575
4: -1.1604913, 0.9625165, -1.1527280, 0.9577714, -2.1182628, 2.1152444
5: -0.9652208, 1.0083866, -0.9595886, 1.0039959, -1.9692167, 1.9679753
6: -0.8874640, 1.0943329, -0.8820717, 1.0890374, -1.9765015, 1.9764047
7: -1.1280476, 1.1069928, -1.1217581, 1.1013004, -2.2293479, 2.2287509
8: -1.4906927, 1.5830584, -1.4811000, 1.5818691, -3.0725617, 3.0641584
9: -0.9008090, 1.1545565, -0.8961564, 1.1494105, -2.0502195, 2.0507131

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=129, inp2_unstable=129, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 137

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6813741, upper bound: 3.6996222
time: 1.50 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7001756, upper bound: 3.7001756
time: 1.85 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -1.1462979, 0.9297220, -1.2215023, 0.9734242, -2.1197221, 2.1512241
1: -0.8925265, 0.8735266, -0.9535491, 0.9201962, -1.8127227, 1.8270757
2: -0.9128386, 1.5498278, -1.0033677, 1.6031204, -2.5159590, 2.5531955
3: -0.9655280, 0.8501917, -1.0558460, 0.8875699, -1.8530979, 1.9060377
4: -1.1045774, 0.9267977, -1.1962926, 0.9837828, -2.0883603, 2.1230903
5: -0.9233721, 0.9775290, -0.9916061, 1.0255891, -1.9489613, 1.9691352
6: -0.8512925, 1.0551265, -0.9078528, 1.1178652, -1.9691577, 1.9629793
7: -1.0800151, 1.0644822, -1.1573603, 1.1332127, -2.2132277, 2.2218425
8: -1.4183735, 1.6091639, -1.5236320, 1.5489385, -2.9673119, 3.1327958
9: -0.8666105, 1.1191626, -0.9211030, 1.1734855, -2.0400960, 2.0402656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=127, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6877874, upper bound: 3.6197535
time: 1.76 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6221746, upper bound: 3.6193417
time: 1.23 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -1.1462979, 0.9297220, -1.1871551, 0.9537766, -2.1000745, 2.1168771
1: -0.8925265, 0.8735266, -0.9257241, 0.8984902, -1.7910167, 1.7992507
2: -0.9128386, 1.5498278, -0.9638585, 1.5842650, -2.4971037, 2.5136862
3: -0.9655280, 0.8501917, -1.0126770, 0.8704512, -1.8359792, 1.8628687
4: -1.1045774, 0.9267977, -1.1527280, 0.9577714, -2.0623488, 2.0795257
5: -0.9233721, 0.9775290, -0.9595886, 1.0039959, -1.9273680, 1.9371177
6: -0.8512925, 1.0551265, -0.8820717, 1.0890374, -1.9403299, 1.9371983
7: -1.0800151, 1.0644822, -1.1217581, 1.1013004, -2.1813154, 2.1862402
8: -1.4183735, 1.6091639, -1.4811000, 1.5818691, -3.0002422, 3.0902638
9: -0.8666105, 1.1191626, -0.8961564, 1.1494105, -2.0160210, 2.0153189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=127, inp2_unstable=129, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6877874, upper bound: 3.6197535
time: 1.52 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6221746, upper bound: 3.6193417
time: 1.46 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -1.1934845, 0.9576916, -1.1790082, 0.9477135, -2.1411982, 2.1366997
1: -0.9307069, 0.9027662, -0.9194198, 0.8935461, -1.8242530, 1.8221860
2: -0.9715015, 1.5880483, -0.9493456, 1.5668555, -2.5383570, 2.5373940
3: -1.0202091, 0.8736806, -1.0061703, 0.8660859, -1.8862950, 1.8798509
4: -1.1604913, 0.9625165, -1.1457578, 0.9516307, -2.1121221, 2.1082745
5: -0.9652208, 1.0083866, -0.9535775, 0.9979627, -1.9631835, 1.9619641
6: -0.8874640, 1.0943329, -0.8753743, 1.0821372, -1.9696013, 1.9697073
7: -1.1280476, 1.1069928, -1.1143459, 1.0947887, -2.2228363, 2.2213387
8: -1.4906927, 1.5830584, -1.4587713, 1.5714149, -3.0621071, 3.0418296
9: -0.9008090, 1.1545565, -0.8899143, 1.1417463, -2.0425553, 2.0444708

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=129, inp2_unstable=129, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6815866, upper bound: 3.7023945
time: 1.50 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7003953, upper bound: 3.7027810
time: 3.78 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1.1934845, 0.9576916, -1.1456043, 0.9291941, -2.1226788, 2.1032958
1: -0.9307069, 0.9027662, -0.8920125, 0.8729835, -1.8036903, 1.7947787
2: -0.9715015, 1.5880483, -0.9118081, 1.5491328, -2.5206342, 2.4998565
3: -1.0202091, 0.8736806, -0.9648210, 0.8498217, -1.8700308, 1.8385016
4: -1.1604913, 0.9625165, -1.1038303, 0.9262789, -2.0867703, 2.0663469
5: -0.9652208, 1.0083866, -0.9228160, 0.9769496, -1.9421704, 1.9312027
6: -0.8874640, 1.0943329, -0.8505538, 1.0545474, -1.9420115, 1.9448867
7: -1.1280476, 1.1069928, -1.0793865, 1.0638889, -2.1919365, 2.1863794
8: -1.4906927, 1.5830584, -1.4168493, 1.6075697, -3.0982621, 2.9999075
9: -0.9008090, 1.1545565, -0.8660536, 1.1184500, -2.0192590, 2.0206101

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=129, inp2_unstable=127, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6815866, upper bound: 3.7023945
time: 1.74 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7003953, upper bound: 3.7027810
time: 1.69 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -1.1462979, 0.9297220, -1.1790082, 0.9477135, -2.0940113, 2.1087303
1: -0.8925265, 0.8735266, -0.9194198, 0.8935461, -1.7860726, 1.7929463
2: -0.9128386, 1.5498278, -0.9493456, 1.5668555, -2.4796941, 2.4991734
3: -0.9655280, 0.8501917, -1.0061703, 0.8660859, -1.8316139, 1.8563620
4: -1.1045774, 0.9267977, -1.1457578, 0.9516307, -2.0562081, 2.0725555
5: -0.9233721, 0.9775290, -0.9535775, 0.9979627, -1.9213349, 1.9311066
6: -0.8512925, 1.0551265, -0.8753743, 1.0821372, -1.9334297, 1.9305009
7: -1.0800151, 1.0644822, -1.1143459, 1.0947887, -2.1748037, 2.1788282
8: -1.4183735, 1.6091639, -1.4587713, 1.5714149, -2.9897881, 3.0679352
9: -0.8666105, 1.1191626, -0.8899143, 1.1417463, -2.0083568, 2.0090768

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=127, inp2_unstable=129, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6900792, upper bound: 3.6227954
time: 1.48 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6223139, upper bound: 3.6223712
time: 1.27 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -1.1462979, 0.9297220, -1.1456043, 0.9291941, -2.0754919, 2.0753262
1: -0.8925265, 0.8735266, -0.8920125, 0.8729835, -1.7655100, 1.7655391
2: -0.9128386, 1.5498278, -0.9118081, 1.5491328, -2.4619713, 2.4616361
3: -0.9655280, 0.8501917, -0.9648210, 0.8498217, -1.8153497, 1.8150127
4: -1.1045774, 0.9267977, -1.1038303, 0.9262789, -2.0308564, 2.0306282
5: -0.9233721, 0.9775290, -0.9228160, 0.9769496, -1.9003217, 1.9003451
6: -0.8512925, 1.0551265, -0.8505538, 1.0545474, -1.9058399, 1.9056804
7: -1.0800151, 1.0644822, -1.0793865, 1.0638889, -2.1439040, 2.1438687
8: -1.4183735, 1.6091639, -1.4168493, 1.6075697, -3.0259426, 3.0260131
9: -0.8666105, 1.1191626, -0.8660536, 1.1184500, -1.9850606, 1.9852161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=127, inp2_unstable=127, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6900792, upper bound: 3.6227954
time: 1.93 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6223139, upper bound: 3.6223712
time: 1.27 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.68 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 8, lower bound: -3.7033540, upper bound: 3.7034503
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 8, lower bound: -3.7036106, upper bound: 3.7036106
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 8, lower bound: -3.7033540, upper bound: 3.7034850
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 8, lower bound: -3.7036106, upper bound: 3.7036260
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 8, lower bound: -3.7034323, upper bound: 3.7036294
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 8, lower bound: -3.7036261, upper bound: 3.7037266
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 8, lower bound: -3.7034323, upper bound: 3.7037464
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 8, lower bound: -3.7036261, upper bound: 3.7038223
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 8, lower bound: -3.7024927, upper bound: 3.7000314
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 8, lower bound: -3.7029499, upper bound: 3.7004295
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 8, lower bound: -3.7024927, upper bound: 3.7027886
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 8, lower bound: -3.7029499, upper bound: 3.7030690
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 8, lower bound: -3.7026123, upper bound: 3.7002222
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 8, lower bound: -3.7029751, upper bound: 3.7006293
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 8, lower bound: -3.7026123, upper bound: 3.7031703
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 8, lower bound: -3.7029751, upper bound: 3.7034843
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 8, lower bound: -3.6813741, upper bound: 3.6996222
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 8, lower bound: -3.7001756, upper bound: 3.7001756
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 8, lower bound: -3.6813741, upper bound: 3.6996222
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 8, lower bound: -3.7001756, upper bound: 3.7001756
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 8, lower bound: -3.6877874, upper bound: 3.6197535
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 8, lower bound: -3.6221746, upper bound: 3.6193417
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 8, lower bound: -3.6877874, upper bound: 3.6197535
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 8, lower bound: -3.6221746, upper bound: 3.6193417
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 8, lower bound: -3.6815866, upper bound: 3.7023945
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 8, lower bound: -3.7003953, upper bound: 3.7027810
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 8, lower bound: -3.6815866, upper bound: 3.7023945
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 8, lower bound: -3.7003953, upper bound: 3.7027810
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 8, lower bound: -3.6900792, upper bound: 3.6227954
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 8, lower bound: -3.6223139, upper bound: 3.6223712
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 8, lower bound: -3.6900792, upper bound: 3.6227954
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 8, lower bound: -3.6223139, upper bound: 3.6223712

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.7498992, 0.6906404, -1.1481541, 0.9273930, -1.6772921, 1.8387946
1: -0.5940168, 0.6083853, -0.8954868, 0.8710797, -1.4650965, 1.5038722
2: -0.4509220, 1.2591189, -0.9128944, 1.5537306, -2.0046525, 2.1720133
3: -0.5034613, 0.6497082, -0.9692515, 0.8503561, -1.3538175, 1.6189597
4: -0.6432903, 0.6250859, -1.1076493, 0.9283968, -1.5716871, 1.7327353
5: -0.5921451, 0.6899128, -0.9264235, 0.9749236, -1.5670687, 1.6163363
6: -0.5016267, 0.7464218, -0.8462256, 1.0566555, -1.5582821, 1.5926474
7: -0.6961854, 0.7124593, -1.0840650, 1.0669485, -1.7631339, 1.7965243
8: -0.7909718, 1.5304153, -1.4139940, 1.5455306, -2.3365023, 2.9444094
9: -0.5786676, 0.8021612, -0.8664832, 1.1146988, -1.6933664, 1.6686444

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=127, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5718975, upper bound: 3.6856204
time: 1.71 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5716330, upper bound: 3.6192978
time: 1.27 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.8849511, 0.7712959, -1.2215023, 0.9734242, -1.8583753, 1.9927982
1: -0.6948996, 0.6947341, -0.9535491, 0.9201962, -1.6150959, 1.6482832
2: -0.6027899, 1.3758440, -1.0033677, 1.6031204, -2.2059104, 2.3792117
3: -0.6593508, 0.7152038, -1.0558460, 0.8875699, -1.5469208, 1.7710497
4: -0.7941504, 0.7288600, -1.1962926, 0.9837828, -1.7779331, 1.9251527
5: -0.6923543, 0.7898130, -0.9916061, 1.0255891, -1.7179434, 1.7814190
6: -0.6204342, 0.8444942, -0.9078528, 1.1178652, -1.7382994, 1.7523470
7: -0.8214942, 0.8329949, -1.1573603, 1.1332127, -1.9547069, 1.9903553
8: -1.0156063, 1.5348587, -1.5236320, 1.5489385, -2.5645447, 3.0584900
9: -0.6755854, 0.9032372, -0.9211030, 1.1734855, -1.8490710, 1.8243402

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=94, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7034503, upper bound: 3.7033540
time: 1.42 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7034503, upper bound: 3.7036106
time: 2.06 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.7498992, 0.6906404, -1.1063476, 0.9023004, -1.6521995, 1.7969880
1: -0.5940168, 0.6083853, -0.8614897, 0.8453042, -1.4393210, 1.4698749
2: -0.4509220, 1.2591189, -0.8593577, 1.5175031, -1.9684252, 2.1184766
3: -0.5034613, 0.6497082, -0.9215403, 0.8294817, -1.3329430, 1.5712485
4: -0.6432903, 0.6250859, -1.0589498, 0.8964415, -1.5397317, 1.6840358
5: -0.5921451, 0.6899128, -0.8895599, 0.9473463, -1.5394914, 1.5794727
6: -0.5016267, 0.7464218, -0.8139750, 1.0220398, -1.5236664, 1.5603968
7: -0.6961854, 0.7124593, -1.0412529, 1.0290250, -1.7252103, 1.7537122
8: -0.7909718, 1.5304153, -1.3481930, 1.5677047, -2.3586764, 2.8786082
9: -0.5786676, 0.8021612, -0.8360571, 1.0831776, -1.6618452, 1.6382184

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=126, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5720452, upper bound: 3.6905329
time: 1.48 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5717673, upper bound: 3.6205211
time: 1.76 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.8849511, 0.7712959, -1.1790082, 0.9477135, -1.8326645, 1.9503042
1: -0.6948996, 0.6947341, -0.9194198, 0.8935461, -1.5884457, 1.6141539
2: -0.6027899, 1.3758440, -0.9493456, 1.5668555, -2.1696455, 2.3251896
3: -0.6593508, 0.7152038, -1.0061703, 0.8660859, -1.5254366, 1.7213740
4: -0.7941504, 0.7288600, -1.1457578, 0.9516307, -1.7457812, 1.8746178
5: -0.6923543, 0.7898130, -0.9535775, 0.9979627, -1.6903169, 1.7433906
6: -0.6204342, 0.8444942, -0.8753743, 1.0821372, -1.7025714, 1.7198685
7: -0.8214942, 0.8329949, -1.1143459, 1.0947887, -1.9162829, 1.9473407
8: -1.0156063, 1.5348587, -1.4587713, 1.5714149, -2.5870209, 2.9936295
9: -0.6755854, 0.9032372, -0.8899143, 1.1417463, -1.8173318, 1.7931515

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=94, inp2_unstable=129, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7036294, upper bound: 3.7034323
time: 1.47 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7036294, upper bound: 3.7036260
time: 1.55 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.7222407, 0.6742139, -1.1481541, 0.9273930, -1.6496336, 1.8223681
1: -0.5717611, 0.5904015, -0.8954868, 0.8710797, -1.4428408, 1.4858882
2: -0.4184340, 1.2256221, -0.9128944, 1.5537306, -1.9721646, 2.1385164
3: -0.4736900, 0.6343694, -0.9692515, 0.8503561, -1.3240461, 1.6036210
4: -0.6134740, 0.6030147, -1.1076493, 0.9283968, -1.5418708, 1.7106640
5: -0.5731425, 0.6689765, -0.9264235, 0.9749236, -1.5480661, 1.5954001
6: -0.4829884, 0.7237632, -0.8462256, 1.0566555, -1.5396440, 1.5699887
7: -0.6710814, 0.6871932, -1.0840650, 1.0669485, -1.7380300, 1.7712581
8: -0.7382430, 1.5527720, -1.4139940, 1.5455306, -2.2837734, 2.9667656
9: -0.5604296, 0.7814227, -0.8664832, 1.1146988, -1.6751285, 1.6479058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=74, inp2_unstable=127, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5724056, upper bound: 3.6858634
time: 1.34 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5721306, upper bound: 3.6194399
time: 1.43 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.8440781, 0.7487532, -1.2215023, 0.9734242, -1.8175023, 1.9702555
1: -0.6637570, 0.6703729, -0.9535491, 0.9201962, -1.5839531, 1.6239221
2: -0.5550326, 1.3367251, -1.0033677, 1.6031204, -2.1581531, 2.3400927
3: -0.6107832, 0.6956552, -1.0558460, 0.8875699, -1.4983530, 1.7515012
4: -0.7491193, 0.6973783, -1.1962926, 0.9837828, -1.7329021, 1.8936710
5: -0.6596397, 0.7621418, -0.9916061, 1.0255891, -1.6852288, 1.7537479
6: -0.5867524, 0.8152532, -0.9078528, 1.1178652, -1.7046176, 1.7231059
7: -0.7816032, 0.7968085, -1.1573603, 1.1332127, -1.9148159, 1.9541688
8: -0.9458506, 1.5565381, -1.5236320, 1.5489385, -2.4947891, 3.0801699
9: -0.6460938, 0.8747255, -0.9211030, 1.1734855, -1.8195794, 1.7958285

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7034850, upper bound: 3.7035749
time: 1.56 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7034850, upper bound: 3.7037266
time: 1.61 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.7222407, 0.6742139, -1.1063476, 0.9023004, -1.6245410, 1.7805614
1: -0.5717611, 0.5904015, -0.8614897, 0.8453042, -1.4170654, 1.4518912
2: -0.4184340, 1.2256221, -0.8593577, 1.5175031, -1.9359372, 2.0849798
3: -0.4736900, 0.6343694, -0.9215403, 0.8294817, -1.3031716, 1.5559096
4: -0.6134740, 0.6030147, -1.0589498, 0.8964415, -1.5099154, 1.6619645
5: -0.5731425, 0.6689765, -0.8895599, 0.9473463, -1.5204887, 1.5585363
6: -0.4829884, 0.7237632, -0.8139750, 1.0220398, -1.5050282, 1.5377381
7: -0.6710814, 0.6871932, -1.0412529, 1.0290250, -1.7001064, 1.7284460
8: -0.7382430, 1.5527720, -1.3481930, 1.5677047, -2.3059475, 2.9009643
9: -0.5604296, 0.7814227, -0.8360571, 1.0831776, -1.6436071, 1.6174798

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=74, inp2_unstable=126, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5724725, upper bound: 3.6922343
time: 1.38 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5722074, upper bound: 3.6207231
time: 1.47 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.8440781, 0.7487532, -1.1790082, 0.9477135, -1.7917917, 1.9277614
1: -0.6637570, 0.6703729, -0.9194198, 0.8935461, -1.5573032, 1.5897927
2: -0.5550326, 1.3367251, -0.9493456, 1.5668555, -2.1218882, 2.2860708
3: -0.6107832, 0.6956552, -1.0061703, 0.8660859, -1.4768691, 1.7018255
4: -0.7491193, 0.6973783, -1.1457578, 0.9516307, -1.7007501, 1.8431361
5: -0.6596397, 0.7621418, -0.9535775, 0.9979627, -1.6576023, 1.7157193
6: -0.5867524, 0.8152532, -0.8753743, 1.0821372, -1.6688896, 1.6906276
7: -0.7816032, 0.7968085, -1.1143459, 1.0947887, -1.8763919, 1.9111545
8: -0.9458506, 1.5565381, -1.4587713, 1.5714149, -2.5172648, 3.0153093
9: -0.6460938, 0.8747255, -0.8899143, 1.1417463, -1.7878401, 1.7646397

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=129, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7037229, upper bound: 3.7037296
time: 1.37 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7037229, upper bound: 3.7038223
time: 1.41 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.7498992, 0.6906404, -1.1160506, 0.9094023, -1.6593015, 1.8066911
1: -0.5940168, 0.6083853, -0.8690294, 0.8513769, -1.4453938, 1.4774146
2: -0.4509220, 1.2591189, -0.8759344, 1.5357180, -1.9866400, 2.1350532
3: -0.5034613, 0.6497082, -0.9298328, 0.8346396, -1.3381009, 1.5795410
4: -0.6432903, 0.6250859, -1.0677291, 0.9038042, -1.5470945, 1.6928151
5: -0.5921451, 0.6899128, -0.8969173, 0.9545611, -1.5467062, 1.5868301
6: -0.5016267, 0.7464218, -0.8221802, 1.0302030, -1.5318296, 1.5686020
7: -0.6961854, 0.7124593, -1.0502553, 1.0370147, -1.7332001, 1.7627146
8: -0.7909718, 1.5304153, -1.3734913, 1.5793042, -2.3702760, 2.9039066
9: -0.5786676, 0.8021612, -0.8434741, 1.0922356, -1.6709032, 1.6456354

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=126, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5701044, upper bound: 3.6619132
time: 1.68 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5693393, upper bound: 3.5753002
time: 1.46 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.8849511, 0.7712959, -1.1934845, 0.9576916, -1.8426425, 1.9647804
1: -0.6948996, 0.6947341, -0.9307069, 0.9027662, -1.5976658, 1.6254410
2: -0.6027899, 1.3758440, -0.9715015, 1.5880483, -2.1908383, 2.3473454
3: -0.6593508, 0.7152038, -1.0202091, 0.8736806, -1.5330315, 1.7354128
4: -0.7941504, 0.7288600, -1.1604913, 0.9625165, -1.7566669, 1.8893514
5: -0.6923543, 0.7898130, -0.9652208, 1.0083866, -1.7007408, 1.7550337
6: -0.6204342, 0.8444942, -0.8874640, 1.0943329, -1.7147672, 1.7319583
7: -0.8214942, 0.8329949, -1.1280476, 1.1069928, -1.9284871, 1.9610424
8: -1.0156063, 1.5348587, -1.4906927, 1.5830584, -2.5986648, 3.0255511
9: -0.6755854, 0.9032372, -0.9008090, 1.1545565, -1.8301420, 1.8040463

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=94, inp2_unstable=129, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7026034, upper bound: 3.6816254
time: 1.49 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7026034, upper bound: 3.7004295
time: 1.58 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.7498992, 0.6906404, -1.0710930, 0.8825604, -1.6324596, 1.7617334
1: -0.5940168, 0.6083853, -0.8329939, 0.8232377, -1.4172546, 1.4413792
2: -0.4509220, 1.2591189, -0.8202010, 1.4984818, -1.9494038, 2.0793200
3: -0.5034613, 0.6497082, -0.8778994, 0.8121650, -1.3156264, 1.5276077
4: -0.6432903, 0.6250859, -1.0147362, 0.8695178, -1.5128081, 1.6398221
5: -0.5921451, 0.6899128, -0.8567881, 0.9246824, -1.5168275, 1.5467010
6: -0.5016267, 0.7464218, -0.7874100, 0.9930605, -1.4946871, 1.5338318
7: -0.6961854, 0.7124593, -1.0039227, 0.9969144, -1.6930997, 1.7163820
8: -0.7909718, 1.5304153, -1.3031493, 1.6054368, -2.3964086, 2.8335648
9: -0.5786676, 0.8021612, -0.8107481, 1.0580591, -1.6367267, 1.6129093

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=118, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5703824, upper bound: 3.6671078
time: 1.61 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5696654, upper bound: 3.5778689
time: 1.40 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.8849511, 0.7712959, -1.1462979, 0.9297220, -1.8146729, 1.9175938
1: -0.6948996, 0.6947341, -0.8925265, 0.8735266, -1.5684261, 1.5872606
2: -0.6027899, 1.3758440, -0.9128386, 1.5498278, -2.1526177, 2.2886825
3: -0.6593508, 0.7152038, -0.9655280, 0.8501917, -1.5095425, 1.6807318
4: -0.7941504, 0.7288600, -1.1045774, 0.9267977, -1.7209482, 1.8334374
5: -0.6923543, 0.7898130, -0.9233721, 0.9775290, -1.6698833, 1.7131851
6: -0.6204342, 0.8444942, -0.8512925, 1.0551265, -1.6755607, 1.6957867
7: -0.8214942, 0.8329949, -1.0800151, 1.0644822, -1.8859764, 1.9130099
8: -1.0156063, 1.5348587, -1.4183735, 1.6091639, -2.6247702, 2.9532311
9: -0.6755854, 0.9032372, -0.8666105, 1.1191626, -1.7947481, 1.7698478

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=94, inp2_unstable=127, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6191821, upper bound: 3.6726912
time: 3.70 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6186636, upper bound: 3.5790925
time: 1.33 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.7222407, 0.6742139, -1.1160506, 0.9094023, -1.6316431, 1.7902646
1: -0.5717611, 0.5904015, -0.8690294, 0.8513769, -1.4231380, 1.4594309
2: -0.4184340, 1.2256221, -0.8759344, 1.5357180, -1.9541520, 2.1015563
3: -0.4736900, 0.6343694, -0.9298328, 0.8346396, -1.3083296, 1.5642023
4: -0.6134740, 0.6030147, -1.0677291, 0.9038042, -1.5172782, 1.6707438
5: -0.5731425, 0.6689765, -0.8969173, 0.9545611, -1.5277035, 1.5658939
6: -0.4829884, 0.7237632, -0.8221802, 1.0302030, -1.5131915, 1.5459433
7: -0.6710814, 0.6871932, -1.0502553, 1.0370147, -1.7080960, 1.7374485
8: -0.7382430, 1.5527720, -1.3734913, 1.5793042, -2.3175473, 2.9262629
9: -0.5604296, 0.7814227, -0.8434741, 1.0922356, -1.6526651, 1.6248968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=74, inp2_unstable=126, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5704778, upper bound: 3.6622849
time: 1.40 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5695649, upper bound: 3.5754114
time: 1.34 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.8440781, 0.7487532, -1.1934845, 0.9576916, -1.8017697, 1.9422377
1: -0.6637570, 0.6703729, -0.9307069, 0.9027662, -1.5665233, 1.6010797
2: -0.5550326, 1.3367251, -0.9715015, 1.5880483, -2.1430809, 2.3082266
3: -0.6107832, 0.6956552, -1.0202091, 0.8736806, -1.4844637, 1.7158643
4: -0.7491193, 0.6973783, -1.1604913, 0.9625165, -1.7116358, 1.8578696
5: -0.6596397, 0.7621418, -0.9652208, 1.0083866, -1.6680262, 1.7273626
6: -0.5867524, 0.8152532, -0.8874640, 1.0943329, -1.6810853, 1.7027173
7: -0.7816032, 0.7968085, -1.1280476, 1.1069928, -1.8885961, 1.9248562
8: -0.9458506, 1.5565381, -1.4906927, 1.5830584, -2.5289087, 3.0472305
9: -0.6460938, 0.8747255, -0.9008090, 1.1545565, -1.8006504, 1.7755345

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=129, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7026290, upper bound: 3.6818272
time: 1.47 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7026290, upper bound: 3.7006293
time: 1.67 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.7222407, 0.6742139, -1.0710930, 0.8825604, -1.6048012, 1.7453070
1: -0.5717611, 0.5904015, -0.8329939, 0.8232377, -1.3949988, 1.4233954
2: -0.4184340, 1.2256221, -0.8202010, 1.4984818, -1.9169158, 2.0458231
3: -0.4736900, 0.6343694, -0.8778994, 0.8121650, -1.2858551, 1.5122688
4: -0.6134740, 0.6030147, -1.0147362, 0.8695178, -1.4829917, 1.6177509
5: -0.5731425, 0.6689765, -0.8567881, 0.9246824, -1.4978249, 1.5257646
6: -0.4829884, 0.7237632, -0.7874100, 0.9930605, -1.4760489, 1.5111732
7: -0.6710814, 0.6871932, -1.0039227, 0.9969144, -1.6679957, 1.6911159
8: -0.7382430, 1.5527720, -1.3031493, 1.6054368, -2.3436799, 2.8559208
9: -0.5604296, 0.7814227, -0.8107481, 1.0580591, -1.6184888, 1.5921707

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=74, inp2_unstable=118, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5706183, upper bound: 3.6687338
time: 1.42 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5697800, upper bound: 3.5780980
time: 1.52 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.8440781, 0.7487532, -1.1462979, 0.9297220, -1.7738001, 1.8950511
1: -0.6637570, 0.6703729, -0.8925265, 0.8735266, -1.5372837, 1.5628994
2: -0.5550326, 1.3367251, -0.9128386, 1.5498278, -2.1048603, 2.2495637
3: -0.6107832, 0.6956552, -0.9655280, 0.8501917, -1.4609749, 1.6611832
4: -0.7491193, 0.6973783, -1.1045774, 0.9267977, -1.6759171, 1.8019557
5: -0.6596397, 0.7621418, -0.9233721, 0.9775290, -1.6371686, 1.6855140
6: -0.5867524, 0.8152532, -0.8512925, 1.0551265, -1.6418790, 1.6665456
7: -0.7816032, 0.7968085, -1.0800151, 1.0644822, -1.8460853, 1.8768237
8: -0.9458506, 1.5565381, -1.4183735, 1.6091639, -2.5550146, 2.9749110
9: -0.6460938, 0.8747255, -0.8666105, 1.1191626, -1.7652564, 1.7413360

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=127, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6202904, upper bound: 3.6746899
time: 1.55 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6197359, upper bound: 3.5793037
time: 1.41 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.7094821, 0.6650710, -1.1481541, 0.9273930, -1.6368752, 1.8132250
1: -0.5609788, 0.5800449, -0.8954868, 0.8710797, -1.4320585, 1.4755318
2: -0.4091291, 1.2244315, -0.9128944, 1.5537306, -1.9628596, 2.1373258
3: -0.4555055, 0.6259102, -0.9692515, 0.8503561, -1.3058616, 1.5951617
4: -0.5963171, 0.5904274, -1.1076493, 0.9283968, -1.5247140, 1.6980767
5: -0.5641767, 0.6563267, -0.9264235, 0.9749236, -1.5391003, 1.5827503
6: -0.4730721, 0.7126814, -0.8462256, 1.0566555, -1.5297276, 1.5589070
7: -0.6569301, 0.6723369, -1.0840650, 1.0669485, -1.7238786, 1.7564018
8: -0.7264873, 1.5624757, -1.4139940, 1.5455306, -2.2720177, 2.9764693
9: -0.5516040, 0.7706635, -0.8664832, 1.1146988, -1.6663028, 1.6371467

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=65, inp2_unstable=127, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5193911, upper bound: 3.6695118
time: 1.61 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5190783, upper bound: 3.6114792
time: 1.62 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.8742264, 0.7670830, -1.2215023, 0.9734242, -1.8476505, 1.9885854
1: -0.6864972, 0.6892030, -0.9535491, 0.9201962, -1.6066933, 1.6427522
2: -0.5923334, 1.3691396, -1.0033677, 1.6031204, -2.1954539, 2.3725071
3: -0.6446398, 0.7102579, -1.0558460, 0.8875699, -1.5322096, 1.7661039
4: -0.7812951, 0.7202976, -1.1962926, 0.9837828, -1.7650778, 1.9165902
5: -0.6831884, 0.7839862, -0.9916061, 1.0255891, -1.7087775, 1.7755923
6: -0.6132211, 0.8373356, -0.9078528, 1.1178652, -1.7310863, 1.7451885
7: -0.8100977, 0.8226141, -1.1573603, 1.1332127, -1.9433104, 1.9799745
8: -1.0035493, 1.5676322, -1.5236320, 1.5489385, -2.5524879, 3.0912640
9: -0.6684821, 0.8983816, -0.9211030, 1.1734855, -1.8419676, 1.8194846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=92, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7000314, upper bound: 3.7024927
time: 1.54 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7000314, upper bound: 3.7029500
time: 4.36 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.7094821, 0.6650710, -1.1098049, 0.9055201, -1.6150022, 1.7748759
1: -0.5609788, 0.5800449, -0.8640885, 0.8471579, -1.4081366, 1.4441334
2: -0.4091291, 1.2244315, -0.8684145, 1.5319718, -1.9411008, 2.0928459
3: -0.4555055, 0.6259102, -0.9225000, 0.8314926, -1.2869980, 1.5484102
4: -0.5963171, 0.5904274, -1.0602330, 0.8990732, -1.4953904, 1.6506604
5: -0.5641767, 0.6563267, -0.8914068, 0.9501853, -1.5143620, 1.5477335
6: -0.4730721, 0.7126814, -0.8168061, 1.0250499, -1.4981220, 1.5294874
7: -0.6569301, 0.6723369, -1.0439904, 1.0313488, -1.6882789, 1.7163272
8: -0.7264873, 1.5624757, -1.3639810, 1.5781779, -2.3046651, 2.9264567
9: -0.5516040, 0.7706635, -0.8388622, 1.0871391, -1.6387432, 1.6095257

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=65, inp2_unstable=126, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5190690, upper bound: 3.6572999
time: 1.31 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5187074, upper bound: 3.5743171
time: 1.21 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.8742264, 0.7670830, -1.1871551, 0.9537766, -1.8280029, 1.9542382
1: -0.6864972, 0.6892030, -0.9257241, 0.8984902, -1.5849874, 1.6149272
2: -0.5923334, 1.3691396, -0.9638585, 1.5842650, -2.1765985, 2.3329980
3: -0.6446398, 0.7102579, -1.0126770, 0.8704512, -1.5150909, 1.7229348
4: -0.7812951, 0.7202976, -1.1527280, 0.9577714, -1.7390665, 1.8730257
5: -0.6831884, 0.7839862, -0.9595886, 1.0039959, -1.6871843, 1.7435749
6: -0.6132211, 0.8373356, -0.8820717, 1.0890374, -1.7022586, 1.7194073
7: -0.8100977, 0.8226141, -1.1217581, 1.1013004, -1.9113981, 1.9443722
8: -1.0035493, 1.5676322, -1.4811000, 1.5818691, -2.5854185, 3.0487320
9: -0.6684821, 0.8983816, -0.8961564, 1.1494105, -1.8178926, 1.7945380

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=92, inp2_unstable=129, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6996222, upper bound: 3.6813741
time: 1.38 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6996222, upper bound: 3.7001756
time: 1.42 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.8530254, 0.7574339, -1.2215023, 0.9734242, -1.8264496, 1.9789362
1: -0.6702126, 0.6770992, -0.9535491, 0.9201962, -1.5904088, 1.6306484
2: -0.5743517, 1.3624271, -1.0033677, 1.6031204, -2.1774721, 2.3657947
3: -0.6159139, 0.7004040, -1.0558460, 0.8875699, -1.5034838, 1.7562500
4: -0.7554982, 0.7035162, -1.1962926, 0.9837828, -1.7392809, 1.8998089
5: -0.6654902, 0.7711605, -0.9916061, 1.0255891, -1.6910793, 1.7627666
6: -0.5970553, 0.8228754, -0.9078528, 1.1178652, -1.7149205, 1.7307281
7: -0.7886799, 0.8024640, -1.1573603, 1.1332127, -1.9218926, 1.9598243
8: -0.9806290, 1.5964582, -1.5236320, 1.5489385, -2.5295675, 3.1200900
9: -0.6540712, 0.8859549, -0.9211030, 1.1734855, -1.8275567, 1.8070579

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=90, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6671078, upper bound: 3.5703824
time: 1.81 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6726912, upper bound: 3.6191821
time: 1.93 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.6440119, 0.6118391, -1.0868341, 0.8896823, -1.5336941, 1.6986731
1: -0.5041932, 0.5267908, -0.8465621, 0.8301913, -1.3343844, 1.3733529
2: -0.3633332, 1.1695341, -0.8411919, 1.5211673, -1.8845005, 2.0107260
3: -0.3819892, 0.5737923, -0.8960766, 0.8195898, -1.2015791, 1.4698689
4: -0.5193256, 0.5245317, -1.0328540, 0.8817410, -1.4010665, 1.5573857
5: -0.5214867, 0.5886931, -0.8717379, 0.9324290, -1.4539156, 1.4604311
6: -0.4217668, 0.6589385, -0.7944601, 1.0062410, -1.4280078, 1.4533986
7: -0.5816638, 0.5936893, -1.0217854, 1.0110166, -1.5926805, 1.6154747
8: -0.6266382, 1.6057832, -1.3286734, 1.5444535, -2.1710916, 2.9329629
9: -0.4973323, 0.7214990, -0.8213224, 1.0656481, -1.5629804, 1.5428214

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=124, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5778689, upper bound: 3.5696654
time: 1.46 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5790925, upper bound: 3.6186636
time: 1.20 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.8530254, 0.7574339, -1.1871551, 0.9537766, -1.8068020, 1.9445890
1: -0.6702126, 0.6770992, -0.9257241, 0.8984902, -1.5687027, 1.6028233
2: -0.5743517, 1.3624271, -0.9638585, 1.5842650, -2.1586165, 2.3262856
3: -0.6159139, 0.7004040, -1.0126770, 0.8704512, -1.4863651, 1.7130810
4: -0.7554982, 0.7035162, -1.1527280, 0.9577714, -1.7132696, 1.8562442
5: -0.6654902, 0.7711605, -0.9595886, 1.0039959, -1.6694860, 1.7307491
6: -0.5970553, 0.8228754, -0.8820717, 1.0890374, -1.6860926, 1.7049471
7: -0.7886799, 0.8024640, -1.1217581, 1.1013004, -1.8899803, 1.9242221
8: -0.9806290, 1.5964582, -1.4811000, 1.5818691, -2.5624981, 3.0775580
9: -0.6540712, 0.8859549, -0.8961564, 1.1494105, -1.8034817, 1.7821113

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=90, inp2_unstable=129, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6622493, upper bound: 3.5192573
time: 1.45 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6708247, upper bound: 3.5766225
time: 1.67 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.6440119, 0.6118391, -1.0538086, 0.8706053, -1.5146172, 1.6656476
1: -0.5041932, 0.5267908, -0.8199511, 0.8088338, -1.3130270, 1.3467419
2: -0.3633332, 1.1695341, -0.8025618, 1.5013624, -1.8646957, 1.9720958
3: -0.3819892, 0.5737923, -0.8548449, 0.8029428, -1.1849320, 1.4286373
4: -0.5193256, 0.5245317, -0.9908340, 0.8559622, -1.3752878, 1.5153657
5: -0.5214867, 0.5886931, -0.8402588, 0.9101624, -1.4316492, 1.4289520
6: -0.4217668, 0.6589385, -0.7687543, 0.9783113, -1.4000782, 1.4276928
7: -0.5816638, 0.5936893, -0.9863403, 0.9807572, -1.5624211, 1.5800296
8: -0.6266382, 1.6057832, -1.2849200, 1.5769031, -2.2035413, 2.8890388
9: -0.4973323, 0.7214990, -0.7973835, 1.0410837, -1.5384160, 1.5188825

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=35, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=115, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5768629, upper bound: 3.5188174
time: 1.61 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5786944, upper bound: 3.5762089
time: 1.41 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.7094821, 0.6650710, -1.1063476, 0.9023004, -1.6117826, 1.7714186
1: -0.5609788, 0.5800449, -0.8614897, 0.8453042, -1.4062829, 1.4415345
2: -0.4091291, 1.2244315, -0.8593577, 1.5175031, -1.9266322, 2.0837891
3: -0.4555055, 0.6259102, -0.9215403, 0.8294817, -1.2849872, 1.5474504
4: -0.5963171, 0.5904274, -1.0589498, 0.8964415, -1.4927585, 1.6493772
5: -0.5641767, 0.6563267, -0.8895599, 0.9473463, -1.5115230, 1.5458865
6: -0.4730721, 0.7126814, -0.8139750, 1.0220398, -1.4951118, 1.5266564
7: -0.6569301, 0.6723369, -1.0412529, 1.0290250, -1.6859550, 1.7135897
8: -0.7264873, 1.5624757, -1.3481930, 1.5677047, -2.2941918, 2.9106679
9: -0.5516040, 0.7706635, -0.8360571, 1.0831776, -1.6347816, 1.6067207

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=65, inp2_unstable=126, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5195520, upper bound: 3.6721805
time: 1.80 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5192074, upper bound: 3.6117902
time: 1.50 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.8742264, 0.7670830, -1.1790082, 0.9477135, -1.8219399, 1.9460913
1: -0.6864972, 0.6892030, -0.9194198, 0.8935461, -1.5800433, 1.6086228
2: -0.5923334, 1.3691396, -0.9493456, 1.5668555, -2.1591890, 2.3184853
3: -0.6446398, 0.7102579, -1.0061703, 0.8660859, -1.5107257, 1.7164282
4: -0.7812951, 0.7202976, -1.1457578, 0.9516307, -1.7329258, 1.8660555
5: -0.6831884, 0.7839862, -0.9535775, 0.9979627, -1.6811512, 1.7375636
6: -0.6132211, 0.8373356, -0.8753743, 1.0821372, -1.6953583, 1.7127099
7: -0.8100977, 0.8226141, -1.1143459, 1.0947887, -1.9048864, 1.9369600
8: -1.0035493, 1.5676322, -1.4587713, 1.5714149, -2.5749643, 3.0264030
9: -0.6684821, 0.8983816, -0.8899143, 1.1417463, -1.8102283, 1.7882959

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=92, inp2_unstable=129, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7002222, upper bound: 3.7026123
time: 1.50 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.7002222, upper bound: 3.7029751
time: 1.64 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.7094821, 0.6650710, -1.0704333, 0.8820498, -1.5915320, 1.7355042
1: -0.5609788, 0.5800449, -0.8325109, 0.8227046, -1.3836834, 1.4125559
2: -0.4091291, 1.2244315, -0.8192167, 1.4978338, -1.9069629, 2.0436482
3: -0.4555055, 0.6259102, -0.8772107, 0.8118067, -1.2673123, 1.5031209
4: -0.5963171, 0.5904274, -1.0140095, 0.8690170, -1.4653342, 1.6044369
5: -0.5641767, 0.6563267, -0.8562449, 0.9241116, -1.4882883, 1.5125716
6: -0.4730721, 0.7126814, -0.7866906, 0.9924997, -1.4655718, 1.4993720
7: -0.6569301, 0.6723369, -1.0033158, 0.9963481, -1.6532784, 1.6756527
8: -0.7264873, 1.5624757, -1.3017231, 1.6038575, -2.3303447, 2.8641982
9: -0.5516040, 0.7706635, -0.8102128, 1.0573615, -1.6089655, 1.5808764

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=65, inp2_unstable=118, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5192573, upper bound: 3.6620318
time: 1.61 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5188174, upper bound: 3.5768629
time: 1.21 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.8742264, 0.7670830, -1.1456043, 0.9291941, -1.8034205, 1.9126873
1: -0.6864972, 0.6892030, -0.8920125, 0.8729835, -1.5594807, 1.5812155
2: -0.5923334, 1.3691396, -0.9118081, 1.5491328, -2.1414661, 2.2809477
3: -0.6446398, 0.7102579, -0.9648210, 0.8498217, -1.4944615, 1.6750789
4: -0.7812951, 0.7202976, -1.1038303, 0.9262789, -1.7075740, 1.8241279
5: -0.6831884, 0.7839862, -0.9228160, 0.9769496, -1.6601380, 1.7068021
6: -0.6132211, 0.8373356, -0.8505538, 1.0545474, -1.6677685, 1.6878895
7: -0.8100977, 0.8226141, -1.0793865, 1.0638889, -1.8739866, 1.9020007
8: -1.0035493, 1.5676322, -1.4168493, 1.6075697, -2.6111190, 2.9844813
9: -0.6684821, 0.8983816, -0.8660536, 1.1184500, -1.7869322, 1.7644353

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=92, inp2_unstable=127, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5766225, upper bound: 3.6708059
time: 1.34 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5762089, upper bound: 3.5786944
time: 1.25 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.8530254, 0.7574339, -1.1790082, 0.9477135, -1.8007389, 1.9364421
1: -0.6702126, 0.6770992, -0.9194198, 0.8935461, -1.5637586, 1.5965190
2: -0.5743517, 1.3624271, -0.9493456, 1.5668555, -2.1412072, 2.3117728
3: -0.6159139, 0.7004040, -1.0061703, 0.8660859, -1.4819998, 1.7065743
4: -0.7554982, 0.7035162, -1.1457578, 0.9516307, -1.7071289, 1.8492740
5: -0.6654902, 0.7711605, -0.9535775, 0.9979627, -1.6634529, 1.7247381
6: -0.5970553, 0.8228754, -0.8753743, 1.0821372, -1.6791925, 1.6982497
7: -0.7886799, 0.8024640, -1.1143459, 1.0947887, -1.8834686, 1.9168099
8: -0.9806290, 1.5964582, -1.4587713, 1.5714149, -2.5520439, 3.0552297
9: -0.6540712, 0.8859549, -0.8899143, 1.1417463, -1.7958175, 1.7758691

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=90, inp2_unstable=129, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6687403, upper bound: 3.5708594
time: 5.99 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6745903, upper bound: 3.6204705
time: 2.26 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.6440119, 0.6118391, -1.0522465, 0.8687720, -1.5127839, 1.6640855
1: -0.5041932, 0.5267908, -0.8188677, 0.8084453, -1.3126385, 1.3456585
2: -0.3633332, 1.1695341, -0.7965305, 1.4878114, -1.8511447, 1.9660646
3: -0.3819892, 0.5737923, -0.8563170, 0.8019770, -1.1839662, 1.4301093
4: -0.5193256, 0.5245317, -0.9920189, 0.8551142, -1.3744397, 1.5165507
5: -0.5214867, 0.5886931, -0.8403701, 0.9088497, -1.4303365, 1.4290632
6: -0.4217668, 0.6589385, -0.7678466, 0.9770172, -1.3987840, 1.4267850
7: -0.5816638, 0.5936893, -0.9856521, 0.9805603, -1.5622241, 1.5793414
8: -0.6266382, 1.6057832, -1.2716765, 1.5665590, -2.1931973, 2.8759563
9: -0.4973323, 0.7214990, -0.7960597, 1.0388745, -1.5362068, 1.5175588

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=115, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5780533, upper bound: 3.5700727
time: 1.32 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5792315, upper bound: 3.6199407
time: 1.36 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.8530254, 0.7574339, -1.1456043, 0.9291941, -1.7822195, 1.9030381
1: -0.6702126, 0.6770992, -0.8920125, 0.8729835, -1.5431960, 1.5691116
2: -0.5743517, 1.3624271, -0.9118081, 1.5491328, -2.1234846, 2.2742352
3: -0.6159139, 0.7004040, -0.9648210, 0.8498217, -1.4657356, 1.6652250
4: -0.7554982, 0.7035162, -1.1038303, 0.9262789, -1.6817771, 1.8073466
5: -0.6654902, 0.7711605, -0.9228160, 0.9769496, -1.6424397, 1.6939766
6: -0.5970553, 0.8228754, -0.8505538, 1.0545474, -1.6516027, 1.6734293
7: -0.7886799, 0.8024640, -1.0793865, 1.0638889, -1.8525689, 1.8818505
8: -0.9806290, 1.5964582, -1.4168493, 1.6075697, -2.5881984, 3.0133076
9: -0.6540712, 0.8859549, -0.8660536, 1.1184500, -1.7725213, 1.7520084

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=90, inp2_unstable=127, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6223139, upper bound: 3.6223712
time: 1.68 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6223139, upper bound: 3.6223712
time: 1.20 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.6440119, 0.6118391, -1.0143991, 0.8488254, -1.4928372, 1.6262381
1: -0.5041932, 0.5267908, -0.7896934, 0.7844815, -1.2886747, 1.3164842
2: -0.3633332, 1.1695341, -0.7563757, 1.4665864, -1.8299196, 1.9259098
3: -0.3819892, 0.5737923, -0.8096391, 0.7829872, -1.1649765, 1.3834314
4: -0.5193256, 0.5245317, -0.9441562, 0.8270260, -1.3463516, 1.4686879
5: -0.5214867, 0.5886931, -0.8049737, 0.8833268, -1.4048135, 1.3936667
6: -0.4217668, 0.6589385, -0.7391852, 0.9451736, -1.3669405, 1.3981236
7: -0.5816638, 0.5936893, -0.9463702, 0.9472179, -1.5288818, 1.5400596
8: -0.6266382, 1.6057832, -1.2218800, 1.6024715, -2.2291098, 2.8259234
9: -0.4973323, 0.7214990, -0.7698816, 1.0107820, -1.5081143, 1.4913807

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=110, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5771008, upper bound: 3.5230059
time: 1.53 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5788344, upper bound: 3.5788898
time: 1.14 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.21 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.5718975, upper bound: 3.6856204
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.5716330, upper bound: 3.6192978
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.7034503, upper bound: 3.7033540
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.7034503, upper bound: 3.7036106
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.5720452, upper bound: 3.6905329
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.5717673, upper bound: 3.6205211
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.7036294, upper bound: 3.7034323
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.7036294, upper bound: 3.7036260
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.5724056, upper bound: 3.6858634
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.5721306, upper bound: 3.6194399
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.7034850, upper bound: 3.7035749
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.7034850, upper bound: 3.7037266
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.5724725, upper bound: 3.6922343
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.5722074, upper bound: 3.6207231
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.7037229, upper bound: 3.7037296
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.7037229, upper bound: 3.7038223
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.5701044, upper bound: 3.6619132
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.5693393, upper bound: 3.5753002
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.7026034, upper bound: 3.6816254
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.7026034, upper bound: 3.7004295
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.5703824, upper bound: 3.6671078
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.5696654, upper bound: 3.5778689
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.6191821, upper bound: 3.6726912
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.6186636, upper bound: 3.5790925
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.5704778, upper bound: 3.6622849
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.5695649, upper bound: 3.5754114
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.7026290, upper bound: 3.6818272
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.7026290, upper bound: 3.7006293
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.5706183, upper bound: 3.6687338
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.5697800, upper bound: 3.5780980
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.6202904, upper bound: 3.6746899
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.6197359, upper bound: 3.5793037
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.5193911, upper bound: 3.6695118
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.5190783, upper bound: 3.6114792
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.7000314, upper bound: 3.7024927
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.7000314, upper bound: 3.7029500
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.5190690, upper bound: 3.6572999
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.5187074, upper bound: 3.5743171
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.6996222, upper bound: 3.6813741
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.6996222, upper bound: 3.7001756
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.6671078, upper bound: 3.5703824
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.6726912, upper bound: 3.6191821
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.5778689, upper bound: 3.5696654
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.5790925, upper bound: 3.6186636
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.6622493, upper bound: 3.5192573
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.6708247, upper bound: 3.5766225
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.5768629, upper bound: 3.5188174
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.5786944, upper bound: 3.5762089
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.5195520, upper bound: 3.6721805
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.5192074, upper bound: 3.6117902
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.7002222, upper bound: 3.7026123
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.7002222, upper bound: 3.7029751
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.5192573, upper bound: 3.6620318
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.5188174, upper bound: 3.5768629
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.5766225, upper bound: 3.6708059
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.5762089, upper bound: 3.5786944
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.6687403, upper bound: 3.5708594
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.6745903, upper bound: 3.6204705
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.5780533, upper bound: 3.5700727
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.5792315, upper bound: 3.6199407
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.6223139, upper bound: 3.6223712
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.6223139, upper bound: 3.6223712
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.5771008, upper bound: 3.5230059
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 8, lower bound: -3.5788344, upper bound: 3.5788898

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.7498992, 0.6906404, -0.8582047, 0.7562012, -1.5061004, 1.5488452
1: -0.5940168, 0.6083853, -0.6748229, 0.6770743, -1.2710912, 1.2832081
2: -0.4509220, 1.2591189, -0.5773535, 1.3683352, -1.8192573, 1.8364724
3: -0.5034613, 0.6497082, -0.6231498, 0.7028407, -1.2063019, 1.2728579
4: -0.6432903, 0.6250859, -0.7609344, 0.7073906, -1.3506808, 1.3860203
5: -0.5921451, 0.6899128, -0.6697260, 0.7704628, -1.3626078, 1.3596387
6: -0.5016267, 0.7464218, -0.5949098, 0.8261688, -1.3277955, 1.3413316
7: -0.6961854, 0.7124593, -0.7950886, 0.8075733, -1.5037587, 1.5075479
8: -0.7909718, 1.5304153, -0.9844482, 1.5345674, -2.3255391, 2.5148635
9: -0.5786676, 0.8021612, -0.6556405, 0.8841013, -1.4627689, 1.4578017

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=90, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 137

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5716330, upper bound: 3.6192978
time: 1.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5716330, upper bound: 3.6192978
time: 1.21 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.6681275, 0.6344774, -0.6436942, 0.6097843, -1.2779118, 1.2781715
1: -0.5301463, 0.5486343, -0.5053918, 0.5260400, -1.0561863, 1.0540261
2: -0.3732882, 1.1852838, -0.3621349, 1.1757877, -1.5490760, 1.5474187
3: -0.4119392, 0.5995863, -0.3836681, 0.5753074, -0.9872466, 0.9832544
4: -0.5528595, 0.5541098, -0.5198111, 0.5253024, -1.0781618, 1.0739210
5: -0.5389526, 0.6179965, -0.5210704, 0.5869219, -1.1258745, 1.1390668
6: -0.4387485, 0.6799842, -0.4175265, 0.6594937, -1.0982423, 1.0975107
7: -0.6147015, 0.6344651, -0.5834085, 0.5967019, -1.2114034, 1.2178736
8: -0.6552932, 1.5269337, -0.6295719, 1.5443664, -2.1980217, 2.1565051
9: -0.5192574, 0.7358742, -0.4955094, 0.7172167, -1.2364740, 1.2313836

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=51, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.0586180, upper bound: 3.3856956
time: 1.50 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5467674, upper bound: 3.5946240
time: 1.47 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.8849511, 0.7712959, -0.7498992, 0.6906404, -1.5755916, 1.5211952
1: -0.6948996, 0.6947341, -0.5940168, 0.6083853, -1.3032849, 1.2887509
2: -0.6027899, 1.3758440, -0.4509220, 1.2591189, -1.8619089, 1.8267660
3: -0.6593508, 0.7152038, -0.5034613, 0.6497082, -1.3090590, 1.2186651
4: -0.7941504, 0.7288600, -0.6432903, 0.6250859, -1.4192364, 1.3721503
5: -0.6923543, 0.7898130, -0.5921451, 0.6899128, -1.3822670, 1.3819580
6: -0.6204342, 0.8444942, -0.5016267, 0.7464218, -1.3668561, 1.3461208
7: -0.8214942, 0.8329949, -0.6961854, 0.7124593, -1.5339534, 1.5291803
8: -1.0156063, 1.5348587, -0.7909718, 1.5304153, -2.5460215, 2.3258302
9: -0.6755854, 0.9032372, -0.5786676, 0.8021612, -1.4777467, 1.4819049

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=94, inp2_unstable=78, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6856204, upper bound: 3.5718975
time: 1.50 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6192977, upper bound: 3.5716330
time: 1.27 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.8849511, 0.7712959, -0.8849511, 0.7712959, -1.6562469, 1.6562469
1: -0.6948996, 0.6947341, -0.6948996, 0.6947341, -1.3896337, 1.3896337
2: -0.6027899, 1.3758440, -0.6027899, 1.3758440, -1.9786339, 1.9786339
3: -0.6593508, 0.7152038, -0.6593508, 0.7152038, -1.3745546, 1.3745546
4: -0.7941504, 0.7288600, -0.7941504, 0.7288600, -1.5230105, 1.5230105
5: -0.6923543, 0.7898130, -0.6923543, 0.7898130, -1.4821672, 1.4821672
6: -0.6204342, 0.8444942, -0.6204342, 0.8444942, -1.4649284, 1.4649284
7: -0.8214942, 0.8329949, -0.8214942, 0.8329949, -1.6544890, 1.6544890
8: -1.0156063, 1.5348587, -1.0156063, 1.5348587, -2.5504646, 2.5504644
9: -0.6755854, 0.9032372, -0.6755854, 0.9032372, -1.5788226, 1.5788226

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=94, inp2_unstable=94, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6856204, upper bound: 3.6203546
time: 1.96 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6192978, upper bound: 3.6201390
time: 1.37 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.7498992, 0.6906404, -0.8105316, 0.7298173, -1.4797165, 1.5011721
1: -0.5940168, 0.6083853, -0.6392021, 0.6485250, -1.2425418, 1.2475874
2: -0.4509220, 1.2591189, -0.5240676, 1.3259603, -1.7768823, 1.7831864
3: -0.5034613, 0.6497082, -0.5673954, 0.6800897, -1.1835510, 1.2171036
4: -0.6432903, 0.6250859, -0.7083856, 0.6711948, -1.3144851, 1.3334715
5: -0.5921451, 0.6899128, -0.6335961, 0.7378805, -1.3300256, 1.3235090
6: -0.5016267, 0.7464218, -0.5562324, 0.7921594, -1.2937860, 1.3026543
7: -0.6961854, 0.7124593, -0.7487205, 0.7658302, -1.4620155, 1.4611797
8: -0.7909718, 1.5304153, -0.9062009, 1.5564458, -2.3474176, 2.4366162
9: -0.5786676, 0.8021612, -0.6216494, 0.8509103, -1.4295779, 1.4238106

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=85, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5717673, upper bound: 3.6205211
time: 1.43 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5717673, upper bound: 3.6205211
time: 1.76 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.6681275, 0.6344774, -0.6336532, 0.6037244, -1.2718518, 1.2681305
1: -0.5301463, 0.5486343, -0.4970237, 0.5189049, -1.0490513, 1.0456580
2: -0.3732882, 1.1852838, -0.3559743, 1.1531411, -1.5264294, 1.5412581
3: -0.4119392, 0.5995863, -0.3744665, 0.5683330, -0.9802722, 0.9740528
4: -0.5528595, 0.5541098, -0.5105125, 0.5169896, -1.0698491, 1.0646224
5: -0.5389526, 0.6179965, -0.5143556, 0.5789285, -1.1178811, 1.1323521
6: -0.4387485, 0.6799842, -0.4116655, 0.6512663, -1.0900148, 1.0916497
7: -0.6147015, 0.6344651, -0.5743891, 0.5862142, -1.2009158, 1.2088542
8: -0.6552932, 1.5269337, -0.6005872, 1.5663713, -2.2200046, 2.1275201
9: -0.5192574, 0.7358742, -0.4875829, 0.7117791, -1.2310364, 1.2234571

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=49, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.0586675, upper bound: 3.3856956
time: 1.37 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5468984, upper bound: 3.5964007
time: 1.37 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.8849511, 0.7712959, -0.7222407, 0.6742139, -1.5591650, 1.4935367
1: -0.6948996, 0.6947341, -0.5717611, 0.5904015, -1.2853011, 1.2664952
2: -0.6027899, 1.3758440, -0.4184340, 1.2256221, -1.8284121, 1.7942780
3: -0.6593508, 0.7152038, -0.4736900, 0.6343694, -1.2937202, 1.1888938
4: -0.7941504, 0.7288600, -0.6134740, 0.6030147, -1.3971651, 1.3423340
5: -0.6923543, 0.7898130, -0.5731425, 0.6689765, -1.3613307, 1.3629555
6: -0.6204342, 0.8444942, -0.4829884, 0.7237632, -1.3441974, 1.3274827
7: -0.8214942, 0.8329949, -0.6710814, 0.6871932, -1.5086874, 1.5040762
8: -1.0156063, 1.5348587, -0.7382430, 1.5527720, -2.5683777, 2.2731013
9: -0.6755854, 0.9032372, -0.5604296, 0.7814227, -1.4570081, 1.4636669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=94, inp2_unstable=74, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 137

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6858633, upper bound: 3.5724056
time: 1.76 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6194399, upper bound: 3.5721306
time: 1.48 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.8849511, 0.7712959, -0.8440781, 0.7487532, -1.6337042, 1.6153741
1: -0.6948996, 0.6947341, -0.6637570, 0.6703729, -1.3652725, 1.3584912
2: -0.6027899, 1.3758440, -0.5550326, 1.3367251, -1.9395151, 1.9308765
3: -0.6593508, 0.7152038, -0.6107832, 0.6956552, -1.3550060, 1.3259869
4: -0.7941504, 0.7288600, -0.7491193, 0.6973783, -1.4915287, 1.4779794
5: -0.6923543, 0.7898130, -0.6596397, 0.7621418, -1.4544961, 1.4494526
6: -0.6204342, 0.8444942, -0.5867524, 0.8152532, -1.4356874, 1.4312466
7: -0.8214942, 0.8329949, -0.7816032, 0.7968085, -1.6183028, 1.6145980
8: -1.0156063, 1.5348587, -0.9458506, 1.5565381, -2.5721445, 2.4807091
9: -0.6755854, 0.9032372, -0.6460938, 0.8747255, -1.5503109, 1.5493309

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=94, inp2_unstable=89, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6858634, upper bound: 3.6216135
time: 1.48 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6194399, upper bound: 3.6213871
time: 1.38 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.7222407, 0.6742139, -0.8582047, 0.7562012, -1.4784420, 1.5324187
1: -0.5717611, 0.5904015, -0.6748229, 0.6770743, -1.2488354, 1.2652243
2: -0.4184340, 1.2256221, -0.5773535, 1.3683352, -1.7867693, 1.8029755
3: -0.4736900, 0.6343694, -0.6231498, 0.7028407, -1.1765306, 1.2575192
4: -0.6134740, 0.6030147, -0.7609344, 0.7073906, -1.3208646, 1.3639491
5: -0.5731425, 0.6689765, -0.6697260, 0.7704628, -1.3436053, 1.3387024
6: -0.4829884, 0.7237632, -0.5949098, 0.8261688, -1.3091573, 1.3186729
7: -0.6710814, 0.6871932, -0.7950886, 0.8075733, -1.4786546, 1.4822817
8: -0.7382430, 1.5527720, -0.9844482, 1.5345674, -2.2728105, 2.5372200
9: -0.5604296, 0.7814227, -0.6556405, 0.8841013, -1.4445310, 1.4370632

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=74, inp2_unstable=90, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5721306, upper bound: 3.6194399
time: 1.22 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5721306, upper bound: 3.6194399
time: 1.18 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.6516190, 0.6240643, -0.6436942, 0.6097843, -1.2614033, 1.2677585
1: -0.5168293, 0.5368063, -0.5053918, 0.5260400, -1.0428693, 1.0421981
2: -0.3630740, 1.1548004, -0.3621349, 1.1757877, -1.5388616, 1.5169353
3: -0.3967934, 0.5880464, -0.3836681, 0.5753074, -0.9721009, 0.9717146
4: -0.5367474, 0.5402665, -0.5198111, 0.5253024, -1.0620497, 1.0600777
5: -0.5282872, 0.6039501, -0.5210704, 0.5869219, -1.1152091, 1.1250205
6: -0.4285384, 0.6665362, -0.4175265, 0.6594937, -1.0880321, 1.0840627
7: -0.5999638, 0.6171570, -0.5834085, 0.5967019, -1.1966658, 1.2005655
8: -0.6140552, 1.5491552, -0.6295719, 1.5443664, -2.1567540, 2.1774974
9: -0.5054486, 0.7261571, -0.4955094, 0.7172167, -1.2226653, 1.2216666

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=51, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.0584058, upper bound: 3.3820500
time: 1.24 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5478709, upper bound: 3.5947675
time: 2.42 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.8440781, 0.7487532, -0.7498992, 0.6906404, -1.5347185, 1.4986525
1: -0.6637570, 0.6703729, -0.5940168, 0.6083853, -1.2721423, 1.2643898
2: -0.5550326, 1.3367251, -0.4509220, 1.2591189, -1.8141515, 1.7876471
3: -0.6107832, 0.6956552, -0.5034613, 0.6497082, -1.2604914, 1.1991165
4: -0.7491193, 0.6973783, -0.6432903, 0.6250859, -1.3742054, 1.3406687
5: -0.6596397, 0.7621418, -0.5921451, 0.6899128, -1.3495524, 1.3542869
6: -0.5867524, 0.8152532, -0.5016267, 0.7464218, -1.3331742, 1.3168799
7: -0.7816032, 0.7968085, -0.6961854, 0.7124593, -1.4940624, 1.4929938
8: -0.9458506, 1.5565381, -0.7909718, 1.5304153, -2.4762659, 2.3475099
9: -0.6460938, 0.8747255, -0.5786676, 0.8021612, -1.4482551, 1.4533931

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=78, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6905329, upper bound: 3.5720452
time: 1.42 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6205211, upper bound: 3.5717673
time: 1.49 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.8440781, 0.7487532, -0.8849511, 0.7712959, -1.6153741, 1.6337042
1: -0.6637570, 0.6703729, -0.6948996, 0.6947341, -1.3584912, 1.3652725
2: -0.5550326, 1.3367251, -0.6027899, 1.3758440, -1.9308765, 1.9395151
3: -0.6107832, 0.6956552, -0.6593508, 0.7152038, -1.3259869, 1.3550060
4: -0.7491193, 0.6973783, -0.7941504, 0.7288600, -1.4779794, 1.4915287
5: -0.6596397, 0.7621418, -0.6923543, 0.7898130, -1.4494526, 1.4544961
6: -0.5867524, 0.8152532, -0.6204342, 0.8444942, -1.4312466, 1.4356874
7: -0.7816032, 0.7968085, -0.8214942, 0.8329949, -1.6145980, 1.6183028
8: -0.9458506, 1.5565381, -1.0156063, 1.5348587, -2.4807086, 2.5721443
9: -0.6460938, 0.8747255, -0.6755854, 0.9032372, -1.5493309, 1.5503109

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=94, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6905329, upper bound: 3.6205025
time: 1.47 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6205211, upper bound: 3.6202749
time: 1.54 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.7222407, 0.6742139, -0.8105316, 0.7298173, -1.4520581, 1.4847455
1: -0.5717611, 0.5904015, -0.6392021, 0.6485250, -1.2202861, 1.2296035
2: -0.4184340, 1.2256221, -0.5240676, 1.3259603, -1.7443943, 1.7496896
3: -0.4736900, 0.6343694, -0.5673954, 0.6800897, -1.1537796, 1.2017648
4: -0.6134740, 0.6030147, -0.7083856, 0.6711948, -1.2846687, 1.3114004
5: -0.5731425, 0.6689765, -0.6335961, 0.7378805, -1.3110230, 1.3025726
6: -0.4829884, 0.7237632, -0.5562324, 0.7921594, -1.2751479, 1.2799956
7: -0.6710814, 0.6871932, -0.7487205, 0.7658302, -1.4369116, 1.4359136
8: -0.7382430, 1.5527720, -0.9062009, 1.5564458, -2.2946887, 2.4589729
9: -0.5604296, 0.7814227, -0.6216494, 0.8509103, -1.4113400, 1.4030721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=74, inp2_unstable=85, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5722074, upper bound: 3.6207231
time: 1.30 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5722074, upper bound: 3.6207231
time: 1.29 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.6516190, 0.6240643, -0.6336532, 0.6037244, -1.2553433, 1.2577175
1: -0.5168293, 0.5368063, -0.4970237, 0.5189049, -1.0357342, 1.0338299
2: -0.3630740, 1.1548004, -0.3559743, 1.1531411, -1.5162151, 1.5107747
3: -0.3967934, 0.5880464, -0.3744665, 0.5683330, -0.9651264, 0.9625130
4: -0.5367474, 0.5402665, -0.5105125, 0.5169896, -1.0537370, 1.0507791
5: -0.5282872, 0.6039501, -0.5143556, 0.5789285, -1.1072158, 1.1183057
6: -0.4285384, 0.6665362, -0.4116655, 0.6512663, -1.0798047, 1.0782017
7: -0.5999638, 0.6171570, -0.5743891, 0.5862142, -1.1861781, 1.1915461
8: -0.6140552, 1.5491552, -0.6005872, 1.5663713, -2.1786790, 2.1484540
9: -0.5054486, 0.7261571, -0.4875829, 0.7117791, -1.2172277, 1.2137400

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=49, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.0584058, upper bound: 3.3820500
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5480089, upper bound: 3.5966521
time: 1.24 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.8440781, 0.7487532, -0.7222407, 0.6742139, -1.5182921, 1.4709940
1: -0.6637570, 0.6703729, -0.5717611, 0.5904015, -1.2541585, 1.2421341
2: -0.5550326, 1.3367251, -0.4184340, 1.2256221, -1.7806547, 1.7551591
3: -0.6107832, 0.6956552, -0.4736900, 0.6343694, -1.2451526, 1.1693451
4: -0.7491193, 0.6973783, -0.6134740, 0.6030147, -1.3521340, 1.3108523
5: -0.6596397, 0.7621418, -0.5731425, 0.6689765, -1.3286161, 1.3352842
6: -0.5867524, 0.8152532, -0.4829884, 0.7237632, -1.3105156, 1.2982416
7: -0.7816032, 0.7968085, -0.6710814, 0.6871932, -1.4687963, 1.4678899
8: -0.9458506, 1.5565381, -0.7382430, 1.5527720, -2.4986227, 2.2947812
9: -0.6460938, 0.8747255, -0.5604296, 0.7814227, -1.4275165, 1.4351552

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=74, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6922415, upper bound: 3.5725899
time: 1.69 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6206140, upper bound: 3.5723188
time: 1.46 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.8440781, 0.7487532, -0.8440781, 0.7487532, -1.5928314, 1.5928314
1: -0.6637570, 0.6703729, -0.6637570, 0.6703729, -1.3341299, 1.3341299
2: -0.5550326, 1.3367251, -0.5550326, 1.3367251, -1.8917577, 1.8917577
3: -0.6107832, 0.6956552, -0.6107832, 0.6956552, -1.3064384, 1.3064384
4: -0.7491193, 0.6973783, -0.7491193, 0.6973783, -1.4464977, 1.4464977
5: -0.6596397, 0.7621418, -0.6596397, 0.7621418, -1.4217815, 1.4217815
6: -0.5867524, 0.8152532, -0.5867524, 0.8152532, -1.4020057, 1.4020057
7: -0.7816032, 0.7968085, -0.7816032, 0.7968085, -1.5784117, 1.5784117
8: -0.9458506, 1.5565381, -0.9458506, 1.5565381, -2.5023885, 2.5023885
9: -0.6460938, 0.8747255, -0.6460938, 0.8747255, -1.5208192, 1.5208192

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=89, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 137

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6922415, upper bound: 3.6217940
time: 1.85 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6206140, upper bound: 3.6215576
time: 1.31 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.7498992, 0.6906404, -0.8340266, 0.7445253, -1.4944246, 1.5246670
1: -0.5940168, 0.6083853, -0.6565025, 0.6630408, -1.2570577, 1.2648878
2: -0.4509220, 1.2591189, -0.5533026, 1.3539926, -1.8049146, 1.8124216
3: -0.5034613, 0.6497082, -0.5922963, 0.6916150, -1.1950763, 1.2420045
4: -0.6432903, 0.6250859, -0.7326151, 0.6887631, -1.3320534, 1.3577011
5: -0.5921451, 0.6899128, -0.6505908, 0.7551036, -1.3472488, 1.3405037
6: -0.5016267, 0.7464218, -0.5763864, 0.8095620, -1.3111887, 1.3228083
7: -0.6961854, 0.7124593, -0.7706442, 0.7854245, -1.4816098, 1.4831035
8: -0.7909718, 1.5304153, -0.9531162, 1.5678110, -2.3587828, 2.4835315
9: -0.5786676, 0.8021612, -0.6389627, 0.8696222, -1.4482899, 1.4411240

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=35, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=88, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 137

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5693393, upper bound: 3.5753002
time: 1.71 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5693393, upper bound: 3.5753002
time: 1.51 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.6681275, 0.6344774, -0.6260393, 0.5951875, -1.2633150, 1.2605166
1: -0.5301463, 0.5486343, -0.4882822, 0.5109784, -1.0411247, 1.0369165
2: -0.3732882, 1.1852838, -0.3526736, 1.1509999, -1.5242882, 1.5379574
3: -0.4119392, 0.5995863, -0.3645411, 0.5600210, -0.9719603, 0.9641274
4: -0.5528595, 0.5541098, -0.4997985, 0.5068990, -1.0597584, 1.0539083
5: -0.5389526, 0.6179965, -0.5080813, 0.5684903, -1.1074429, 1.1260777
6: -0.4387485, 0.6799842, -0.4046544, 0.6441430, -1.0828916, 1.0846386
7: -0.6147015, 0.6344651, -0.5632961, 0.5735250, -1.1882265, 1.1977613
8: -0.6552932, 1.5269337, -0.5923908, 1.5771157, -2.2309184, 2.1193242
9: -0.5192574, 0.7358742, -0.4804976, 0.7054753, -1.2247328, 1.2163718

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=48, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.0545243, upper bound: 3.3253714
time: 1.27 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5444685, upper bound: 3.5507837
time: 1.50 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.8849511, 0.7712959, -0.7094821, 0.6650710, -1.5500220, 1.4807780
1: -0.6948996, 0.6947341, -0.5609788, 0.5800449, -1.2749445, 1.2557129
2: -0.6027899, 1.3758440, -0.4091291, 1.2244315, -1.8272214, 1.7849731
3: -0.6593508, 0.7152038, -0.4555055, 0.6259102, -1.2852609, 1.1707093
4: -0.7941504, 0.7288600, -0.5963171, 0.5904274, -1.3845778, 1.3251772
5: -0.6923543, 0.7898130, -0.5641767, 0.6563267, -1.3486810, 1.3539896
6: -0.6204342, 0.8444942, -0.4730721, 0.7126814, -1.3331156, 1.3175663
7: -0.8214942, 0.8329949, -0.6569301, 0.6723369, -1.4938312, 1.4899250
8: -1.0156063, 1.5348587, -0.7264873, 1.5624757, -2.5780811, 2.2613456
9: -0.6755854, 0.9032372, -0.5516040, 0.7706635, -1.4462490, 1.4548413

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=35, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=94, inp2_unstable=65, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 137

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6695118, upper bound: 3.5193911
time: 1.56 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6114792, upper bound: 3.5190783
time: 1.55 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.8849511, 0.7712959, -0.8742264, 0.7670830, -1.6520340, 1.6455224
1: -0.6948996, 0.6947341, -0.6864972, 0.6892030, -1.3841026, 1.3812313
2: -0.6027899, 1.3758440, -0.5923334, 1.3691396, -1.9719296, 1.9681774
3: -0.6593508, 0.7152038, -0.6446398, 0.7102579, -1.3696086, 1.3598435
4: -0.7941504, 0.7288600, -0.7812951, 0.7202976, -1.5144480, 1.5101551
5: -0.6923543, 0.7898130, -0.6831884, 0.7839862, -1.4763404, 1.4730015
6: -0.6204342, 0.8444942, -0.6132211, 0.8373356, -1.4577699, 1.4577153
7: -0.8214942, 0.8329949, -0.8100977, 0.8226141, -1.6441083, 1.6430926
8: -1.0156063, 1.5348587, -1.0035493, 1.5676322, -2.5832381, 2.5384076
9: -0.6755854, 0.9032372, -0.6684821, 0.8983816, -1.5739670, 1.5717194

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=94, inp2_unstable=92, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6695118, upper bound: 3.5767451
time: 1.57 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6114792, upper bound: 3.5764608
time: 2.37 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.7498992, 0.6906404, -0.7912142, 0.7206212, -1.4705205, 1.4818547
1: -0.5940168, 0.6083853, -0.6244653, 0.6372649, -1.2312818, 1.2328506
2: -0.4509220, 1.2591189, -0.5065266, 1.3146231, -1.7655451, 1.7656455
3: -0.5034613, 0.6497082, -0.5438306, 0.6706591, -1.1741204, 1.1935389
4: -0.6432903, 0.6250859, -0.6856831, 0.6560810, -1.2993712, 1.3107691
5: -0.5921451, 0.6899128, -0.6199667, 0.7257146, -1.3178596, 1.3098795
6: -0.5016267, 0.7464218, -0.5422966, 0.7782614, -1.2798880, 1.2887185
7: -0.6961854, 0.7124593, -0.7312356, 0.7468467, -1.4430320, 1.4436948
8: -0.7909718, 1.5304153, -0.8826150, 1.5933522, -2.3843238, 2.4130301
9: -0.5786676, 0.8021612, -0.6101183, 0.8399189, -1.4185865, 1.4122795

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=82, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 137

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5696654, upper bound: 3.5778689
time: 1.38 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5696654, upper bound: 3.5778689
time: 1.33 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.6681275, 0.6344774, -0.6127387, 0.5871146, -1.2552421, 1.2472162
1: -0.5301463, 0.5486343, -0.4769712, 0.5011066, -1.0312529, 1.0256054
2: -0.3732882, 1.1852838, -0.3474104, 1.1251478, -1.4984361, 1.5326942
3: -0.4119392, 0.5995863, -0.3517099, 0.5507427, -0.9626820, 0.9512962
4: -0.5528595, 0.5541098, -0.4877988, 0.4954478, -1.0483072, 1.0419085
5: -0.5389526, 0.6179965, -0.4992151, 0.5583050, -1.0972576, 1.1172116
6: -0.4387485, 0.6799842, -0.3971912, 0.6330128, -1.0717614, 1.0771754
7: -0.6147015, 0.6344651, -0.5520501, 0.5588120, -1.1735134, 1.1865151
8: -0.6552932, 1.5269337, -0.5576983, 1.6029645, -2.2568064, 2.0846319
9: -0.5192574, 0.7358742, -0.4705511, 0.6978772, -1.2171347, 1.2064253

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=48, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.0548981, upper bound: 3.3255452
time: 1.37 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5448181, upper bound: 3.5546708
time: 1.58 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.8849511, 0.7712959, -0.8530254, 0.7574339, -1.6423850, 1.6243213
1: -0.6948996, 0.6947341, -0.6702126, 0.6770992, -1.3719988, 1.3649466
2: -0.6027899, 1.3758440, -0.5743517, 1.3624271, -1.9652171, 1.9501957
3: -0.6593508, 0.7152038, -0.6159139, 0.7004040, -1.3597548, 1.3311176
4: -0.7941504, 0.7288600, -0.7554982, 0.7035162, -1.4976666, 1.4843582
5: -0.6923543, 0.7898130, -0.6654902, 0.7711605, -1.4635148, 1.4553032
6: -0.6204342, 0.8444942, -0.5970553, 0.8228754, -1.4433095, 1.4415495
7: -0.8214942, 0.8329949, -0.7886799, 0.8024640, -1.6239582, 1.6216748
8: -1.0156063, 1.5348587, -0.9806290, 1.5964582, -2.6120644, 2.5154877
9: -0.6755854, 0.9032372, -0.6540712, 0.8859549, -1.5615404, 1.5573084

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=94, inp2_unstable=90, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6186636, upper bound: 3.5790925
time: 1.48 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6186636, upper bound: 3.5790925
time: 1.34 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.7770543, 0.7079840, -0.6440119, 0.6118391, -1.3888934, 1.3519958
1: -0.6146731, 0.6256856, -0.5041932, 0.5267908, -1.1414640, 1.1298788
2: -0.4881168, 1.3011345, -0.3633332, 1.1695341, -1.6576509, 1.6644677
3: -0.5298887, 0.6637646, -0.3819892, 0.5737923, -1.1036811, 1.0457538
4: -0.6706136, 0.6453261, -0.5193256, 0.5245317, -1.1951454, 1.1646516
5: -0.6105844, 0.7110474, -0.5214867, 0.5886931, -1.1992776, 1.2325342
6: -0.5239559, 0.7675499, -0.4217668, 0.6589385, -1.1828943, 1.1893167
7: -0.7197153, 0.7353213, -0.5816638, 0.5936893, -1.3134046, 1.3169851
8: -0.8529980, 1.5312879, -0.6266382, 1.6057832, -2.4572849, 2.1579261
9: -0.5979138, 0.8239633, -0.4973323, 0.7214990, -1.3194128, 1.3212956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=82, inp2_unstable=50, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.0789234, upper bound: 3.3462310
time: 1.24 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5940237, upper bound: 3.5562055
time: 1.69 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.7222407, 0.6742139, -0.8340266, 0.7445253, -1.4667661, 1.5082405
1: -0.5717611, 0.5904015, -0.6565025, 0.6630408, -1.2348020, 1.2469040
2: -0.4184340, 1.2256221, -0.5533026, 1.3539926, -1.7724266, 1.7789247
3: -0.4736900, 0.6343694, -0.5922963, 0.6916150, -1.1653050, 1.2266657
4: -0.6134740, 0.6030147, -0.7326151, 0.6887631, -1.3022370, 1.3356297
5: -0.5731425, 0.6689765, -0.6505908, 0.7551036, -1.3282461, 1.3195673
6: -0.4829884, 0.7237632, -0.5763864, 0.8095620, -1.2925503, 1.3001496
7: -0.6710814, 0.6871932, -0.7706442, 0.7854245, -1.4565058, 1.4578373
8: -0.7382430, 1.5527720, -0.9531162, 1.5678110, -2.3060541, 2.5058880
9: -0.5604296, 0.7814227, -0.6389627, 0.8696222, -1.4300518, 1.4203854

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=35, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=74, inp2_unstable=88, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5695649, upper bound: 3.5754114
time: 1.36 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5695649, upper bound: 3.5754114
time: 1.36 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.6516190, 0.6240643, -0.6260393, 0.5951875, -1.2468064, 1.2501036
1: -0.5168293, 0.5368063, -0.4882822, 0.5109784, -1.0278077, 1.0250885
2: -0.3630740, 1.1548004, -0.3526736, 1.1509999, -1.5140738, 1.5074741
3: -0.3967934, 0.5880464, -0.3645411, 0.5600210, -0.9568144, 0.9525876
4: -0.5367474, 0.5402665, -0.4997985, 0.5068990, -1.0436463, 1.0400651
5: -0.5282872, 0.6039501, -0.5080813, 0.5684903, -1.0967776, 1.1120315
6: -0.4285384, 0.6665362, -0.4046544, 0.6441430, -1.0726814, 1.0711906
7: -0.5999638, 0.6171570, -0.5632961, 0.5735250, -1.1734889, 1.1804531
8: -0.6140552, 1.5491552, -0.5923908, 1.5771157, -2.1896515, 2.1405494
9: -0.5054486, 0.7261571, -0.4804976, 0.7054753, -1.2109239, 1.2066547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=48, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.0537355, upper bound: 3.3229289
time: 1.31 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5453498, upper bound: 3.5509021
time: 1.52 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.8440781, 0.7487532, -0.7094821, 0.6650710, -1.5091491, 1.4582353
1: -0.6637570, 0.6703729, -0.5609788, 0.5800449, -1.2438020, 1.2313516
2: -0.5550326, 1.3367251, -0.4091291, 1.2244315, -1.7794640, 1.7458541
3: -0.6107832, 0.6956552, -0.4555055, 0.6259102, -1.2366934, 1.1511607
4: -0.7491193, 0.6973783, -0.5963171, 0.5904274, -1.3395467, 1.2936954
5: -0.6596397, 0.7621418, -0.5641767, 0.6563267, -1.3159664, 1.3263185
6: -0.5867524, 0.8152532, -0.4730721, 0.7126814, -1.2994337, 1.2883253
7: -0.7816032, 0.7968085, -0.6569301, 0.6723369, -1.4539400, 1.4537387
8: -0.9458506, 1.5565381, -0.7264873, 1.5624757, -2.5083263, 2.2830253
9: -0.6460938, 0.8747255, -0.5516040, 0.7706635, -1.4167573, 1.4263295

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=35, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=65, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6721805, upper bound: 3.5195519
time: 1.63 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6117901, upper bound: 3.5192074
time: 1.32 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.8440781, 0.7487532, -0.8742264, 0.7670830, -1.6111612, 1.6229796
1: -0.6637570, 0.6703729, -0.6864972, 0.6892030, -1.3529601, 1.3568701
2: -0.5550326, 1.3367251, -0.5923334, 1.3691396, -1.9241722, 1.9290586
3: -0.6107832, 0.6956552, -0.6446398, 0.7102579, -1.3210411, 1.3402951
4: -0.7491193, 0.6973783, -0.7812951, 0.7202976, -1.4694170, 1.4786735
5: -0.6596397, 0.7621418, -0.6831884, 0.7839862, -1.4436258, 1.4453303
6: -0.5867524, 0.8152532, -0.6132211, 0.8373356, -1.4240880, 1.4284743
7: -0.7816032, 0.7968085, -0.8100977, 0.8226141, -1.6042173, 1.6069062
8: -0.9458506, 1.5565381, -1.0035493, 1.5676322, -2.5134826, 2.5600874
9: -0.6460938, 0.8747255, -0.6684821, 0.8983816, -1.5444753, 1.5432076

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=92, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6721805, upper bound: 3.5768996
time: 1.89 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6117902, upper bound: 3.5765855
time: 1.61 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.7222407, 0.6742139, -0.7912142, 0.7206212, -1.4428620, 1.4654281
1: -0.5717611, 0.5904015, -0.6244653, 0.6372649, -1.2090261, 1.2148668
2: -0.4184340, 1.2256221, -0.5065266, 1.3146231, -1.7330571, 1.7321486
3: -0.4736900, 0.6343694, -0.5438306, 0.6706591, -1.1443491, 1.1782000
4: -0.6134740, 0.6030147, -0.6856831, 0.6560810, -1.2695550, 1.2886978
5: -0.5731425, 0.6689765, -0.6199667, 0.7257146, -1.2988571, 1.2889432
6: -0.4829884, 0.7237632, -0.5422966, 0.7782614, -1.2612498, 1.2660598
7: -0.6710814, 0.6871932, -0.7312356, 0.7468467, -1.4179281, 1.4184288
8: -0.7382430, 1.5527720, -0.8826150, 1.5933522, -2.3315949, 2.4353871
9: -0.5604296, 0.7814227, -0.6101183, 0.8399189, -1.4003484, 1.3915410

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=74, inp2_unstable=82, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5697800, upper bound: 3.5780980
time: 1.55 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5697800, upper bound: 3.5780980
time: 1.28 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.6516190, 0.6240643, -0.6127387, 0.5871146, -1.2387335, 1.2368031
1: -0.5168293, 0.5368063, -0.4769712, 0.5011066, -1.0179359, 1.0137775
2: -0.3630740, 1.1548004, -0.3474104, 1.1251478, -1.4882218, 1.5022109
3: -0.3967934, 0.5880464, -0.3517099, 0.5507427, -0.9475361, 0.9397564
4: -0.5367474, 0.5402665, -0.4877988, 0.4954478, -1.0321951, 1.0280652
5: -0.5282872, 0.6039501, -0.4992151, 0.5583050, -1.0865922, 1.1031651
6: -0.4285384, 0.6665362, -0.3971912, 0.6330128, -1.0615512, 1.0637274
7: -0.5999638, 0.6171570, -0.5520501, 0.5588120, -1.1587758, 1.1692071
8: -0.6140552, 1.5491552, -0.5576983, 1.6029645, -2.2154894, 2.1057253
9: -0.5054486, 0.7261571, -0.4705511, 0.6978772, -1.2033257, 1.1967082

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=48, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.0537355, upper bound: 3.3229289
time: 1.39 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5457576, upper bound: 3.5549962
time: 1.32 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.8440781, 0.7487532, -0.8530254, 0.7574339, -1.6015120, 1.6017786
1: -0.6637570, 0.6703729, -0.6702126, 0.6770992, -1.3408562, 1.3405855
2: -0.5550326, 1.3367251, -0.5743517, 1.3624271, -1.9174597, 1.9110768
3: -0.6107832, 0.6956552, -0.6159139, 0.7004040, -1.3111873, 1.3115691
4: -0.7491193, 0.6973783, -0.7554982, 0.7035162, -1.4526355, 1.4528766
5: -0.6596397, 0.7621418, -0.6654902, 0.7711605, -1.4308002, 1.4276320
6: -0.5867524, 0.8152532, -0.5970553, 0.8228754, -1.4096278, 1.4123085
7: -0.7816032, 0.7968085, -0.7886799, 0.8024640, -1.5840671, 1.5854884
8: -0.9458506, 1.5565381, -0.9806290, 1.5964582, -2.5423086, 2.5371671
9: -0.6460938, 0.8747255, -0.6540712, 0.8859549, -1.5320487, 1.5287967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=90, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 137

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6197358, upper bound: 3.5793037
time: 1.50 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6197358, upper bound: 3.5793037
time: 1.47 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.7430959, 0.6879959, -0.6440119, 0.6118391, -1.3549349, 1.3320076
1: -0.5879996, 0.6040083, -0.5041932, 0.5267908, -1.1147904, 1.1082015
2: -0.4489284, 1.2633713, -0.3633332, 1.1695341, -1.6184624, 1.6267046
3: -0.4928470, 0.6455711, -0.3819892, 0.5737923, -1.0666393, 1.0275602
4: -0.6329191, 0.6187740, -0.5193256, 0.5245317, -1.1574509, 1.1380996
5: -0.5869537, 0.6854190, -0.5214867, 0.5886931, -1.1756468, 1.2069056
6: -0.4985793, 0.7408842, -0.4217668, 0.6589385, -1.1575177, 1.1626511
7: -0.6896145, 0.7039613, -0.5816638, 0.5936893, -1.2833037, 1.2856250
8: -0.7915496, 1.5527903, -0.6266382, 1.6057832, -2.3958004, 2.1794286
9: -0.5756978, 0.7988176, -0.4973323, 0.7214990, -1.2971969, 1.2961500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=76, inp2_unstable=50, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 144

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.2507201, upper bound: 3.4673341
time: 1.84 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5882571, upper bound: 3.5474993
time: 1.81 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.7094821, 0.6650710, -0.8582047, 0.7562012, -1.4656833, 1.5232756
1: -0.5609788, 0.5800449, -0.6748229, 0.6770743, -1.2380531, 1.2548678
2: -0.4091291, 1.2244315, -0.5773535, 1.3683352, -1.7774644, 1.8017850
3: -0.4555055, 0.6259102, -0.6231498, 0.7028407, -1.1583462, 1.2490599
4: -0.5963171, 0.5904274, -0.7609344, 0.7073906, -1.3037077, 1.3513618
5: -0.5641767, 0.6563267, -0.6697260, 0.7704628, -1.3346394, 1.3260527
6: -0.4730721, 0.7126814, -0.5949098, 0.8261688, -1.2992408, 1.3075912
7: -0.6569301, 0.6723369, -0.7950886, 0.8075733, -1.4645034, 1.4674255
8: -0.7264873, 1.5624757, -0.9844482, 1.5345674, -2.2610545, 2.5469236
9: -0.5516040, 0.7706635, -0.6556405, 0.8841013, -1.4357053, 1.4263040

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=65, inp2_unstable=90, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5190783, upper bound: 3.6114792
time: 1.30 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5190783, upper bound: 3.6114792
time: 1.38 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.6424738, 0.6146749, -0.6436942, 0.6097843, -1.2522581, 1.2583690
1: -0.5073165, 0.5281041, -0.5053918, 0.5260400, -1.0333565, 1.0334959
2: -0.3593611, 1.1514735, -0.3621349, 1.1757877, -1.5351487, 1.5136085
3: -0.3860783, 0.5786694, -0.3836681, 0.5753074, -0.9613857, 0.9623376
4: -0.5239696, 0.5293781, -0.5198111, 0.5253024, -1.0492719, 1.0491891
5: -0.5214326, 0.5922335, -0.5210704, 0.5869219, -1.1083546, 1.1133039
6: -0.4205707, 0.6588503, -0.4175265, 0.6594937, -1.0800644, 1.0763768
7: -0.5879630, 0.6024114, -0.5834085, 0.5967019, -1.1846650, 1.1858199
8: -0.6037260, 1.5587642, -0.6295719, 1.5443664, -2.1466246, 2.1873486
9: -0.4966357, 0.7191773, -0.4955094, 0.7172167, -1.2138524, 1.2146868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=51, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.9660847, upper bound: 3.3045510
time: 1.34 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.4943521, upper bound: 3.5867418
time: 1.23 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.8742264, 0.7670830, -0.7498992, 0.6906404, -1.5648668, 1.5169823
1: -0.6864972, 0.6892030, -0.5940168, 0.6083853, -1.2948824, 1.2832198
2: -0.5923334, 1.3691396, -0.4509220, 1.2591189, -1.8514524, 1.8200616
3: -0.6446398, 0.7102579, -0.5034613, 0.6497082, -1.2943480, 1.2137191
4: -0.7812951, 0.7202976, -0.6432903, 0.6250859, -1.4063810, 1.3635879
5: -0.6831884, 0.7839862, -0.5921451, 0.6899128, -1.3731012, 1.3761313
6: -0.6132211, 0.8373356, -0.5016267, 0.7464218, -1.3596430, 1.3389623
7: -0.8100977, 0.8226141, -0.6961854, 0.7124593, -1.5225570, 1.5187995
8: -1.0035493, 1.5676322, -0.7909718, 1.5304153, -2.5339646, 2.3586040
9: -0.6684821, 0.8983816, -0.5786676, 0.8021612, -1.4706433, 1.4770492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=92, inp2_unstable=78, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6619132, upper bound: 3.5701044
time: 1.40 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5753002, upper bound: 3.5693393
time: 1.38 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.8742264, 0.7670830, -0.8849511, 0.7712959, -1.6455224, 1.6520340
1: -0.6864972, 0.6892030, -0.6948996, 0.6947341, -1.3812313, 1.3841026
2: -0.5923334, 1.3691396, -0.6027899, 1.3758440, -1.9681774, 1.9719296
3: -0.6446398, 0.7102579, -0.6593508, 0.7152038, -1.3598435, 1.3696086
4: -0.7812951, 0.7202976, -0.7941504, 0.7288600, -1.5101551, 1.5144480
5: -0.6831884, 0.7839862, -0.6923543, 0.7898130, -1.4730015, 1.4763404
6: -0.6132211, 0.8373356, -0.6204342, 0.8444942, -1.4577153, 1.4577699
7: -0.8100977, 0.8226141, -0.8214942, 0.8329949, -1.6430926, 1.6441083
8: -1.0035493, 1.5676322, -1.0156063, 1.5348587, -2.5384073, 2.5832379
9: -0.6684821, 0.8983816, -0.6755854, 0.9032372, -1.5717194, 1.5739670

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=92, inp2_unstable=94, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6619132, upper bound: 3.6189651
time: 1.52 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5753002, upper bound: 3.6183980
time: 1.59 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.7094821, 0.6650710, -0.8334478, 0.7441336, -1.4536157, 1.4985187
1: -0.5609788, 0.5800449, -0.6560838, 0.6626592, -1.2236381, 1.2361287
2: -0.4091291, 1.2244315, -0.5525093, 1.3532078, -1.7623370, 1.7769408
3: -0.4555055, 0.6259102, -0.5917079, 0.6913407, -1.1468462, 1.2176180
4: -0.5963171, 0.5904274, -0.7320283, 0.6883411, -1.2846582, 1.3224556
5: -0.5641767, 0.6563267, -0.6501848, 0.7546390, -1.3188157, 1.3065115
6: -0.4730721, 0.7126814, -0.5758235, 0.8091384, -1.2822105, 1.2885048
7: -0.6569301, 0.6723369, -0.7701181, 0.7849550, -1.4418851, 1.4424549
8: -0.7264873, 1.5624757, -0.9516553, 1.5670443, -2.2935314, 2.5141308
9: -0.5516040, 0.7706635, -0.6385127, 0.8691258, -1.4207299, 1.4091762

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=65, inp2_unstable=88, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5187074, upper bound: 3.5743171
time: 1.18 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5187074, upper bound: 3.5743173
time: 1.23 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.6424738, 0.6146749, -0.6255849, 0.5948222, -1.2372961, 1.2402598
1: -0.5073165, 0.5281041, -0.4880233, 0.5106414, -1.0179579, 1.0161275
2: -0.3593611, 1.1514735, -0.3523021, 1.1503584, -1.5097195, 1.5037756
3: -0.3860783, 0.5786694, -0.3642755, 0.5598359, -0.9459141, 0.9429449
4: -0.5239696, 0.5293781, -0.4994429, 0.5066095, -1.0305791, 1.0288210
5: -0.5214326, 0.5922335, -0.5077643, 0.5680562, -1.0894889, 1.0999979
6: -0.4205707, 0.6588503, -0.4041807, 0.6438341, -1.0644047, 1.0630310
7: -0.5879630, 0.6024114, -0.5630331, 0.5732810, -1.1612442, 1.1654446
8: -0.6037260, 1.5587642, -0.5911899, 1.5744008, -2.1764307, 2.1488154
9: -0.4966357, 0.7191773, -0.4800379, 0.7050371, -1.2016728, 1.1992152

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=48, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.9655738, upper bound: 3.2916273
time: 1.64 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.4939823, upper bound: 3.5497608
time: 1.29 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.8742264, 0.7670830, -0.7058572, 0.6625150, -1.5367414, 1.4729402
1: -0.6864972, 0.6892030, -0.5581445, 0.5773371, -1.2638342, 1.2473476
2: -0.5923334, 1.3691396, -0.4061656, 1.2214956, -1.8138291, 1.7753052
3: -0.6446398, 0.7102579, -0.4512085, 0.6236278, -1.2682676, 1.1614664
4: -0.7812951, 0.7202976, -0.5923377, 0.5870969, -1.3683920, 1.3126353
5: -0.6831884, 0.7839862, -0.5619008, 0.6530196, -1.3362081, 1.3458869
6: -0.6132211, 0.8373356, -0.4702115, 0.7098351, -1.3230562, 1.3075471
7: -0.8100977, 0.8226141, -0.6531004, 0.6688879, -1.4789855, 1.4757146
8: -1.0035493, 1.5676322, -0.7205987, 1.5618743, -2.5654235, 2.2882309
9: -0.6684821, 0.8983816, -0.5489818, 0.7675232, -1.4360054, 1.4473634

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=35, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=92, inp2_unstable=65, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6574076, upper bound: 3.5190690
time: 1.87 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5743172, upper bound: 3.5187074
time: 1.38 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.8742264, 0.7670830, -0.8690663, 0.7640336, -1.6382599, 1.6361494
1: -0.6864972, 0.6892030, -0.6825625, 0.6858301, -1.3723272, 1.3717655
2: -0.5923334, 1.3691396, -0.5865808, 1.3658108, -1.9581442, 1.9557204
3: -0.6446398, 0.7102579, -0.6381330, 0.7078161, -1.3524559, 1.3483908
4: -0.7812951, 0.7202976, -0.7752427, 0.7162150, -1.4975101, 1.4955404
5: -0.6831884, 0.7839862, -0.6788208, 0.7802136, -1.4634020, 1.4628069
6: -0.6132211, 0.8373356, -0.6083322, 0.8337021, -1.4469233, 1.4456677
7: -0.8100977, 0.8226141, -0.8051032, 0.8177836, -1.6278813, 1.6277173
8: -1.0035493, 1.5676322, -0.9954921, 1.5667419, -2.5702913, 2.5631239
9: -0.6684821, 0.8983816, -0.6646044, 0.8945228, -1.5630050, 1.5629860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=92, inp2_unstable=92, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6574076, upper bound: 3.5764382
time: 1.60 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5743173, upper bound: 3.5760984
time: 1.52 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.7912142, 0.7206212, -0.7498992, 0.6906404, -1.4818547, 1.4705205
1: -0.6244653, 0.6372649, -0.5940168, 0.6083853, -1.2328506, 1.2312818
2: -0.5065266, 1.3146231, -0.4509220, 1.2591189, -1.7656455, 1.7655451
3: -0.5438306, 0.6706591, -0.5034613, 0.6497082, -1.1935389, 1.1741204
4: -0.6856831, 0.6560810, -0.6432903, 0.6250859, -1.3107691, 1.2993712
5: -0.6199667, 0.7257146, -0.5921451, 0.6899128, -1.3098795, 1.3178596
6: -0.5422966, 0.7782614, -0.5016267, 0.7464218, -1.2887185, 1.2798880
7: -0.7312356, 0.7468467, -0.6961854, 0.7124593, -1.4436948, 1.4430320
8: -0.8826150, 1.5933522, -0.7909718, 1.5304153, -2.4130301, 2.3843241
9: -0.6101183, 0.8399189, -0.5786676, 0.8021612, -1.4122795, 1.4185865

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=82, inp2_unstable=78, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 144

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.3477684, upper bound: 3.4461180
time: 1.70 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6439715, upper bound: 3.5409959
time: 1.39 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.8530254, 0.7574339, -0.8849511, 0.7712959, -1.6243213, 1.6423850
1: -0.6702126, 0.6770992, -0.6948996, 0.6947341, -1.3649466, 1.3719988
2: -0.5743517, 1.3624271, -0.6027899, 1.3758440, -1.9501957, 1.9652171
3: -0.6159139, 0.7004040, -0.6593508, 0.7152038, -1.3311176, 1.3597548
4: -0.7554982, 0.7035162, -0.7941504, 0.7288600, -1.4843582, 1.4976666
5: -0.6654902, 0.7711605, -0.6923543, 0.7898130, -1.4553032, 1.4635148
6: -0.5970553, 0.8228754, -0.6204342, 0.8444942, -1.4415495, 1.4433095
7: -0.7886799, 0.8024640, -0.8214942, 0.8329949, -1.6216748, 1.6239582
8: -0.9806290, 1.5964582, -1.0156063, 1.5348587, -2.5154877, 2.6120644
9: -0.6540712, 0.8859549, -0.6755854, 0.9032372, -1.5573084, 1.5615404

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=90, inp2_unstable=94, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 144

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6446634, upper bound: 3.6162584
time: 1.97 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6446634, upper bound: 3.6191821
time: 1.62 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.6127387, 0.5871146, -0.6681275, 0.6344774, -1.2472162, 1.2552421
1: -0.4769712, 0.5011066, -0.5301463, 0.5486343, -1.0256054, 1.0312529
2: -0.3474104, 1.1251478, -0.3732882, 1.1852838, -1.5326942, 1.4984361
3: -0.3517099, 0.5507427, -0.4119392, 0.5995863, -0.9512962, 0.9626820
4: -0.4877988, 0.4954478, -0.5528595, 0.5541098, -1.0419085, 1.0483072
5: -0.4992151, 0.5583050, -0.5389526, 0.6179965, -1.1172116, 1.0972576
6: -0.3971912, 0.6330128, -0.4387485, 0.6799842, -1.0771754, 1.0717614
7: -0.5520501, 0.5588120, -0.6147015, 0.6344651, -1.1865151, 1.1735134
8: -0.5576983, 1.6029645, -0.6552932, 1.5269337, -2.0846319, 2.2568059
9: -0.4705511, 0.6978772, -0.5192574, 0.7358742, -1.2064253, 1.2171347

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=53, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 144

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.0077403, upper bound: 3.2849560
time: 1.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5546707, upper bound: 3.5448182
time: 1.42 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.6440119, 0.6118391, -0.7770543, 0.7079840, -1.3519958, 1.3888934
1: -0.5041932, 0.5267908, -0.6146731, 0.6256856, -1.1298788, 1.1414640
2: -0.3633332, 1.1695341, -0.4881168, 1.3011345, -1.6644677, 1.6576509
3: -0.3819892, 0.5737923, -0.5298887, 0.6637646, -1.0457538, 1.1036811
4: -0.5193256, 0.5245317, -0.6706136, 0.6453261, -1.1646516, 1.1951454
5: -0.5214867, 0.5886931, -0.6105844, 0.7110474, -1.2325342, 1.1992776
6: -0.4217668, 0.6589385, -0.5239559, 0.7675499, -1.1893167, 1.1828943
7: -0.5816638, 0.5936893, -0.7197153, 0.7353213, -1.3169851, 1.3134046
8: -0.6266382, 1.6057832, -0.8529980, 1.5312879, -2.1579261, 2.4572852
9: -0.4973323, 0.7214990, -0.5979138, 0.8239633, -1.3212956, 1.3194128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=82, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 144

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5231919, upper bound: 3.6119127
time: 1.26 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5231919, upper bound: 3.6186637
time: 1.46 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.7912142, 0.7206212, -0.7058572, 0.6625150, -1.4537292, 1.4264784
1: -0.6244653, 0.6372649, -0.5581445, 0.5773371, -1.2018024, 1.1954094
2: -0.5065266, 1.3146231, -0.4061656, 1.2214956, -1.7280222, 1.7207887
3: -0.5438306, 0.6706591, -0.4512085, 0.6236278, -1.1674584, 1.1218677
4: -0.6856831, 0.6560810, -0.5923377, 0.5870969, -1.2727799, 1.2484187
5: -0.6199667, 0.7257146, -0.5619008, 0.6530196, -1.2729863, 1.2876154
6: -0.5422966, 0.7782614, -0.4702115, 0.7098351, -1.2521317, 1.2484729
7: -0.7312356, 0.7468467, -0.6531004, 0.6688879, -1.4001236, 1.3999472
8: -0.8826150, 1.5933522, -0.7205987, 1.5618743, -2.4444892, 2.3139505
9: -0.6101183, 0.8399189, -0.5489818, 0.7675232, -1.3776414, 1.3889008

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=35, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=82, inp2_unstable=65, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 144

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.3317755, upper bound: 3.4201425
time: 1.46 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6377287, upper bound: 3.4889880
time: 1.43 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.8530254, 0.7574339, -0.8690663, 0.7640336, -1.6170590, 1.6265001
1: -0.6702126, 0.6770992, -0.6825625, 0.6858301, -1.3560426, 1.3596616
2: -0.5743517, 1.3624271, -0.5865808, 1.3658108, -1.9401624, 1.9490080
3: -0.6159139, 0.7004040, -0.6381330, 0.7078161, -1.3237300, 1.3385370
4: -0.7554982, 0.7035162, -0.7752427, 0.7162150, -1.4717132, 1.4787589
5: -0.6654902, 0.7711605, -0.6788208, 0.7802136, -1.4457037, 1.4499815
6: -0.5970553, 0.8228754, -0.6083322, 0.8337021, -1.4307574, 1.4312075
7: -0.7886799, 0.8024640, -0.8051032, 0.8177836, -1.6064634, 1.6075672
8: -0.9806290, 1.5964582, -0.9954921, 1.5667419, -2.5473707, 2.5919504
9: -0.6540712, 0.8859549, -0.6646044, 0.8945228, -1.5485940, 1.5505593

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=90, inp2_unstable=92, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 144

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6413911, upper bound: 3.5753957
time: 1.53 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6413911, upper bound: 3.5766225
time: 1.59 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.6127387, 0.5871146, -0.6405416, 0.6130708, -1.2258096, 1.2276562
1: -0.4769712, 0.5011066, -0.5056301, 0.5265185, -1.0034897, 1.0067368
2: -0.3474104, 1.1251478, -0.3584019, 1.1493610, -1.4967715, 1.4835497
3: -0.3517099, 0.5507427, -0.3842203, 0.5771338, -0.9288437, 0.9349630
4: -0.4877988, 0.4954478, -0.5217917, 0.5275503, -1.0153490, 1.0172395
5: -0.4992151, 0.5583050, -0.5200908, 0.5902042, -1.0894192, 1.0783958
6: -0.3971912, 0.6330128, -0.4189952, 0.6573092, -1.0545003, 1.0520079
7: -0.5520501, 0.5588120, -0.5860267, 0.6000727, -1.1521227, 1.1448387
8: -0.5576983, 1.6029645, -0.5999986, 1.5582500, -2.1147649, 2.2012863
9: -0.4705511, 0.6978772, -0.4948884, 0.7177141, -1.1882652, 1.1927657

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=50, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 144

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.0051862, upper bound: 3.2657957
time: 1.37 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5536203, upper bound: 3.4941026
time: 1.38 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.6440119, 0.6118391, -0.7606057, 0.6997439, -1.3437557, 1.3724447
1: -0.5041932, 0.5267908, -0.6017329, 0.6154308, -1.1196240, 1.1285236
2: -0.3633332, 1.1695341, -0.4720040, 1.2893022, -1.6526355, 1.6415381
3: -0.3819892, 0.5737923, -0.5097950, 0.6551446, -1.0371338, 1.0835874
4: -0.5193256, 0.5245317, -0.6508079, 0.6322190, -1.1515446, 1.1753396
5: -0.5214867, 0.5886931, -0.5989046, 0.6995243, -1.2210110, 1.1875978
6: -0.4217668, 0.6589385, -0.5126008, 0.7550837, -1.1768506, 1.1715393
7: -0.5816638, 0.5936893, -0.7050179, 0.7187584, -1.3004222, 1.2987072
8: -0.6266382, 1.6057832, -0.8312756, 1.5629337, -2.1895719, 2.4353275
9: -0.4973323, 0.7214990, -0.5879540, 0.8139456, -1.3112779, 1.3094530

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=35, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=78, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 144

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5228240, upper bound: 3.5744655
time: 1.59 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5228240, upper bound: 3.5762089
time: 1.54 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.7094821, 0.6650710, -0.8105316, 0.7298173, -1.4392995, 1.4756026
1: -0.5609788, 0.5800449, -0.6392021, 0.6485250, -1.2095038, 1.2192471
2: -0.4091291, 1.2244315, -0.5240676, 1.3259603, -1.7350893, 1.7484992
3: -0.4555055, 0.6259102, -0.5673954, 0.6800897, -1.1355951, 1.1933055
4: -0.5963171, 0.5904274, -0.7083856, 0.6711948, -1.2675118, 1.2988131
5: -0.5641767, 0.6563267, -0.6335961, 0.7378805, -1.3020571, 1.2899228
6: -0.4730721, 0.7126814, -0.5562324, 0.7921594, -1.2652315, 1.2689137
7: -0.6569301, 0.6723369, -0.7487205, 0.7658302, -1.4227602, 1.4210573
8: -0.7264873, 1.5624757, -0.9062009, 1.5564458, -2.2829332, 2.4686766
9: -0.5516040, 0.7706635, -0.6216494, 0.8509103, -1.4025143, 1.3923129

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=65, inp2_unstable=85, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5192074, upper bound: 3.6117902
time: 1.59 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5192074, upper bound: 3.6117902
time: 1.45 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.6424738, 0.6146749, -0.6336532, 0.6037244, -1.2461982, 1.2483280
1: -0.5073165, 0.5281041, -0.4970237, 0.5189049, -1.0262215, 1.0251279
2: -0.3593611, 1.1514735, -0.3559743, 1.1531411, -1.5125022, 1.5074478
3: -0.3860783, 0.5786694, -0.3744665, 0.5683330, -0.9544113, 0.9531360
4: -0.5239696, 0.5293781, -0.5105125, 0.5169896, -1.0409592, 1.0398905
5: -0.5214326, 0.5922335, -0.5143556, 0.5789285, -1.1003611, 1.1065891
6: -0.4205707, 0.6588503, -0.4116655, 0.6512663, -1.0718369, 1.0705158
7: -0.5879630, 0.6024114, -0.5743891, 0.5862142, -1.1741773, 1.1768005
8: -0.6037260, 1.5587642, -0.6005872, 1.5663713, -2.1686077, 2.1583562
9: -0.4966357, 0.7191773, -0.4875829, 0.7117791, -1.2084148, 1.2067602

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=49, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.9661738, upper bound: 3.3045638
time: 1.26 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.4944868, upper bound: 3.5877776
time: 1.31 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.8742264, 0.7670830, -0.7222407, 0.6742139, -1.5484403, 1.4893239
1: -0.6864972, 0.6892030, -0.5717611, 0.5904015, -1.2768986, 1.2609642
2: -0.5923334, 1.3691396, -0.4184340, 1.2256221, -1.8179555, 1.7875736
3: -0.6446398, 0.7102579, -0.4736900, 0.6343694, -1.2790092, 1.1839478
4: -0.7812951, 0.7202976, -0.6134740, 0.6030147, -1.3843098, 1.3337716
5: -0.6831884, 0.7839862, -0.5731425, 0.6689765, -1.3521650, 1.3571286
6: -0.6132211, 0.8373356, -0.4829884, 0.7237632, -1.3369843, 1.3203241
7: -0.8100977, 0.8226141, -0.6710814, 0.6871932, -1.4972908, 1.4936955
8: -1.0035493, 1.5676322, -0.7382430, 1.5527720, -2.5563209, 2.3058753
9: -0.6684821, 0.8983816, -0.5604296, 0.7814227, -1.4499048, 1.4588113

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=92, inp2_unstable=74, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6622848, upper bound: 3.5704778
time: 1.75 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5754114, upper bound: 3.5695649
time: 2.06 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.8742264, 0.7670830, -0.8440781, 0.7487532, -1.6229796, 1.6111612
1: -0.6864972, 0.6892030, -0.6637570, 0.6703729, -1.3568701, 1.3529601
2: -0.5923334, 1.3691396, -0.5550326, 1.3367251, -1.9290586, 1.9241722
3: -0.6446398, 0.7102579, -0.6107832, 0.6956552, -1.3402951, 1.3210411
4: -0.7812951, 0.7202976, -0.7491193, 0.6973783, -1.4786735, 1.4694170
5: -0.6831884, 0.7839862, -0.6596397, 0.7621418, -1.4453303, 1.4436258
6: -0.6132211, 0.8373356, -0.5867524, 0.8152532, -1.4284743, 1.4240880
7: -0.8100977, 0.8226141, -0.7816032, 0.7968085, -1.6069062, 1.6042173
8: -1.0035493, 1.5676322, -0.9458506, 1.5565381, -2.5600872, 2.5134826
9: -0.6684821, 0.8983816, -0.6460938, 0.8747255, -1.5432076, 1.5444753

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=92, inp2_unstable=89, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6622849, upper bound: 3.6201733
time: 1.57 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5754114, upper bound: 3.6195517
time: 1.72 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.7094821, 0.6650710, -0.7909331, 0.7203431, -1.4298252, 1.4560040
1: -0.5609788, 0.5800449, -0.6242751, 0.6370211, -1.1979998, 1.2043201
2: -0.4091291, 1.2244315, -0.5060699, 1.3142037, -1.7233329, 1.7305014
3: -0.4555055, 0.6259102, -0.5435931, 0.6705233, -1.1260288, 1.1695032
4: -0.5963171, 0.5904274, -0.6854116, 0.6558771, -1.2521942, 1.2758390
5: -0.5641767, 0.6563267, -0.6197749, 0.7253982, -1.2895749, 1.2761016
6: -0.4730721, 0.7126814, -0.5419029, 0.7780363, -1.2511084, 1.2545843
7: -0.6569301, 0.6723369, -0.7310076, 0.7466441, -1.4035742, 1.4033445
8: -0.7264873, 1.5624757, -0.8818099, 1.5918084, -2.3182957, 2.4442856
9: -0.5516040, 0.7706635, -0.6098512, 0.8395666, -1.3911707, 1.3805147

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=65, inp2_unstable=82, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5188174, upper bound: 3.5768629
time: 1.34 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5188174, upper bound: 3.5768629
time: 1.67 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.6424738, 0.6146749, -0.6122563, 0.5866613, -1.2291350, 1.2269311
1: -0.5073165, 0.5281041, -0.4766731, 0.5007225, -1.0080390, 1.0047772
2: -0.3593611, 1.1514735, -0.3469353, 1.1246632, -1.4840243, 1.4984088
3: -0.3860783, 0.5786694, -0.3514294, 0.5505390, -0.9366173, 0.9300988
4: -0.5239696, 0.5293781, -0.4874117, 0.4951105, -1.0190800, 1.0167898
5: -0.5214326, 0.5922335, -0.4988486, 0.5577812, -1.0792139, 1.0910821
6: -0.4205707, 0.6588503, -0.3965691, 0.6326833, -1.0532540, 1.0554193
7: -0.5879630, 0.6024114, -0.5517539, 0.5585721, -1.1465352, 1.1541654
8: -0.6037260, 1.5587642, -0.5566682, 1.5991105, -2.2011623, 2.1142576
9: -0.4966357, 0.7191773, -0.4700070, 0.6972843, -1.1939200, 1.1891843

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=48, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.9656609, upper bound: 3.2916308
time: 1.24 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.4941025, upper bound: 3.5536203
time: 1.28 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.8742264, 0.7670830, -0.8527289, 0.7571376, -1.6313640, 1.6198120
1: -0.6864972, 0.6892030, -0.6699942, 0.6768299, -1.3633270, 1.3591971
2: -0.5923334, 1.3691396, -0.5738064, 1.3619649, -1.9542984, 1.9429460
3: -0.6446398, 0.7102579, -0.6156490, 0.7002548, -1.3448946, 1.3259070
4: -0.7812951, 0.7202976, -0.7552004, 0.7032928, -1.4845879, 1.4754980
5: -0.6831884, 0.7839862, -0.6652343, 0.7708324, -1.4540207, 1.4492204
6: -0.6132211, 0.8373356, -0.5966308, 0.8226357, -1.4358568, 1.4339664
7: -0.8100977, 0.8226141, -0.7884397, 0.8022155, -1.6123132, 1.6110538
8: -1.0035493, 1.5676322, -0.9797240, 1.5948653, -2.5984147, 2.5473554
9: -0.6684821, 0.8983816, -0.6537904, 0.8855772, -1.5540593, 1.5521719

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=92, inp2_unstable=90, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5762089, upper bound: 3.5786944
time: 1.48 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5762089, upper bound: 3.5786944
time: 1.39 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.7638693, 0.7018578, -0.6435032, 0.6113493, -1.3752186, 1.3453610
1: -0.6042974, 0.6177524, -0.5038793, 0.5263673, -1.1306647, 1.1216316
2: -0.4754443, 1.2918049, -0.3628445, 1.1689740, -1.6444182, 1.6546494
3: -0.5135840, 0.6569063, -0.3816750, 0.5735791, -1.0871631, 1.0385814
4: -0.6547424, 0.6348746, -0.5188925, 0.5241812, -1.1789236, 1.1537671
5: -0.6012453, 0.7022828, -0.5210841, 0.5881500, -1.1893953, 1.2233670
6: -0.5154751, 0.7576230, -0.4210747, 0.6585856, -1.1740607, 1.1786976
7: -0.7079332, 0.7218845, -0.5813544, 0.5934363, -1.3013695, 1.3032389
8: -0.8366102, 1.5637184, -0.6255056, 1.6018834, -2.4367642, 2.1892240
9: -0.5902100, 0.8166108, -0.4967336, 0.7208853, -1.3110952, 1.3133444

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=79, inp2_unstable=50, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.0092818, upper bound: 3.3222819
time: 1.65 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5519410, upper bound: 3.5558012
time: 1.29 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.7912142, 0.7206212, -0.7222407, 0.6742139, -1.4654281, 1.4428620
1: -0.6244653, 0.6372649, -0.5717611, 0.5904015, -1.2148668, 1.2090261
2: -0.5065266, 1.3146231, -0.4184340, 1.2256221, -1.7321486, 1.7330571
3: -0.5438306, 0.6706591, -0.4736900, 0.6343694, -1.1782000, 1.1443491
4: -0.6856831, 0.6560810, -0.6134740, 0.6030147, -1.2886978, 1.2695550
5: -0.6199667, 0.7257146, -0.5731425, 0.6689765, -1.2889432, 1.2988571
6: -0.5422966, 0.7782614, -0.4829884, 0.7237632, -1.2660598, 1.2612498
7: -0.7312356, 0.7468467, -0.6710814, 0.6871932, -1.4184288, 1.4179281
8: -0.8826150, 1.5933522, -0.7382430, 1.5527720, -2.4353871, 2.3315949
9: -0.6101183, 0.8399189, -0.5604296, 0.7814227, -1.3915410, 1.4003484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=82, inp2_unstable=74, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 144

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.3539563, upper bound: 3.4502387
time: 1.40 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6456764, upper bound: 3.5418664
time: 1.92 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.8530254, 0.7574339, -0.8440781, 0.7487532, -1.6017786, 1.6015120
1: -0.6702126, 0.6770992, -0.6637570, 0.6703729, -1.3405855, 1.3408562
2: -0.5743517, 1.3624271, -0.5550326, 1.3367251, -1.9110768, 1.9174597
3: -0.6159139, 0.7004040, -0.6107832, 0.6956552, -1.3115691, 1.3111873
4: -0.7554982, 0.7035162, -0.7491193, 0.6973783, -1.4528766, 1.4526355
5: -0.6654902, 0.7711605, -0.6596397, 0.7621418, -1.4276320, 1.4308002
6: -0.5970553, 0.8228754, -0.5867524, 0.8152532, -1.4123085, 1.4096278
7: -0.7886799, 0.8024640, -0.7816032, 0.7968085, -1.5854884, 1.5840671
8: -0.9806290, 1.5964582, -0.9458506, 1.5565381, -2.5371671, 2.5423088
9: -0.6540712, 0.8859549, -0.6460938, 0.8747255, -1.5287967, 1.5320487

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=90, inp2_unstable=89, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 144

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6482510, upper bound: 3.6175905
time: 1.63 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6482510, upper bound: 3.6204706
time: 1.62 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.6127387, 0.5871146, -0.6516190, 0.6240643, -1.2368031, 1.2387335
1: -0.4769712, 0.5011066, -0.5168293, 0.5368063, -1.0137775, 1.0179359
2: -0.3474104, 1.1251478, -0.3630740, 1.1548004, -1.5022109, 1.4882218
3: -0.3517099, 0.5507427, -0.3967934, 0.5880464, -0.9397564, 0.9475361
4: -0.4877988, 0.4954478, -0.5367474, 0.5402665, -1.0280652, 1.0321951
5: -0.4992151, 0.5583050, -0.5282872, 0.6039501, -1.1031651, 1.0865922
6: -0.3971912, 0.6330128, -0.4285384, 0.6665362, -1.0637274, 1.0615512
7: -0.5520501, 0.5588120, -0.5999638, 0.6171570, -1.1692071, 1.1587758
8: -0.5576983, 1.6029645, -0.6140552, 1.5491552, -2.1057253, 2.2154894
9: -0.4705511, 0.6978772, -0.5054486, 0.7261571, -1.1967082, 1.2033257

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=50, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 144

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.0077403, upper bound: 3.2849560
time: 1.44 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5549750, upper bound: 3.5459382
time: 1.33 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.6440119, 0.6118391, -0.7430959, 0.6879959, -1.3320076, 1.3549349
1: -0.5041932, 0.5267908, -0.5879996, 0.6040083, -1.1082015, 1.1147904
2: -0.3633332, 1.1695341, -0.4489284, 1.2633713, -1.6267046, 1.6184624
3: -0.3819892, 0.5737923, -0.4928470, 0.6455711, -1.0275602, 1.0666393
4: -0.5193256, 0.5245317, -0.6329191, 0.6187740, -1.1380996, 1.1574509
5: -0.5214867, 0.5886931, -0.5869537, 0.6854190, -1.2069056, 1.1756468
6: -0.4217668, 0.6589385, -0.4985793, 0.7408842, -1.1626511, 1.1575177
7: -0.5816638, 0.5936893, -0.6896145, 0.7039613, -1.2856250, 1.2833037
8: -0.6266382, 1.6057832, -0.7915496, 1.5527903, -2.1794286, 2.3958004
9: -0.4973323, 0.7214990, -0.5756978, 0.7988176, -1.2961500, 1.2971969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=76, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 144

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5232981, upper bound: 3.6130399
time: 1.52 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5232981, upper bound: 3.6199407
time: 1.47 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.8530254, 0.7574339, -0.8527289, 0.7571376, -1.6101630, 1.6101627
1: -0.6702126, 0.6770992, -0.6699942, 0.6768299, -1.3470426, 1.3470933
2: -0.5743517, 1.3624271, -0.5738064, 1.3619649, -1.9363166, 1.9362335
3: -0.6159139, 0.7004040, -0.6156490, 0.7002548, -1.3161687, 1.3160532
4: -0.7554982, 0.7035162, -0.7552004, 0.7032928, -1.4587910, 1.4587166
5: -0.6654902, 0.7711605, -0.6652343, 0.7708324, -1.4363225, 1.4363948
6: -0.5970553, 0.8228754, -0.5966308, 0.8226357, -1.4196908, 1.4195061
7: -0.7886799, 0.8024640, -0.7884397, 0.8022155, -1.5908954, 1.5909038
8: -0.9806290, 1.5964582, -0.9797240, 1.5948653, -2.5754943, 2.5761819
9: -0.6540712, 0.8859549, -0.6537904, 0.8855772, -1.5396484, 1.5397452

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=90, inp2_unstable=90, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 144

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6456185, upper bound: 3.5780524
time: 1.79 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6733205, upper bound: 3.5793142
time: 1.83 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.8530254, 0.7574339, -0.6435032, 0.6113493, -1.4643748, 1.4009371
1: -0.6702126, 0.6770992, -0.5038793, 0.5263673, -1.1965799, 1.1809785
2: -0.5743517, 1.3624271, -0.3628445, 1.1689740, -1.7433257, 1.7252716
3: -0.6159139, 0.7004040, -0.3816750, 0.5735791, -1.1894929, 1.0820791
4: -0.7554982, 0.7035162, -0.5188925, 0.5241812, -1.2796794, 1.2224088
5: -0.6654902, 0.7711605, -0.5210841, 0.5881500, -1.2536402, 1.2922447
6: -0.5970553, 0.8228754, -0.4210747, 0.6585856, -1.2556409, 1.2439500
7: -0.7886799, 0.8024640, -0.5813544, 0.5934363, -1.3821162, 1.3838184
8: -0.9806290, 1.5964582, -0.6255056, 1.6018834, -2.5808301, 2.2219639
9: -0.6540712, 0.8859549, -0.4967336, 0.7208853, -1.3749565, 1.3826884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=90, inp2_unstable=50, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 144

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6456185, upper bound: 3.5780524
time: 3.70 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6733205, upper bound: 3.5793142
time: 6.72 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 12.03 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5716330, upper bound: 3.6192978
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5716330, upper bound: 3.6192978
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.0586180, upper bound: 3.3856956
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5467674, upper bound: 3.5946240
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6856204, upper bound: 3.5718975
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6192977, upper bound: 3.5716330
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6856204, upper bound: 3.6203546
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6192978, upper bound: 3.6201390
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5717673, upper bound: 3.6205211
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5717673, upper bound: 3.6205211
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.0586675, upper bound: 3.3856956
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5468984, upper bound: 3.5964007
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6858633, upper bound: 3.5724056
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6194399, upper bound: 3.5721306
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6858634, upper bound: 3.6216135
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6194399, upper bound: 3.6213871
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5721306, upper bound: 3.6194399
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5721306, upper bound: 3.6194399
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.0584058, upper bound: 3.3820500
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5478709, upper bound: 3.5947675
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6905329, upper bound: 3.5720452
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6205211, upper bound: 3.5717673
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6905329, upper bound: 3.6205025
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6205211, upper bound: 3.6202749
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5722074, upper bound: 3.6207231
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5722074, upper bound: 3.6207231
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.0584058, upper bound: 3.3820500
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5480089, upper bound: 3.5966521
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6922415, upper bound: 3.5725899
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6206140, upper bound: 3.5723188
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6922415, upper bound: 3.6217940
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6206140, upper bound: 3.6215576
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5693393, upper bound: 3.5753002
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5693393, upper bound: 3.5753002
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.0545243, upper bound: 3.3253714
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5444685, upper bound: 3.5507837
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6695118, upper bound: 3.5193911
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6114792, upper bound: 3.5190783
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6695118, upper bound: 3.5767451
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6114792, upper bound: 3.5764608
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5696654, upper bound: 3.5778689
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5696654, upper bound: 3.5778689
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.0548981, upper bound: 3.3255452
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5448181, upper bound: 3.5546708
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6186636, upper bound: 3.5790925
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6186636, upper bound: 3.5790925
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.0789234, upper bound: 3.3462310
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5940237, upper bound: 3.5562055
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5695649, upper bound: 3.5754114
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5695649, upper bound: 3.5754114
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.0537355, upper bound: 3.3229289
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5453498, upper bound: 3.5509021
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6721805, upper bound: 3.5195519
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6117901, upper bound: 3.5192074
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6721805, upper bound: 3.5768996
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6117902, upper bound: 3.5765855
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5697800, upper bound: 3.5780980
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5697800, upper bound: 3.5780980
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.0537355, upper bound: 3.3229289
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5457576, upper bound: 3.5549962
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6197358, upper bound: 3.5793037
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6197358, upper bound: 3.5793037
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.2507201, upper bound: 3.4673341
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5882571, upper bound: 3.5474993
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5190783, upper bound: 3.6114792
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5190783, upper bound: 3.6114792
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 12.03
Output dim: 8, lower bound: -2.9660847, upper bound: 3.3045510
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.4943521, upper bound: 3.5867418
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6619132, upper bound: 3.5701044
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5753002, upper bound: 3.5693393
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6619132, upper bound: 3.6189651
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5753002, upper bound: 3.6183980
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5187074, upper bound: 3.5743171
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5187074, upper bound: 3.5743173
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 12.03
Output dim: 8, lower bound: -2.9655738, upper bound: 3.2916273
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.4939823, upper bound: 3.5497608
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6574076, upper bound: 3.5190690
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5743172, upper bound: 3.5187074
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6574076, upper bound: 3.5764382
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5743173, upper bound: 3.5760984
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.3477684, upper bound: 3.4461180
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6439715, upper bound: 3.5409959
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6446634, upper bound: 3.6162584
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6446634, upper bound: 3.6191821
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.0077403, upper bound: 3.2849560
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5546707, upper bound: 3.5448182
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5231919, upper bound: 3.6119127
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5231919, upper bound: 3.6186637
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.3317755, upper bound: 3.4201425
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6377287, upper bound: 3.4889880
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6413911, upper bound: 3.5753957
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6413911, upper bound: 3.5766225
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.0051862, upper bound: 3.2657957
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5536203, upper bound: 3.4941026
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5228240, upper bound: 3.5744655
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5228240, upper bound: 3.5762089
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5192074, upper bound: 3.6117902
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5192074, upper bound: 3.6117902
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 12.03
Output dim: 8, lower bound: -2.9661738, upper bound: 3.3045638
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.4944868, upper bound: 3.5877776
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6622848, upper bound: 3.5704778
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5754114, upper bound: 3.5695649
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6622849, upper bound: 3.6201733
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5754114, upper bound: 3.6195517
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5188174, upper bound: 3.5768629
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5188174, upper bound: 3.5768629
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 12.03
Output dim: 8, lower bound: -2.9656609, upper bound: 3.2916308
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.4941025, upper bound: 3.5536203
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5762089, upper bound: 3.5786944
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5762089, upper bound: 3.5786944
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.0092818, upper bound: 3.3222819
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5519410, upper bound: 3.5558012
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.3539563, upper bound: 3.4502387
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6456764, upper bound: 3.5418664
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6482510, upper bound: 3.6175905
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6482510, upper bound: 3.6204706
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.0077403, upper bound: 3.2849560
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5549750, upper bound: 3.5459382
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5232981, upper bound: 3.6130399
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.5232981, upper bound: 3.6199407
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6456185, upper bound: 3.5780524
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6733205, upper bound: 3.5793142
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6456185, upper bound: 3.5780524
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 12.03
Output dim: 8, lower bound: -3.6733205, upper bound: 3.5793142
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 12.03
Output dim: 8, lower bound: -3.5771008, upper bound: 3.5230059
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 12.03
Output dim: 8, lower bound: -3.5788344, upper bound: 3.5788898

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 5.36 + 599.04 = 604.40 seconds
