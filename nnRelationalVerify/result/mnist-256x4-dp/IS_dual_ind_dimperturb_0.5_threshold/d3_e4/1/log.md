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
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=54, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0624631, 0.0465500, -0.0624631, 0.0465500, -0.1090131, 0.1090131)
1: (-0.0575841, 0.0544078, -0.0575841, 0.0544078, -0.1119918, 0.1119918)
2: (-0.0618003, 0.1491337, -0.0618003, 0.1491337, -0.2109340, 0.2109340)
3: (-0.0388912, 0.0644384, -0.0388912, 0.0644384, -0.1033296, 0.1033296)
4: (-0.0695920, 0.0709374, -0.0695920, 0.0709374, -0.1405294, 0.1405294)
5: (-0.0448468, 0.0806352, -0.0448468, 0.0806352, -0.1254819, 0.1254819)
6: (-0.1138270, 0.0848912, -0.1138270, 0.0848912, -0.1987182, 0.1987182)
7: (0.8231879, 1.0149322, 0.8231879, 1.0149322, -0.1917442, 0.1917442)
8: (-0.0986703, 0.1093244, -0.0986703, 0.1093244, -0.2079947, 0.2079947)
9: (-0.0866429, 0.1129852, -0.0866429, 0.1129852, -0.1996281, 0.1996281)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.74 + 2.03 = 3.78 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.80 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.78 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.78 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.78
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.78
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0543049, 0.0423527, -0.0611331, 0.0458657, -0.1001706, 0.1034858
1: -0.0494341, 0.0477899, -0.0562554, 0.0533289, -0.1027629, 0.1040453
2: -0.0529187, 0.1390919, -0.0603523, 0.1474966, -0.2004153, 0.1994442
3: -0.0311730, 0.0576337, -0.0376329, 0.0633291, -0.0945021, 0.0952667
4: -0.0614679, 0.0607440, -0.0682676, 0.0692756, -0.1307435, 0.1290116
5: -0.0365674, 0.0714648, -0.0434970, 0.0791402, -0.1157076, 0.1149618
6: -0.1043677, 0.0759142, -0.1122849, 0.0834277, -0.1877954, 0.1881991
7: 0.8361411, 1.0101410, 0.8252998, 1.0141512, -0.1780101, 0.1848412
8: -0.0899547, 0.0985287, -0.0972494, 0.1075644, -0.1975191, 0.1957781
9: -0.0800795, 0.0963693, -0.0855729, 0.1102763, -0.1903558, 0.1819422

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=54, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 255

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.89 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.85 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.1112903, 0.0716702, -0.0577291, 0.0441144, -0.1554047, 0.1293993
1: -0.1063616, 0.0940159, -0.0528549, 0.0505675, -0.1569291, 0.1468708
2: -0.1149564, 0.2092329, -0.0566465, 0.1433067, -0.2582630, 0.2658795
3: -0.0850842, 0.1051642, -0.0344126, 0.0604898, -0.1455740, 0.1395768
4: -0.1182146, 0.1319446, -0.0648778, 0.0650224, -0.1832370, 0.1968224
5: -0.0943986, 0.1355196, -0.0400425, 0.0753138, -0.1697125, 0.1755621
6: -0.1704405, 0.1386180, -0.1083380, 0.0796821, -0.2501226, 0.2469560
7: 0.7456647, 1.0436066, 0.8307044, 1.0121520, -0.2664872, 0.2129022
8: -0.1508328, 0.1739358, -0.0936129, 0.1030599, -0.2538927, 0.2675487
9: -0.1259251, 0.2124307, -0.0828343, 0.1033434, -0.2292685, 0.2952650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=54, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.87 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.81 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.44 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.44
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.44
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.44
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.44
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0543049, 0.0423527, -0.0543049, 0.0423527, -0.0966576, 0.0966576
1: -0.0494341, 0.0477899, -0.0494341, 0.0477899, -0.0972239, 0.0972239
2: -0.0529187, 0.1390919, -0.0529187, 0.1390919, -0.1920106, 0.1920106
3: -0.0311730, 0.0576337, -0.0311730, 0.0576337, -0.0888068, 0.0888068
4: -0.0614679, 0.0607440, -0.0614679, 0.0607440, -0.1222119, 0.1222119
5: -0.0365674, 0.0714648, -0.0365674, 0.0714648, -0.1080322, 0.1080322
6: -0.1043677, 0.0759142, -0.1043677, 0.0759142, -0.1802819, 0.1802819
7: 0.8361411, 1.0101410, 0.8361411, 1.0101410, -0.1739999, 0.1739999
8: -0.0899547, 0.0985287, -0.0899547, 0.0985287, -0.1884834, 0.1884834
9: -0.0800795, 0.0963693, -0.0800795, 0.0963693, -0.1764488, 0.1764488

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.90 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0543049, 0.0423527, -0.1112903, 0.0716702, -0.1259751, 0.1536430
1: -0.0494341, 0.0477899, -0.1063616, 0.0940159, -0.1434500, 0.1541514
2: -0.0529187, 0.1390919, -0.1149564, 0.2092329, -0.2621516, 0.2540483
3: -0.0311730, 0.0576337, -0.0850842, 0.1051642, -0.1363373, 0.1427179
4: -0.0614679, 0.0607440, -0.1182146, 0.1319446, -0.1934125, 0.1789586
5: -0.0365674, 0.0714648, -0.0943986, 0.1355196, -0.1720870, 0.1658634
6: -0.1043677, 0.0759142, -0.1704405, 0.1386180, -0.2429857, 0.2463548
7: 0.8361411, 1.0101410, 0.7456647, 1.0436066, -0.2074655, 0.2644763
8: -0.0899547, 0.0985287, -0.1508328, 0.1739358, -0.2638905, 0.2493615
9: -0.0800795, 0.0963693, -0.1259251, 0.2124307, -0.2925102, 0.2222943

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 91

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.92 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.1101669, 0.0710923, -0.0533778, 0.0418758, -0.1520426, 0.1244701
1: -0.1052393, 0.0931047, -0.0485079, 0.0470378, -0.1522771, 0.1416127
2: -0.1137335, 0.2078502, -0.0519094, 0.1379508, -0.2516842, 0.2597596
3: -0.0840214, 0.1042273, -0.0302959, 0.0568605, -0.1408819, 0.1345232
4: -0.1170959, 0.1305410, -0.0605447, 0.0595857, -0.1766816, 0.1910856
5: -0.0932585, 0.1342569, -0.0356266, 0.0704227, -0.1636812, 0.1698835
6: -0.1691381, 0.1373819, -0.1032928, 0.0748941, -0.2440321, 0.2406747
7: 0.7474484, 1.0429469, 0.8376130, 1.0095966, -0.2621482, 0.2053339
8: -0.1496327, 0.1724493, -0.0889642, 0.0973020, -0.2469347, 0.2614135
9: -0.1250213, 0.2101426, -0.0793336, 0.0944811, -0.2195024, 0.2894763

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 75

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.92 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.1084047, 0.0701857, -0.0777329, 0.0544059, -0.1628105, 0.1479186
1: -0.1034789, 0.0916752, -0.0728383, 0.0667945, -0.1702734, 0.1645135
2: -0.1118150, 0.2056812, -0.0784239, 0.1679286, -0.2797436, 0.2841051
3: -0.0823543, 0.1027574, -0.0533372, 0.0771747, -0.1595290, 0.1560946
4: -0.1153411, 0.1283392, -0.0847978, 0.0900163, -0.2053574, 0.2131370
5: -0.0914702, 0.1322761, -0.0603432, 0.0977993, -0.1892695, 0.1926193
6: -0.1670948, 0.1354429, -0.1315319, 0.1016933, -0.2687881, 0.2669747
7: 0.7502462, 1.0419121, 0.7989441, 1.0238998, -0.2736536, 0.2429680
8: -0.1477501, 0.1701174, -0.1149832, 0.1295304, -0.2772805, 0.2851006
9: -0.1236036, 0.2065537, -0.0989277, 0.1440850, -0.2676886, 0.3054814

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 75

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.95 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.67 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0499664, 0.0401207, -0.0532180, 0.0417936, -0.0917601, 0.0933387
1: -0.0451001, 0.0442705, -0.0483483, 0.0469082, -0.0920083, 0.0926189
2: -0.0481956, 0.1337520, -0.0517355, 0.1377542, -0.1859498, 0.1854874
3: -0.0270686, 0.0540152, -0.0301448, 0.0567272, -0.0837959, 0.0841600
4: -0.0571476, 0.0553234, -0.0603857, 0.0593860, -0.1165337, 0.1157090
5: -0.0321646, 0.0665881, -0.0354645, 0.0702431, -0.1024077, 0.1020526
6: -0.0993374, 0.0711405, -0.1031075, 0.0747183, -0.1740557, 0.1742480
7: 0.8430293, 1.0075935, 0.8378666, 1.0095028, -0.1664735, 0.1697269
8: -0.0853199, 0.0927878, -0.0887937, 0.0970906, -0.1824105, 0.1815815
9: -0.0765891, 0.0875333, -0.0792051, 0.0941558, -0.1707449, 0.1667384

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0743118, 0.0526458, -0.0515389, 0.0409297, -0.1152415, 0.1041847
1: -0.0694207, 0.0640194, -0.0466709, 0.0455461, -0.1149668, 0.1106902
2: -0.0746995, 0.1637177, -0.0499074, 0.1356874, -0.2103869, 0.2136252
3: -0.0501007, 0.0743212, -0.0285563, 0.0553267, -0.1054274, 0.1028775
4: -0.0813911, 0.0857418, -0.0587135, 0.0572880, -0.1386792, 0.1444553
5: -0.0568714, 0.0939538, -0.0337604, 0.0683556, -0.1252270, 0.1277142
6: -0.1275652, 0.0979289, -0.1011606, 0.0728707, -0.2004359, 0.1990895
7: 0.8043758, 1.0218906, 0.8405327, 1.0085169, -0.2041411, 0.1813580
8: -0.1113284, 0.1250034, -0.0869998, 0.0948686, -0.2061970, 0.2120032
9: -0.0961754, 0.1371173, -0.0778542, 0.0907358, -0.1869112, 0.2149714

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0499664, 0.0401207, -0.1101669, 0.0710923, -0.1210587, 0.1502876
1: -0.0451001, 0.0442705, -0.1052393, 0.0931047, -0.1382048, 0.1495098
2: -0.0481956, 0.1337520, -0.1137335, 0.2078502, -0.2560458, 0.2474854
3: -0.0270686, 0.0540152, -0.0840214, 0.1042273, -0.1312959, 0.1380366
4: -0.0571476, 0.0553234, -0.1170959, 0.1305410, -0.1876886, 0.1724193
5: -0.0321646, 0.0665881, -0.0932585, 0.1342569, -0.1664215, 0.1598466
6: -0.0993374, 0.0711405, -0.1691381, 0.1373819, -0.2367193, 0.2402785
7: 0.8430293, 1.0075935, 0.7474484, 1.0429469, -0.1999177, 0.2601451
8: -0.0853199, 0.0927878, -0.1496327, 0.1724493, -0.2577692, 0.2424205
9: -0.0765891, 0.0875333, -0.1250213, 0.2101426, -0.2867317, 0.2125546

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=29, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 75

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0743118, 0.0526458, -0.1084047, 0.0701857, -0.1444975, 0.1610505
1: -0.0694207, 0.0640194, -0.1034789, 0.0916752, -0.1610959, 0.1674982
2: -0.0746995, 0.1637177, -0.1118150, 0.2056812, -0.2803807, 0.2755327
3: -0.0501007, 0.0743212, -0.0823543, 0.1027574, -0.1528581, 0.1566755
4: -0.0813911, 0.0857418, -0.1153411, 0.1283392, -0.2097303, 0.2010829
5: -0.0568714, 0.0939538, -0.0914702, 0.1322761, -0.1891474, 0.1854240
6: -0.1275652, 0.0979289, -0.1670948, 0.1354429, -0.2630081, 0.2650237
7: 0.8043758, 1.0218906, 0.7502462, 1.0419121, -0.2375363, 0.2716444
8: -0.1113284, 0.1250034, -0.1477501, 0.1701174, -0.2814458, 0.2727535
9: -0.0961754, 0.1371173, -0.1236036, 0.2065537, -0.3027290, 0.2607209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=29, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 75

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1271883, 0.0798494, -0.0487076, 0.0394611, -0.1666494, 0.1285571
1: -0.1222434, 0.1069124, -0.0438388, 0.0432305, -0.1654739, 0.1507512
2: -0.1322640, 0.2288012, -0.0468407, 0.1321738, -0.2644378, 0.2756419
3: -0.1001246, 0.1184246, -0.0259384, 0.0529457, -0.1530703, 0.1443630
4: -0.1340461, 0.1518085, -0.0558709, 0.0537969, -0.1878430, 0.2076794
5: -0.1105326, 0.1533900, -0.0308635, 0.0652413, -0.1757739, 0.1842534
6: -0.1888739, 0.1561115, -0.0979231, 0.0697296, -0.2586035, 0.2540346
7: 0.7204231, 1.0529430, 0.8450650, 1.0068694, -0.2864463, 0.2078780
8: -0.1678169, 0.1949732, -0.0839502, 0.0912238, -0.2590407, 0.2789235
9: -0.1387154, 0.2448101, -0.0755729, 0.0849220, -0.2236374, 0.3203830

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=17, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.02 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1084628, 0.0702156, -0.0533778, 0.0418758, -0.1503386, 0.1235934
1: -0.1035370, 0.0917223, -0.0485079, 0.0470378, -0.1505747, 0.1402303
2: -0.1118782, 0.2057528, -0.0519094, 0.1379508, -0.2498290, 0.2576622
3: -0.0824093, 0.1028060, -0.0302959, 0.0568605, -0.1392697, 0.1331019
4: -0.1153990, 0.1284118, -0.0605447, 0.0595857, -0.1749847, 0.1889565
5: -0.0915292, 0.1323414, -0.0356266, 0.0704227, -0.1619519, 0.1679680
6: -0.1671621, 0.1355069, -0.1032928, 0.0748941, -0.2420562, 0.2387997
7: 0.7501539, 1.0419461, 0.8376130, 1.0095966, -0.2594427, 0.2043330
8: -0.1478122, 0.1701943, -0.0889642, 0.0973020, -0.2451142, 0.2591585
9: -0.1236504, 0.2066720, -0.0793336, 0.0944811, -0.2181314, 0.2860056

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1253263, 0.0788914, -0.0731021, 0.0520234, -0.1773498, 0.1519935
1: -0.1203833, 0.1054019, -0.0682122, 0.0630380, -0.1834214, 0.1736141
2: -0.1302369, 0.2265094, -0.0733825, 0.1622287, -0.2924656, 0.2998918
3: -0.0983630, 0.1168715, -0.0489562, 0.0733122, -0.1716752, 0.1658277
4: -0.1321918, 0.1494820, -0.0801864, 0.0842303, -0.2164221, 0.2296684
5: -0.1086430, 0.1512970, -0.0556437, 0.0925939, -0.2012369, 0.2069407
6: -0.1867149, 0.1540626, -0.1261625, 0.0965978, -0.2833127, 0.2802251
7: 0.7233796, 1.0518495, 0.8062965, 1.0211799, -0.2978003, 0.2455530
8: -0.1658277, 0.1925093, -0.1100360, 0.1234026, -0.2892303, 0.3025453
9: -0.1372173, 0.2410178, -0.0952021, 0.1346534, -0.2718707, 0.3362199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.10 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1067123, 0.0693150, -0.0777329, 0.0544059, -0.1611181, 0.1470479
1: -0.1017882, 0.0903024, -0.0728383, 0.0667945, -0.1685828, 0.1631407
2: -0.1099726, 0.2035982, -0.0784239, 0.1679286, -0.2779012, 0.2820221
3: -0.0807533, 0.1013459, -0.0533372, 0.0771747, -0.1579279, 0.1546831
4: -0.1136558, 0.1262247, -0.0847978, 0.0900163, -0.2036721, 0.2110225
5: -0.0897527, 0.1303737, -0.0603432, 0.0977993, -0.1875520, 0.1907170
6: -0.1651326, 0.1335807, -0.1315319, 0.1016933, -0.2668259, 0.2651125
7: 0.7529331, 1.0409182, 0.7989441, 1.0238998, -0.2709666, 0.2419741
8: -0.1459421, 0.1678780, -0.1149832, 0.1295304, -0.2754725, 0.2828612
9: -0.1222421, 0.2031068, -0.0989277, 0.1440850, -0.2663271, 0.3020346

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.93 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.71 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.71
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.71
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.71
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.71
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.71
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.71
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.71
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.71
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.71
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.71
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.71
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.71
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.71
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.71
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.71
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.71
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0455947, 0.0381972, -0.0697847, 0.0503168, -0.0959115, 0.1079819
1: -0.0410053, 0.0405350, -0.0648982, 0.0603470, -0.1013523, 0.1054333
2: -0.0435914, 0.1285242, -0.0697711, 0.1581455, -0.2017369, 0.1982952
3: -0.0240294, 0.0501743, -0.0458178, 0.0705453, -0.0945746, 0.0959921
4: -0.0535419, 0.0503250, -0.0768829, 0.0800854, -0.1336273, 0.1272079
5: -0.0277718, 0.0623557, -0.0522771, 0.0888651, -0.1166369, 0.1146328
6: -0.0947213, 0.0664938, -0.1223161, 0.0929475, -0.1876688, 0.1888099
7: 0.8497430, 1.0051798, 0.8115636, 1.0192318, -0.1694888, 0.1936162
8: -0.0814885, 0.0880208, -0.1064920, 0.1190128, -0.2005013, 0.1945128
9: -0.0730368, 0.0802459, -0.0925333, 0.1278970, -0.2009338, 0.1727792

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 255

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0499664, 0.0401207, -0.0516121, 0.0409674, -0.0909338, 0.0917329
1: -0.0451001, 0.0442705, -0.0467441, 0.0456055, -0.0907056, 0.0910146
2: -0.0481956, 0.1337520, -0.0499873, 0.1357776, -0.1839732, 0.1837392
3: -0.0270686, 0.0540152, -0.0286256, 0.0553878, -0.0824564, 0.0826407
4: -0.0571476, 0.0553234, -0.0587865, 0.0573796, -0.1145273, 0.1141098
5: -0.0321646, 0.0665881, -0.0338348, 0.0684380, -0.1006026, 0.1004229
6: -0.0993374, 0.0711405, -0.1012455, 0.0729513, -0.1722887, 0.1723860
7: 0.8430293, 1.0075935, 0.8404164, 1.0085597, -0.1655304, 0.1671771
8: -0.0853199, 0.0927878, -0.0870781, 0.0949656, -0.1802855, 0.1798659
9: -0.0765891, 0.0875333, -0.0779131, 0.0908851, -0.1674742, 0.1654464

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 33

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0696582, 0.0502516, -0.0679497, 0.0493726, -0.1190308, 0.1182013
1: -0.0647718, 0.0602443, -0.0630650, 0.0588584, -0.1236302, 0.1233093
2: -0.0696332, 0.1579897, -0.0677732, 0.1558868, -0.2255200, 0.2257629
3: -0.0456981, 0.0704396, -0.0440817, 0.0690147, -0.1147127, 0.1145213
4: -0.0767568, 0.0799273, -0.0750556, 0.0777926, -0.1545494, 0.1549829
5: -0.0521486, 0.0887228, -0.0504148, 0.0868023, -0.1389510, 0.1391376
6: -0.1221694, 0.0928082, -0.1201885, 0.0909283, -0.2130976, 0.2129966
7: 0.8117645, 1.0191574, 0.8144772, 1.0181543, -0.2063897, 0.2046802
8: -0.1063568, 0.1188453, -0.1045316, 0.1165845, -0.2229413, 0.2233770
9: -0.0924314, 0.1276392, -0.0910569, 0.1241595, -0.2165909, 0.2186961

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0743118, 0.0526458, -0.0499465, 0.0401105, -0.1144223, 0.1025923
1: -0.0694207, 0.0640194, -0.0450802, 0.0442543, -0.1136751, 0.1090995
2: -0.0746995, 0.1637177, -0.0481739, 0.1337274, -0.2084269, 0.2118916
3: -0.0501007, 0.0743212, -0.0270498, 0.0539985, -0.1040992, 0.1013710
4: -0.0813911, 0.0857418, -0.0571277, 0.0552985, -0.1366896, 0.1428695
5: -0.0568714, 0.0939538, -0.0321444, 0.0665657, -0.1234371, 0.1260982
6: -0.1275652, 0.0979289, -0.0993143, 0.0711185, -0.1986837, 0.1972432
7: 0.8043758, 1.0218906, 0.8430610, 1.0075817, -0.2032059, 0.1788297
8: -0.1113284, 0.1250034, -0.0852986, 0.0927615, -0.2040899, 0.2103020
9: -0.0961754, 0.1371173, -0.0765731, 0.0874926, -0.1836680, 0.2136904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 255

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0455947, 0.0381972, -0.1271883, 0.0798494, -0.1254441, 0.1653855
1: -0.0410053, 0.0405350, -0.1222434, 0.1069124, -0.1479177, 0.1627785
2: -0.0435914, 0.1285242, -0.1322640, 0.2288012, -0.2723927, 0.2607882
3: -0.0240294, 0.0501743, -0.1001246, 0.1184246, -0.1424540, 0.1502988
4: -0.0535419, 0.0503250, -0.1340461, 0.1518085, -0.2053504, 0.1843710
5: -0.0277718, 0.0623557, -0.1105326, 0.1533900, -0.1811618, 0.1728883
6: -0.0947213, 0.0664938, -0.1888739, 0.1561115, -0.2508328, 0.2553677
7: 0.8497430, 1.0051798, 0.7204231, 1.0529430, -0.2032000, 0.2847567
8: -0.0814885, 0.0880208, -0.1678169, 0.1949732, -0.2764617, 0.2558377
9: -0.0730368, 0.0802459, -0.1387154, 0.2448101, -0.3178469, 0.2189613

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=56, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.26 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0499664, 0.0401207, -0.1084628, 0.0702156, -0.1201820, 0.1485835
1: -0.0451001, 0.0442705, -0.1035370, 0.0917223, -0.1368224, 0.1478075
2: -0.0481956, 0.1337520, -0.1118782, 0.2057528, -0.2539484, 0.2456302
3: -0.0270686, 0.0540152, -0.0824093, 0.1028060, -0.1298746, 0.1364244
4: -0.0571476, 0.0553234, -0.1153990, 0.1284118, -0.1855595, 0.1707224
5: -0.0321646, 0.0665881, -0.0915292, 0.1323414, -0.1645060, 0.1581173
6: -0.0993374, 0.0711405, -0.1671621, 0.1355069, -0.2348443, 0.2383026
7: 0.8430293, 1.0075935, 0.7501539, 1.0419461, -0.1989168, 0.2574396
8: -0.0853199, 0.0927878, -0.1478122, 0.1701943, -0.2555142, 0.2406000
9: -0.0765891, 0.0875333, -0.1236504, 0.2066720, -0.2832611, 0.2111836

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0696582, 0.0502516, -0.1253263, 0.0788914, -0.1485496, 0.1755779
1: -0.0647718, 0.0602443, -0.1203833, 0.1054019, -0.1701737, 0.1806276
2: -0.0696332, 0.1579897, -0.1302369, 0.2265094, -0.2961426, 0.2882266
3: -0.0456981, 0.0704396, -0.0983630, 0.1168715, -0.1625696, 0.1688026
4: -0.0767568, 0.0799273, -0.1321918, 0.1494820, -0.2262389, 0.2121192
5: -0.0521486, 0.0887228, -0.1086430, 0.1512970, -0.2034456, 0.1973658
6: -0.1221694, 0.0928082, -0.1867149, 0.1540626, -0.2762320, 0.2795231
7: 0.8117645, 1.0191574, 0.7233796, 1.0518495, -0.2400849, 0.2957778
8: -0.1063568, 0.1188453, -0.1658277, 0.1925093, -0.2988661, 0.2846730
9: -0.0924314, 0.1276392, -0.1372173, 0.2410178, -0.3334492, 0.2648565

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0743118, 0.0526458, -0.1067123, 0.0693150, -0.1436268, 0.1593581
1: -0.0694207, 0.0640194, -0.1017882, 0.0903024, -0.1597231, 0.1658076
2: -0.0746995, 0.1637177, -0.1099726, 0.2035982, -0.2782977, 0.2736903
3: -0.0501007, 0.0743212, -0.0807533, 0.1013459, -0.1514466, 0.1550745
4: -0.0813911, 0.0857418, -0.1136558, 0.1262247, -0.2076158, 0.1993976
5: -0.0568714, 0.0939538, -0.0897527, 0.1303737, -0.1872451, 0.1837065
6: -0.1275652, 0.0979289, -0.1651326, 0.1335807, -0.2611459, 0.2630615
7: 0.8043758, 1.0218906, 0.7529331, 1.0409182, -0.2365425, 0.2689575
8: -0.1113284, 0.1250034, -0.1459421, 0.1678780, -0.2792064, 0.2709455
9: -0.0961754, 0.1371173, -0.1222421, 0.2031068, -0.2992822, 0.2593593

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1183239, 0.0752888, -0.0363444, 0.0338890, -0.1522130, 0.1116332
1: -0.1133880, 0.0997216, -0.0340615, 0.0312103, -0.1445983, 0.1337831
2: -0.1226135, 0.2178904, -0.0324034, 0.1167947, -0.2394082, 0.2502938
3: -0.0917384, 0.1110309, -0.0180567, 0.0441950, -0.1359335, 0.1290876
4: -0.1252188, 0.1407328, -0.0459557, 0.0403727, -0.1655914, 0.1866885
5: -0.1015366, 0.1434259, -0.0201936, 0.0523730, -0.1539096, 0.1636194
6: -0.1785958, 0.1463576, -0.0845240, 0.0561946, -0.2347904, 0.2308816
7: 0.7344973, 1.0477374, 0.8657830, 1.0001640, -0.2656667, 0.1819544
8: -0.1583470, 0.1832431, -0.0733035, 0.0778389, -0.2361858, 0.2565466
9: -0.1315838, 0.2267559, -0.0648593, 0.0661383, -0.1977221, 0.2916152

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 75

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 91

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1252390, 0.0788465, -0.0418055, 0.0366356, -0.1618746, 0.1206519
1: -0.1202961, 0.1053311, -0.0377808, 0.0371550, -0.1574511, 0.1431119
2: -0.1301418, 0.2264019, -0.0395256, 0.1239706, -0.2541124, 0.2659275
3: -0.0982804, 0.1167987, -0.0216612, 0.0472905, -0.1455709, 0.1384599
4: -0.1321049, 0.1493729, -0.0507267, 0.0459712, -0.1780760, 0.2000996
5: -0.1085544, 0.1511989, -0.0241145, 0.0587371, -0.1672915, 0.1753134
6: -0.1866137, 0.1539665, -0.0907062, 0.0625949, -0.2492086, 0.2446727
7: 0.7235181, 1.0517981, 0.8555781, 1.0031972, -0.2796791, 0.1962200
8: -0.1657344, 0.1923937, -0.0784870, 0.0840042, -0.2497386, 0.2708808
9: -0.1371471, 0.2408399, -0.0699542, 0.0744918, -0.2116389, 0.3107941

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 75

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 91

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.95 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1084628, 0.0702156, -0.0693689, 0.0501028, -0.1585656, 0.1395845
1: -0.1035370, 0.0917223, -0.0644828, 0.0600096, -0.1635466, 0.1562051
2: -0.1118782, 0.2057528, -0.0693182, 0.1576336, -0.2695118, 0.2750710
3: -0.0824093, 0.1028060, -0.0454244, 0.0701984, -0.1526076, 0.1482304
4: -0.1153990, 0.1284118, -0.0764688, 0.0795657, -0.1949647, 0.2048807
5: -0.0915292, 0.1323414, -0.0518550, 0.0883976, -0.1799268, 0.1841964
6: -0.1671621, 0.1355069, -0.1218340, 0.0924898, -0.2596520, 0.2573408
7: 0.7501539, 1.0419461, 0.8122238, 1.0189877, -0.2688338, 0.2297223
8: -0.1478122, 0.1701943, -0.1060477, 0.1184625, -0.2662747, 0.2762420
9: -0.1236504, 0.2066720, -0.0921987, 0.1270500, -0.2507003, 0.2988707

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 75

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.04 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.99 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1084628, 0.0702156, -0.0517257, 0.0410258, -0.1494886, 0.1219412
1: -0.1035370, 0.0917223, -0.0468575, 0.0456976, -0.1492345, 0.1385798
2: -0.1118782, 0.2057528, -0.0501108, 0.1359173, -0.2477955, 0.2558635
3: -0.0824093, 0.1028060, -0.0287329, 0.0554825, -0.1378918, 0.1315389
4: -0.1153990, 0.1284118, -0.0588995, 0.0575214, -0.1729204, 0.1873113
5: -0.0915292, 0.1323414, -0.0339499, 0.0685656, -0.1600948, 0.1662913
6: -0.1671621, 0.1355069, -0.1013772, 0.0730762, -0.2402383, 0.2368841
7: 0.7501539, 1.0419461, 0.8402362, 1.0086265, -0.2584726, 0.2017099
8: -0.1478122, 0.1701943, -0.0871993, 0.0951158, -0.2429280, 0.2573936
9: -0.1236504, 0.2066720, -0.0780045, 0.0911162, -0.2147666, 0.2846765

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 75

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.06 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1164450, 0.0743223, -0.0576214, 0.0440590, -0.1605040, 0.1319436
1: -0.1115111, 0.0981975, -0.0527472, 0.0504801, -0.1619912, 0.1509447
2: -0.1205682, 0.2155778, -0.0565292, 0.1431741, -0.2637422, 0.2721070
3: -0.0899609, 0.1094638, -0.0343106, 0.0604000, -0.1503609, 0.1437743
4: -0.1233478, 0.1383852, -0.0647705, 0.0648878, -0.1882356, 0.2031557
5: -0.0996299, 0.1413139, -0.0399332, 0.0751927, -0.1748226, 0.1812471
6: -0.1764174, 0.1442901, -0.1082131, 0.0795635, -0.2559808, 0.2525032
7: 0.7374804, 1.0466338, 0.8308755, 1.0120888, -0.2746084, 0.2157584
8: -0.1563397, 0.1807570, -0.0934977, 0.1029174, -0.2592571, 0.2742547
9: -0.1300722, 0.2229294, -0.0827476, 0.1031238, -0.2331960, 0.3056771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 75

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.99 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1233811, 0.0778906, -0.0653159, 0.0480176, -0.1713987, 0.1432065
1: -0.1184400, 0.1038239, -0.0604339, 0.0567219, -0.1751619, 0.1642579
2: -0.1281192, 0.2241151, -0.0649059, 0.1526450, -0.2807642, 0.2890210
3: -0.0965227, 0.1152490, -0.0415900, 0.0668179, -0.1633406, 0.1568390
4: -0.1302548, 0.1470515, -0.0724329, 0.0745018, -0.2047566, 0.2194843
5: -0.1066689, 0.1491104, -0.0477419, 0.0838418, -0.1905107, 0.1968523
6: -0.1844595, 0.1519222, -0.1171346, 0.0880302, -0.2724897, 0.2690569
7: 0.7264680, 1.0507070, 0.8186586, 1.0166075, -0.2901396, 0.2320483
8: -0.1637496, 0.1899352, -0.1017180, 0.1130993, -0.2768490, 0.2916532
9: -0.1356524, 0.2370559, -0.0889380, 0.1187954, -0.2544478, 0.3259940

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 75

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.09 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.1067123, 0.0693150, -0.0936708, 0.0626055, -0.1693178, 0.1629858
1: -0.1017882, 0.0903024, -0.0887601, 0.0797233, -0.1815115, 0.1790624
2: -0.1099726, 0.2035982, -0.0957749, 0.1875460, -0.2975186, 0.2993731
3: -0.0807533, 0.1013459, -0.0684154, 0.0904683, -0.1712215, 0.1697612
4: -0.1136558, 0.1262247, -0.1006690, 0.1099300, -0.2235858, 0.2268937
5: -0.0897527, 0.1303737, -0.0765177, 0.1157144, -0.2054671, 0.2068914
6: -0.1651326, 0.1335807, -0.1500114, 0.1192306, -0.2843632, 0.2835920
7: 0.7529331, 1.0409182, 0.7736391, 1.0332595, -0.2803264, 0.2672791
8: -0.1459421, 0.1678780, -0.1320098, 0.1506207, -0.2965628, 0.2998878
9: -0.1222421, 0.2031068, -0.1117500, 0.1765455, -0.2987876, 0.3148568

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 75

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.04 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1067123, 0.0693150, -0.0761195, 0.0535758, -0.1602880, 0.1454345
1: -0.1017882, 0.0903024, -0.0712265, 0.0654857, -0.1672739, 0.1615289
2: -0.1099726, 0.2035982, -0.0766674, 0.1659427, -0.2759152, 0.2802656
3: -0.0807533, 0.1013459, -0.0518108, 0.0758290, -0.1565822, 0.1531567
4: -0.1136558, 0.1262247, -0.0831911, 0.0880003, -0.2016561, 0.2094158
5: -0.0897527, 0.1303737, -0.0587058, 0.0959857, -0.1857384, 0.1890796
6: -0.1651326, 0.1335807, -0.1296611, 0.0999179, -0.2650505, 0.2632418
7: 0.7529331, 1.0409182, 0.8015058, 1.0229521, -0.2700189, 0.2394124
8: -0.1459421, 0.1678780, -0.1132595, 0.1273954, -0.2733375, 0.2811374
9: -0.1222421, 0.2031068, -0.0976297, 0.1407988, -0.2630409, 0.3007365

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 75

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.03 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.75 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.75
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.75
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.75
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.75
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.75
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.75
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.75
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.75
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.75
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.75
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.75
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.75
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.75
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.75
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.75
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.75
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.75
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.75
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.75
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.75
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.75
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.75
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.75
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.75
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.75
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.75
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.75
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.75
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.75
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.75
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.75
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.75
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0339846, 0.0326161, -0.0605089, 0.0455446, -0.0795291, 0.0931250
1: -0.0325271, 0.0284551, -0.0556318, 0.0528225, -0.0853497, 0.0840870
2: -0.0291025, 0.1136425, -0.0596727, 0.1467282, -0.1758307, 0.1733153
3: -0.0165030, 0.0427604, -0.0370424, 0.0628085, -0.0793115, 0.0798028
4: -0.0437445, 0.0382073, -0.0676460, 0.0684956, -0.1122402, 0.1058533
5: -0.0187574, 0.0494234, -0.0428636, 0.0784385, -0.0971959, 0.0922870
6: -0.0818421, 0.0532282, -0.1115611, 0.0827409, -0.1645830, 0.1647893
7: 0.8705006, 0.9987579, 0.8262908, 1.0137846, -0.1432840, 0.1724671
8: -0.0709011, 0.0751689, -0.0965826, 0.1067384, -0.1776395, 0.1717515
9: -0.0624980, 0.0626352, -0.0850707, 0.1090049, -0.1715029, 0.1477059

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0390926, 0.0353715, -0.0678325, 0.0493124, -0.0884050, 0.1032040
1: -0.0358485, 0.0344190, -0.0629480, 0.0587634, -0.0946118, 0.0973670
2: -0.0362477, 0.1204658, -0.0676456, 0.1557426, -0.1919903, 0.1881114
3: -0.0198662, 0.0458658, -0.0439709, 0.0689169, -0.0887830, 0.0898367
4: -0.0485309, 0.0428945, -0.0749389, 0.0776462, -0.1261771, 0.1178334
5: -0.0218661, 0.0558081, -0.0502958, 0.0866707, -0.1085368, 0.1061040
6: -0.0876474, 0.0596492, -0.1200526, 0.0907994, -0.1784467, 0.1797018
7: 0.8602889, 1.0018009, 0.8146631, 1.0180854, -0.1577965, 0.1871378
8: -0.0761014, 0.0809484, -0.1044065, 0.1164294, -0.1925309, 0.1853548
9: -0.0676093, 0.0702181, -0.0909627, 0.1239209, -0.1915302, 0.1611808

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0663129, 0.0485305, -0.0516121, 0.0409674, -0.1072803, 0.1001427
1: -0.0614299, 0.0575307, -0.0467441, 0.0456055, -0.1070354, 0.1042748
2: -0.0659913, 0.1538722, -0.0499873, 0.1357776, -0.2017689, 0.2038594
3: -0.0425333, 0.0676494, -0.0286256, 0.0553878, -0.0979211, 0.0962750
4: -0.0734257, 0.0757475, -0.0587865, 0.0573796, -0.1308053, 0.1345340
5: -0.0487537, 0.0849625, -0.0338348, 0.0684380, -0.1171917, 0.1187973
6: -0.1182907, 0.0891273, -0.1012455, 0.0729513, -0.1912420, 0.1903728
7: 0.8170757, 1.0171930, 0.8404164, 1.0085597, -0.1914840, 0.1767766
8: -0.1027830, 0.1144186, -0.0870781, 0.0949656, -0.1977486, 0.2014967
9: -0.0897401, 0.1208260, -0.0779131, 0.0908851, -0.1806252, 0.1987391

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0483850, 0.0393134, -0.0516121, 0.0409674, -0.0893525, 0.0909256
1: -0.0435336, 0.0429511, -0.0467441, 0.0456055, -0.0891391, 0.0896952
2: -0.0465040, 0.1317791, -0.0499873, 0.1357776, -0.1822816, 0.1817663
3: -0.0257222, 0.0526585, -0.0286256, 0.0553878, -0.0811100, 0.0812841
4: -0.0555928, 0.0534371, -0.0587865, 0.0573796, -0.1129724, 0.1122236
5: -0.0305326, 0.0649423, -0.0338348, 0.0684380, -0.0989706, 0.0987770
6: -0.0975913, 0.0693785, -0.1012455, 0.0729513, -0.1705427, 0.1706241
7: 0.8455721, 1.0066943, 0.8404164, 1.0085597, -0.1629876, 0.1662779
8: -0.0836543, 0.0908919, -0.0870781, 0.0949656, -0.1786199, 0.1779700
9: -0.0753100, 0.0843591, -0.0779131, 0.0908851, -0.1661951, 0.1622722

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0541979, 0.0422977, -0.0586689, 0.0445979, -0.0987958, 0.1009666
1: -0.0493273, 0.0477031, -0.0537937, 0.0513299, -0.1006572, 0.1014968
2: -0.0528022, 0.1389603, -0.0576696, 0.1444635, -0.1972657, 0.1966299
3: -0.0310718, 0.0575446, -0.0353016, 0.0612737, -0.0923456, 0.0928462
4: -0.0613614, 0.0606104, -0.0658136, 0.0661967, -0.1275581, 0.1264240
5: -0.0364589, 0.0713445, -0.0409963, 0.0763702, -0.1128291, 0.1123408
6: -0.1042437, 0.0757965, -0.1094276, 0.0807162, -0.1849599, 0.1852242
7: 0.8363110, 1.0100782, 0.8292123, 1.0127040, -0.1763930, 0.1808659
8: -0.0898404, 0.0983872, -0.0946168, 0.1043035, -0.1941440, 0.1930041
9: -0.0799935, 0.0961514, -0.0835904, 0.1052574, -0.1852509, 0.1797418

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0619229, 0.0462721, -0.0659985, 0.0483688, -0.1102917, 0.1122705
1: -0.0570444, 0.0539695, -0.0611158, 0.0572756, -0.1143201, 0.1150854
2: -0.0612122, 0.1484687, -0.0656491, 0.1534852, -0.2146973, 0.2141178
3: -0.0383801, 0.0639879, -0.0422358, 0.0673872, -0.1057673, 0.1062237
4: -0.0690541, 0.0702625, -0.0731125, 0.0753547, -0.1444088, 0.1433750
5: -0.0442986, 0.0800279, -0.0484346, 0.0846091, -0.1289077, 0.1284626
6: -0.1132006, 0.0842968, -0.1179261, 0.0887813, -0.2019819, 0.2022229
7: 0.8240457, 1.0146149, 0.8175749, 1.0170085, -0.1929628, 0.1970400
8: -0.0980932, 0.1086095, -0.1024472, 0.1140026, -0.2120958, 0.2110567
9: -0.0862083, 0.1118850, -0.0894872, 0.1201856, -0.2063939, 0.2013721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0906388, 0.0610456, -0.0499465, 0.0401105, -0.1307492, 0.1109921
1: -0.0857311, 0.0772637, -0.0450802, 0.0442543, -0.1299855, 0.1223438
2: -0.0924740, 0.1838139, -0.0481739, 0.1337274, -0.2262014, 0.2319878
3: -0.0655468, 0.0879393, -0.0270498, 0.0539985, -0.1195454, 0.1149891
4: -0.0976496, 0.1061416, -0.0571277, 0.0552985, -0.1529481, 0.1632694
5: -0.0734406, 0.1123062, -0.0321444, 0.0665657, -0.1400064, 0.1444506
6: -0.1464958, 0.1158942, -0.0993143, 0.0711185, -0.2176143, 0.2152085
7: 0.7784533, 1.0314788, 0.8430610, 1.0075817, -0.2291284, 0.1884178
8: -0.1287706, 0.1466084, -0.0852986, 0.0927615, -0.2215321, 0.2319069
9: -0.1093106, 0.1703701, -0.0765731, 0.0874926, -0.1968032, 0.2469432

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0726935, 0.0518132, -0.0499465, 0.0401105, -0.1128039, 0.1017597
1: -0.0678040, 0.0627065, -0.0450802, 0.0442543, -0.1120583, 0.1077867
2: -0.0729376, 0.1617257, -0.0481739, 0.1337274, -0.2066649, 0.2098996
3: -0.0485696, 0.0729713, -0.0270498, 0.0539985, -0.1025681, 0.1000211
4: -0.0797795, 0.0837197, -0.0571277, 0.0552985, -0.1350780, 0.1408474
5: -0.0552289, 0.0921347, -0.0321444, 0.0665657, -0.1217946, 0.1242791
6: -0.1256887, 0.0961481, -0.0993143, 0.0711185, -0.1968072, 0.1954624
7: 0.8069453, 1.0209402, 0.8430610, 1.0075817, -0.2006364, 0.1778792
8: -0.1095995, 0.1228618, -0.0852986, 0.0927615, -0.2023609, 0.2081604
9: -0.0948734, 0.1338212, -0.0765731, 0.0874926, -0.1823660, 0.2103943

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0339846, 0.0326161, -0.1183239, 0.0752888, -0.1092734, 0.1509401
1: -0.0325271, 0.0284551, -0.1133880, 0.0997216, -0.1322487, 0.1418431
2: -0.0291025, 0.1136425, -0.1226135, 0.2178904, -0.2469929, 0.2362561
3: -0.0165030, 0.0427604, -0.0917384, 0.1110309, -0.1275339, 0.1344988
4: -0.0437445, 0.0382073, -0.1252188, 0.1407328, -0.1844773, 0.1634261
5: -0.0187574, 0.0494234, -0.1015366, 0.1434259, -0.1621833, 0.1509600
6: -0.0818421, 0.0532282, -0.1785958, 0.1463576, -0.2281997, 0.2318241
7: 0.8705006, 0.9987579, 0.7344973, 1.0477374, -0.1772367, 0.2642606
8: -0.0709011, 0.0751689, -0.1583470, 0.1832431, -0.2541442, 0.2335158
9: -0.0624980, 0.0626352, -0.1315838, 0.2267559, -0.2892539, 0.1942190

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=56, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 75

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0390926, 0.0353715, -0.1252390, 0.0788465, -0.1179391, 0.1606105
1: -0.0358485, 0.0344190, -0.1202961, 0.1053311, -0.1411796, 0.1547152
2: -0.0362477, 0.1204658, -0.1301418, 0.2264019, -0.2626496, 0.2506076
3: -0.0198662, 0.0458658, -0.0982804, 0.1167987, -0.1366649, 0.1441463
4: -0.0485309, 0.0428945, -0.1321049, 0.1493729, -0.1979038, 0.1749993
5: -0.0218661, 0.0558081, -0.1085544, 0.1511989, -0.1730650, 0.1643625
6: -0.0876474, 0.0596492, -0.1866137, 0.1539665, -0.2416139, 0.2462629
7: 0.8602889, 1.0018009, 0.7235181, 1.0517981, -0.1915092, 0.2782828
8: -0.0761014, 0.0809484, -0.1657344, 0.1923937, -0.2684951, 0.2466828
9: -0.0676093, 0.0702181, -0.1371471, 0.2408399, -0.3084492, 0.2073652

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=56, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 75

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0663129, 0.0485305, -0.1084628, 0.0702156, -0.1365285, 0.1569933
1: -0.0614299, 0.0575307, -0.1035370, 0.0917223, -0.1531523, 0.1610676
2: -0.0659913, 0.1538722, -0.1118782, 0.2057528, -0.2717441, 0.2657503
3: -0.0425333, 0.0676494, -0.0824093, 0.1028060, -0.1453393, 0.1500587
4: -0.0734257, 0.0757475, -0.1153990, 0.1284118, -0.2018375, 0.1911465
5: -0.0487537, 0.0849625, -0.0915292, 0.1323414, -0.1810951, 0.1764917
6: -0.1182907, 0.0891273, -0.1671621, 0.1355069, -0.2537976, 0.2562894
7: 0.8170757, 1.0171930, 0.7501539, 1.0419461, -0.2248703, 0.2670391
8: -0.1027830, 0.1144186, -0.1478122, 0.1701943, -0.2729774, 0.2622308
9: -0.0897401, 0.1208260, -0.1236504, 0.2066720, -0.2964121, 0.2444763

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 75

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0483850, 0.0393134, -0.1084628, 0.0702156, -0.1186006, 0.1477762
1: -0.0435336, 0.0429511, -0.1035370, 0.0917223, -0.1352559, 0.1464881
2: -0.0465040, 0.1317791, -0.1118782, 0.2057528, -0.2522568, 0.2436573
3: -0.0257222, 0.0526585, -0.0824093, 0.1028060, -0.1285282, 0.1350678
4: -0.0555928, 0.0534371, -0.1153990, 0.1284118, -0.1840046, 0.1688361
5: -0.0305326, 0.0649423, -0.0915292, 0.1323414, -0.1628739, 0.1564715
6: -0.0975913, 0.0693785, -0.1671621, 0.1355069, -0.2330983, 0.2365407
7: 0.8455721, 1.0066943, 0.7501539, 1.0419461, -0.1963739, 0.2565404
8: -0.0836543, 0.0908919, -0.1478122, 0.1701943, -0.2538486, 0.2387041
9: -0.0753100, 0.0843591, -0.1236504, 0.2066720, -0.2819820, 0.2080094

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 75

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.11 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0541979, 0.0422977, -0.1164450, 0.0743223, -0.1285202, 0.1587428
1: -0.0493273, 0.0477031, -0.1115111, 0.0981975, -0.1475247, 0.1592142
2: -0.0528022, 0.1389603, -0.1205682, 0.2155778, -0.2683800, 0.2595285
3: -0.0310718, 0.0575446, -0.0899609, 0.1094638, -0.1405356, 0.1475055
4: -0.0613614, 0.0606104, -0.1233478, 0.1383852, -0.1997466, 0.1839581
5: -0.0364589, 0.0713445, -0.0996299, 0.1413139, -0.1777728, 0.1709744
6: -0.1042437, 0.0757965, -0.1764174, 0.1442901, -0.2485338, 0.2522139
7: 0.8363110, 1.0100782, 0.7374804, 1.0466338, -0.2103229, 0.2725978
8: -0.0898404, 0.0983872, -0.1563397, 0.1807570, -0.2705974, 0.2547270
9: -0.0799935, 0.0961514, -0.1300722, 0.2229294, -0.3029229, 0.2262236

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 75

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.05 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0619229, 0.0462721, -0.1233811, 0.0778906, -0.1398135, 0.1696531
1: -0.0570444, 0.0539695, -0.1184400, 0.1038239, -0.1608684, 0.1724096
2: -0.0612122, 0.1484687, -0.1281192, 0.2241151, -0.2853272, 0.2765879
3: -0.0383801, 0.0639879, -0.0965227, 0.1152490, -0.1536291, 0.1605106
4: -0.0690541, 0.0702625, -0.1302548, 0.1470515, -0.2161055, 0.2005172
5: -0.0442986, 0.0800279, -0.1066689, 0.1491104, -0.1934090, 0.1866968
6: -0.1132006, 0.0842968, -0.1844595, 0.1519222, -0.2651228, 0.2687563
7: 0.8240457, 1.0146149, 0.7264680, 1.0507070, -0.2266613, 0.2881470
8: -0.0980932, 0.1086095, -0.1637496, 0.1899352, -0.2880284, 0.2723592
9: -0.0862083, 0.1118850, -0.1356524, 0.2370559, -0.3232643, 0.2475373

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 75

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0906388, 0.0610456, -0.1067123, 0.0693150, -0.1599538, 0.1677579
1: -0.0857311, 0.0772637, -0.1017882, 0.0903024, -0.1760335, 0.1790519
2: -0.0924740, 0.1838139, -0.1099726, 0.2035982, -0.2960723, 0.2937864
3: -0.0655468, 0.0879393, -0.0807533, 0.1013459, -0.1668927, 0.1686925
4: -0.0976496, 0.1061416, -0.1136558, 0.1262247, -0.2238742, 0.2197974
5: -0.0734406, 0.1123062, -0.0897527, 0.1303737, -0.2038144, 0.2020589
6: -0.1464958, 0.1158942, -0.1651326, 0.1335807, -0.2800765, 0.2810268
7: 0.7784533, 1.0314788, 0.7529331, 1.0409182, -0.2624649, 0.2785456
8: -0.1287706, 0.1466084, -0.1459421, 0.1678780, -0.2966486, 0.2925505
9: -0.1093106, 0.1703701, -0.1222421, 0.2031068, -0.3124175, 0.2926122

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 75

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0726935, 0.0518132, -0.1067123, 0.0693150, -0.1420085, 0.1585255
1: -0.0678040, 0.0627065, -0.1017882, 0.0903024, -0.1581064, 0.1644948
2: -0.0729376, 0.1617257, -0.1099726, 0.2035982, -0.2765358, 0.2716982
3: -0.0485696, 0.0729713, -0.0807533, 0.1013459, -0.1499155, 0.1537246
4: -0.0797795, 0.0837197, -0.1136558, 0.1262247, -0.2060042, 0.1973755
5: -0.0552289, 0.0921347, -0.0897527, 0.1303737, -0.1856027, 0.1818874
6: -0.1256887, 0.0961481, -0.1651326, 0.1335807, -0.2592694, 0.2612807
7: 0.8069453, 1.0209402, 0.7529331, 1.0409182, -0.2339729, 0.2680070
8: -0.1095995, 0.1228618, -0.1459421, 0.1678780, -0.2774774, 0.2688040
9: -0.0948734, 0.1338212, -0.1222421, 0.2031068, -0.2979802, 0.2560632

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 75

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1147429, 0.0734465, -0.0363444, 0.0338890, -0.1486320, 0.1097909
1: -0.1098107, 0.0968167, -0.0340615, 0.0312103, -0.1410210, 0.1308782
2: -0.1187151, 0.2134828, -0.0324034, 0.1167947, -0.2355098, 0.2458861
3: -0.0883506, 0.1080440, -0.0180567, 0.0441950, -0.1325457, 0.1261007
4: -0.1216527, 0.1362585, -0.0459557, 0.0403727, -0.1620254, 0.1822142
5: -0.0979025, 0.1394006, -0.0201936, 0.0523730, -0.1502755, 0.1595942
6: -0.1744438, 0.1424172, -0.0845240, 0.0561946, -0.2306384, 0.2269412
7: 0.7401829, 1.0456343, 0.8657830, 1.0001640, -0.2599811, 0.1798513
8: -0.1545213, 0.1785046, -0.0733035, 0.0778389, -0.2323602, 0.2518081
9: -0.1287028, 0.2194627, -0.0648593, 0.0661383, -0.1948411, 0.2843220

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.01 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1384956, 0.0856667, -0.0363444, 0.0338890, -0.1723846, 0.1220111
1: -0.1335392, 0.1160847, -0.0340615, 0.0312103, -0.1647496, 0.1501462
2: -0.1445738, 0.2427189, -0.0324034, 0.1167947, -0.2613685, 0.2751223
3: -0.1108219, 0.1278557, -0.0180567, 0.0441950, -0.1550170, 0.1459124
4: -0.1453060, 0.1659364, -0.0459557, 0.0403727, -0.1856787, 0.2118921
5: -0.1220077, 0.1661000, -0.0201936, 0.0523730, -0.1743807, 0.1862936
6: -0.2019843, 0.1685534, -0.0845240, 0.0561946, -0.2581789, 0.2530775
7: 0.7024704, 1.0595833, 0.8657830, 1.0001640, -0.2976936, 0.1938003
8: -0.1798966, 0.2099359, -0.0733035, 0.0778389, -0.2577355, 0.2832394
9: -0.1478122, 0.2678394, -0.0648593, 0.0661383, -0.2139505, 0.3326987

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1216495, 0.0769998, -0.0418055, 0.0366356, -0.1582851, 0.1188053
1: -0.1167102, 0.1024193, -0.0377808, 0.0371550, -0.1538652, 0.1402001
2: -0.1262341, 0.2219837, -0.0395256, 0.1239706, -0.2502047, 0.2615093
3: -0.0948846, 0.1138047, -0.0216612, 0.0472905, -0.1421750, 0.1354659
4: -0.1285305, 0.1448880, -0.0507267, 0.0459712, -0.1745016, 0.1956146
5: -0.1049116, 0.1471640, -0.0241145, 0.0587371, -0.1636487, 0.1712785
6: -0.1824518, 0.1500168, -0.0907062, 0.0625949, -0.2450466, 0.2407231
7: 0.7292173, 1.0496902, 0.8555781, 1.0031972, -0.2739799, 0.1941122
8: -0.1618997, 0.1876439, -0.0784870, 0.0840042, -0.2459039, 0.2661309
9: -0.1342593, 0.2335292, -0.0699542, 0.0744918, -0.2087510, 0.3034834

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1453879, 0.0892125, -0.0418055, 0.0366356, -0.1820234, 0.1310179
1: -0.1404244, 0.1216757, -0.0377808, 0.0371550, -0.1775794, 0.1594564
2: -0.1520770, 0.2512023, -0.0395256, 0.1239706, -0.2760476, 0.2907279
3: -0.1173423, 0.1336044, -0.0216612, 0.0472905, -0.1646327, 0.1552656
4: -0.1521693, 0.1745479, -0.0507267, 0.0459712, -0.1981405, 0.2252746
5: -0.1290022, 0.1738473, -0.0241145, 0.0587371, -0.1877393, 0.1979617
6: -0.2099757, 0.1761373, -0.0907062, 0.0625949, -0.2725705, 0.2668435
7: 0.6915276, 1.0636309, 0.8555781, 1.0031972, -0.3116696, 0.2080529
8: -0.1872596, 0.2190561, -0.0784870, 0.0840042, -0.2712638, 0.2975430
9: -0.1533571, 0.2818767, -0.0699542, 0.0744918, -0.2278489, 0.3518309

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 33

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0909785, 0.0612203, -0.0601080, 0.0453383, -0.1363168, 0.1213284
1: -0.0860704, 0.0775392, -0.0552314, 0.0524973, -0.1385677, 0.1327706
2: -0.0928437, 0.1842320, -0.0592363, 0.1462348, -0.2390785, 0.2434683
3: -0.0658682, 0.0882226, -0.0366631, 0.0624740, -0.1283422, 0.1248857
4: -0.0979879, 0.1065659, -0.0672467, 0.0679948, -0.1659827, 0.1738127
5: -0.0737853, 0.1126880, -0.0424567, 0.0779878, -0.1517732, 0.1551448
6: -0.1468896, 0.1162680, -0.1110963, 0.0822997, -0.2291893, 0.2273642
7: 0.7779139, 1.0316782, 0.8269273, 1.0135490, -0.2356350, 0.2047509
8: -0.1291335, 0.1470578, -0.0961543, 0.1062079, -0.2353414, 0.2432121
9: -0.1095840, 0.1710619, -0.0847482, 0.1081885, -0.2177725, 0.2558101

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.95 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.07 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1008214, 0.0662843, -0.0674400, 0.0491104, -0.1499318, 0.1337242
1: -0.0959033, 0.0855237, -0.0625558, 0.0584449, -0.1543482, 0.1480795
2: -0.1035594, 0.1963473, -0.0672183, 0.1552594, -0.2588187, 0.2635656
3: -0.0751801, 0.0964324, -0.0435995, 0.0685895, -0.1437696, 0.1400318
4: -0.1077895, 0.1188643, -0.0745480, 0.0771557, -0.1849452, 0.1934123
5: -0.0837744, 0.1237521, -0.0498975, 0.0862294, -0.1700037, 0.1736495
6: -0.1583022, 0.1270987, -0.1195974, 0.0903674, -0.2486696, 0.2466961
7: 0.7622862, 1.0374585, 0.8152864, 1.0178549, -0.2555687, 0.2221721
8: -0.1396488, 0.1600826, -0.1039871, 0.1159100, -0.2555588, 0.2640697
9: -0.1175027, 0.1911089, -0.0906469, 0.1231214, -0.2406241, 0.2817558

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.04 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.95 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0909785, 0.0612203, -0.0434419, 0.0373359, -0.1283144, 0.1046623
1: -0.0860704, 0.0775392, -0.0390548, 0.0386710, -0.1247414, 0.1165940
2: -0.0928437, 0.1842320, -0.0413444, 0.1260130, -0.2188567, 0.2255764
3: -0.0658682, 0.0882226, -0.0227234, 0.0482576, -0.1141258, 0.1109459
4: -0.0979879, 0.1065659, -0.0519596, 0.0479239, -0.1459118, 0.1585256
5: -0.0737853, 0.1126880, -0.0256419, 0.0603601, -0.1341454, 0.1383299
6: -0.1468896, 0.1162680, -0.0925071, 0.0642681, -0.2111577, 0.2087750
7: 0.7779139, 1.0316782, 0.8529611, 1.0040116, -0.2260977, 0.1787171
8: -0.1291335, 0.1470578, -0.0798175, 0.0858057, -0.2149392, 0.2268753
9: -0.1095840, 0.1710619, -0.0712829, 0.0770725, -0.1866565, 0.2423448

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.01 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1008214, 0.0662843, -0.0497362, 0.0400023, -0.1408237, 0.1160204
1: -0.0959033, 0.0855237, -0.0448701, 0.0440837, -0.1399870, 0.1303937
2: -0.1035594, 0.1963473, -0.0479449, 0.1334685, -0.2370279, 0.2442921
3: -0.0751801, 0.0964324, -0.0268508, 0.0538231, -0.1290032, 0.1232832
4: -0.1077895, 0.1188643, -0.0569184, 0.0550357, -0.1628252, 0.1757827
5: -0.0837744, 0.1237521, -0.0319310, 0.0663293, -0.1501037, 0.1556830
6: -0.1583022, 0.1270987, -0.0990705, 0.0708870, -0.2291892, 0.2261692
7: 0.7622862, 1.0374585, 0.8433948, 1.0074580, -0.2451718, 0.1940637
8: -0.1396488, 0.1600826, -0.0850740, 0.0924832, -0.2321320, 0.2451566
9: -0.1175027, 0.1911089, -0.0764039, 0.0870644, -0.2045671, 0.2675128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 33

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1083370, 0.0701509, -0.0554907, 0.0429628, -0.1512998, 0.1256416
1: -0.1034112, 0.0916203, -0.0506187, 0.0487518, -0.1521630, 0.1422390
2: -0.1117412, 0.2055979, -0.0542096, 0.1405516, -0.2522928, 0.2598075
3: -0.0822902, 0.1027010, -0.0322949, 0.0586228, -0.1409130, 0.1349959
4: -0.1152737, 0.1282546, -0.0626488, 0.0622257, -0.1774994, 0.1909033
5: -0.0914015, 0.1322000, -0.0377709, 0.0727977, -0.1641992, 0.1699709
6: -0.1670162, 0.1353684, -0.1057426, 0.0772191, -0.2442353, 0.2411110
7: 0.7503536, 1.0418723, 0.8342583, 1.0108373, -0.2604837, 0.2076140
8: -0.1476779, 0.1700278, -0.0912216, 0.1000980, -0.2477758, 0.2612493
9: -0.1235491, 0.2064158, -0.0810335, 0.0987845, -0.2223336, 0.2874493

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.99 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1176037, 0.0749183, -0.0537378, 0.0420610, -0.1596646, 0.1286561
1: -0.1126685, 0.0991374, -0.0488676, 0.0473298, -0.1599983, 0.1480050
2: -0.1218296, 0.2170039, -0.0523013, 0.1383939, -0.2602235, 0.2693052
3: -0.0910570, 0.1104302, -0.0306365, 0.0571608, -0.1482178, 0.1410667
4: -0.1245016, 0.1398329, -0.0609032, 0.0600355, -0.1845371, 0.2007360
5: -0.1008057, 0.1426163, -0.0359919, 0.0708274, -0.1716331, 0.1786082
6: -0.1777608, 0.1455650, -0.1037102, 0.0752902, -0.2530510, 0.2492752
7: 0.7356408, 1.0473142, 0.8370416, 1.0098081, -0.2741673, 0.2102726
8: -0.1575774, 0.1822902, -0.0893489, 0.0977784, -0.2553558, 0.2716390
9: -0.1310044, 0.2252891, -0.0796232, 0.0952143, -0.2262187, 0.3049123

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.01 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1153692, 0.0737688, -0.0631717, 0.0469145, -0.1622838, 0.1369405
1: -0.1104364, 0.0973248, -0.0582920, 0.0549826, -0.1654190, 0.1556168
2: -0.1193970, 0.2142537, -0.0625717, 0.1500058, -0.2694028, 0.2768254
3: -0.0889432, 0.1085665, -0.0395616, 0.0650295, -0.1539726, 0.1481280
4: -0.1222766, 0.1370411, -0.0702976, 0.0718228, -0.1940994, 0.2073387
5: -0.0985382, 0.1401046, -0.0455659, 0.0814317, -0.1799699, 0.1856705
6: -0.1751700, 0.1431064, -0.1146486, 0.0856709, -0.2608409, 0.2577550
7: 0.7391883, 1.0460019, 0.8220631, 1.0153483, -0.2761600, 0.2239388
8: -0.1551905, 0.1793334, -0.0994273, 0.1102620, -0.2654525, 0.2787607
9: -0.1292068, 0.2207384, -0.0872130, 0.1144283, -0.2436351, 0.3079514

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.00 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1244952, 0.0784638, -0.0613889, 0.0459973, -0.1704926, 0.1398527
1: -0.1195531, 0.1047277, -0.0565110, 0.0535363, -0.1730894, 0.1612387
2: -0.1293321, 0.2254864, -0.0606308, 0.1478115, -0.2771435, 0.2861172
3: -0.0975768, 0.1161783, -0.0378749, 0.0635424, -0.1611192, 0.1540532
4: -0.1313642, 0.1484436, -0.0685223, 0.0695953, -0.2009595, 0.2169659
5: -0.1077996, 0.1503628, -0.0437567, 0.0794277, -0.1872273, 0.1941194
6: -0.1857513, 0.1531481, -0.1125815, 0.0837092, -0.2694605, 0.2657296
7: 0.7246991, 1.0513614, 0.8248936, 1.0143014, -0.2896023, 0.2264678
8: -0.1649398, 0.1914094, -0.0975227, 0.1079029, -0.2728427, 0.2889321
9: -0.1365487, 0.2393251, -0.0857787, 0.1107974, -0.2473461, 0.3251038

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 255

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.04 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0893545, 0.0603848, -0.0845518, 0.0579139, -0.1472684, 0.1449366
1: -0.0844481, 0.0762218, -0.0796502, 0.0723259, -0.1567740, 0.1558720
2: -0.0910757, 0.1822331, -0.0858472, 0.1763216, -0.2673973, 0.2680803
3: -0.0643318, 0.0868680, -0.0597882, 0.0828621, -0.1471939, 0.1466562
4: -0.0963707, 0.1045369, -0.0915881, 0.0985361, -0.1949068, 0.1961250
5: -0.0721373, 0.1108625, -0.0672632, 0.1054640, -0.1776013, 0.1781258
6: -0.1450067, 0.1144810, -0.1394380, 0.1091964, -0.2542031, 0.2539190
7: 0.7804925, 1.0307244, 0.7881178, 1.0279040, -0.2474115, 0.2426066
8: -0.1273986, 0.1449088, -0.1222678, 0.1385536, -0.2659521, 0.2671766
9: -0.1082775, 0.1677544, -0.1044136, 0.1579727, -0.2662501, 0.2721680

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 33

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0990658, 0.0653810, -0.0917531, 0.0616189, -0.1606847, 0.1571341
1: -0.0941495, 0.0840995, -0.0868443, 0.0781676, -0.1723170, 0.1709438
2: -0.1016481, 0.1941863, -0.0936871, 0.1851855, -0.2868336, 0.2878734
3: -0.0735192, 0.0949680, -0.0666011, 0.0888686, -0.1623878, 0.1615690
4: -0.1060413, 0.1166707, -0.0987593, 0.1075339, -0.2135752, 0.2154300
5: -0.0819927, 0.1217786, -0.0745715, 0.1135588, -0.1955514, 0.1963502
6: -0.1562666, 0.1251668, -0.1477878, 0.1171204, -0.2733870, 0.2729546
7: 0.7650736, 1.0364275, 0.7766840, 1.0321331, -0.2670595, 0.2597435
8: -0.1377732, 0.1577595, -0.1299610, 0.1480829, -0.2858561, 0.2877206
9: -0.1160903, 0.1875332, -0.1102071, 0.1726397, -0.2887300, 0.2977403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0893545, 0.0603848, -0.0672397, 0.0490074, -0.1383619, 0.1276246
1: -0.0844481, 0.0762218, -0.0623558, 0.0582825, -0.1427306, 0.1385776
2: -0.0910757, 0.1822331, -0.0670003, 0.1550129, -0.2460887, 0.2492334
3: -0.0643318, 0.0868680, -0.0434101, 0.0684225, -0.1327543, 0.1302781
4: -0.0963707, 0.1045369, -0.0743486, 0.0769055, -0.1732762, 0.1788855
5: -0.0721373, 0.1108625, -0.0496943, 0.0860043, -0.1581416, 0.1605568
6: -0.1450067, 0.1144810, -0.1193653, 0.0901471, -0.2351537, 0.2338463
7: 0.7804925, 1.0307244, 0.8156042, 1.0177372, -0.2372446, 0.2151203
8: -0.1273986, 0.1449088, -0.1037732, 0.1156451, -0.2430437, 0.2486820
9: -0.1082775, 0.1677544, -0.0904858, 0.1227135, -0.2309910, 0.2582402

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.01 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0990658, 0.0653810, -0.0740984, 0.0525360, -0.1516018, 0.1394794
1: -0.0941495, 0.0840995, -0.0692075, 0.0638462, -0.1579957, 0.1533070
2: -0.1016481, 0.1941863, -0.0744671, 0.1634550, -0.2651031, 0.2686534
3: -0.0735192, 0.0949680, -0.0498987, 0.0741432, -0.1476624, 0.1448667
4: -0.1060413, 0.1166707, -0.0811786, 0.0854751, -0.1915164, 0.1978493
5: -0.0819927, 0.1217786, -0.0566548, 0.0937138, -0.1757065, 0.1784334
6: -0.1562666, 0.1251668, -0.1273177, 0.0976941, -0.2539607, 0.2524845
7: 0.7650736, 1.0364275, 0.8047146, 1.0217651, -0.2566915, 0.2317129
8: -0.1377732, 0.1577595, -0.1111004, 0.1247209, -0.2624941, 0.2688599
9: -0.1160903, 0.1875332, -0.0960037, 0.1366825, -0.2527728, 0.2835369

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 255

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.86 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.67 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0339846, 0.0326161, -0.0570411, 0.0437605, -0.0777450, 0.0896572
1: -0.0325271, 0.0284551, -0.0521676, 0.0500095, -0.0825366, 0.0806227
2: -0.0291025, 0.1136425, -0.0558975, 0.1424599, -0.1715624, 0.1695401
3: -0.0165030, 0.0427604, -0.0337617, 0.0599160, -0.0764190, 0.0765220
4: -0.0437445, 0.0382073, -0.0641927, 0.0641629, -0.1079074, 0.1024000
5: -0.0187574, 0.0494234, -0.0393443, 0.0745405, -0.0932979, 0.0887677
6: -0.0818421, 0.0532282, -0.1075404, 0.0789251, -0.1607672, 0.1607686
7: 0.8705006, 0.9987579, 0.8317966, 1.0117481, -0.1412475, 0.1669613
8: -0.0709011, 0.0751689, -0.0928778, 0.1021496, -0.1730507, 0.1680467
9: -0.0624980, 0.0626352, -0.0822808, 0.1019422, -0.1644402, 0.1449160

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 68

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0339846, 0.0326161, -0.0819554, 0.0565782, -0.0905628, 0.1145715
1: -0.0325271, 0.0284551, -0.0770565, 0.0702198, -0.1027469, 0.1055117
2: -0.0291025, 0.1136425, -0.0830207, 0.1731259, -0.2022284, 0.1966633
3: -0.0165030, 0.0427604, -0.0573319, 0.0806966, -0.0971996, 0.1000923
4: -0.0437445, 0.0382073, -0.0890026, 0.0952921, -0.1390366, 0.1272100
5: -0.0187574, 0.0494234, -0.0646284, 0.1025456, -0.1213031, 0.1140518
6: -0.0818421, 0.0532282, -0.1364277, 0.1063395, -0.1881816, 0.1896559
7: 0.8705006, 0.9987579, 0.7922398, 1.0263795, -0.1558788, 0.2065181
8: -0.0709011, 0.0751689, -0.1194941, 0.1351179, -0.2060190, 0.1946630
9: -0.0624980, 0.0626352, -0.1023248, 0.1526848, -0.2151828, 0.1649600

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 68

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.14 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0390926, 0.0353715, -0.0643588, 0.0475252, -0.0866178, 0.0997303
1: -0.0358485, 0.0344190, -0.0594778, 0.0559455, -0.0917940, 0.0938969
2: -0.0362477, 0.1204658, -0.0638639, 0.1514670, -0.1877147, 0.1843297
3: -0.0198662, 0.0458658, -0.0406846, 0.0660196, -0.0858857, 0.0865505
4: -0.0485309, 0.0428945, -0.0714798, 0.0733060, -0.1218369, 0.1143742
5: -0.0218661, 0.0558081, -0.0467707, 0.0827659, -0.1046320, 0.1025788
6: -0.0876474, 0.0596492, -0.1160250, 0.0869771, -0.1746245, 0.1756742
7: 0.8602889, 1.0018009, 0.8201783, 1.0160456, -0.1557567, 0.1816226
8: -0.0761014, 0.0809484, -0.1006955, 0.1118329, -0.1879343, 0.1816438
9: -0.0676093, 0.0702181, -0.0881680, 0.1168461, -0.1844554, 0.1583862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 33

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0390926, 0.0353715, -0.0891633, 0.0602865, -0.0993791, 0.1245349
1: -0.0358485, 0.0344190, -0.0842571, 0.0760668, -0.1119152, 0.1186761
2: -0.0362477, 0.1204658, -0.0908677, 0.1819979, -0.2182456, 0.2113335
3: -0.0198662, 0.0458658, -0.0641510, 0.0867086, -0.1065747, 0.1100168
4: -0.0485309, 0.0428945, -0.0961803, 0.1042981, -0.1528290, 0.1390748
5: -0.0218661, 0.0558081, -0.0719433, 0.1106478, -0.1325139, 0.1277514
6: -0.0876474, 0.0596492, -0.1447851, 0.1142707, -0.2019181, 0.2044343
7: 0.8602889, 1.0018009, 0.7807958, 1.0306121, -0.1703233, 0.2210051
8: -0.0761014, 0.0809484, -0.1271944, 0.1446559, -0.2207573, 0.2081428
9: -0.0676093, 0.0702181, -0.1081236, 0.1673651, -0.2349744, 0.1783417

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0570411, 0.0437605, -0.0382439, 0.0349137, -0.0919548, 0.0820044
1: -0.0521676, 0.0500095, -0.0352966, 0.0334281, -0.0855957, 0.0853060
2: -0.0558976, 0.1424600, -0.0350604, 0.1193321, -0.1752296, 0.1775204
3: -0.0337617, 0.0599160, -0.0193073, 0.0453499, -0.0791115, 0.0792233
4: -0.0641927, 0.0641629, -0.0477357, 0.0421157, -0.1063083, 0.1118985
5: -0.0393443, 0.0745405, -0.0213496, 0.0547473, -0.0940916, 0.0958901
6: -0.1075403, 0.0789251, -0.0866829, 0.0585824, -0.1661227, 0.1656079
7: 0.8317966, 1.0117480, 0.8619856, 1.0012956, -0.1694989, 0.1497624
8: -0.0928779, 0.1021496, -0.0752373, 0.0799880, -0.1728659, 0.1773869
9: -0.0822808, 0.1019422, -0.0667601, 0.0689582, -0.1512390, 0.1687023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.53 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0643588, 0.0475252, -0.0442969, 0.0376780, -0.1020368, 0.0918222
1: -0.0594778, 0.0559455, -0.0398294, 0.0394113, -0.0988891, 0.0957749
2: -0.0638639, 0.1514670, -0.0422368, 0.1270103, -0.1908742, 0.1937038
3: -0.0406846, 0.0660196, -0.0232421, 0.0490188, -0.0897035, 0.0892616
4: -0.0714798, 0.0733060, -0.0525880, 0.0488775, -0.1203572, 0.1258940
5: -0.0467707, 0.0827660, -0.0264878, 0.0611526, -0.1079233, 0.1092538
6: -0.1160250, 0.0869771, -0.0933864, 0.0651521, -0.1811770, 0.1803635
7: 0.8201783, 1.0160456, 0.8516830, 1.0044756, -0.1842973, 0.1643626
8: -0.1006955, 0.1118328, -0.0804811, 0.0866854, -0.1873809, 0.1923139
9: -0.0881680, 0.1168461, -0.0719794, 0.0783329, -0.1665009, 0.1888255

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0404796, 0.0360558, -0.0382439, 0.0349137, -0.0753932, 0.0742997
1: -0.0368041, 0.0359003, -0.0352966, 0.0334281, -0.0702322, 0.0711969
2: -0.0380223, 0.1222803, -0.0350604, 0.1193321, -0.1573544, 0.1573408
3: -0.0207822, 0.0466371, -0.0193073, 0.0453499, -0.0661320, 0.0659445
4: -0.0497198, 0.0443550, -0.0477357, 0.0421157, -0.0918354, 0.0920907
5: -0.0229013, 0.0573939, -0.0213496, 0.0547473, -0.0776486, 0.0787435
6: -0.0892158, 0.0612440, -0.0866829, 0.0585824, -0.1477982, 0.1479269
7: 0.8577442, 1.0025570, 0.8619856, 1.0012956, -0.1435513, 0.1405715
8: -0.0773930, 0.0825132, -0.0752373, 0.0799880, -0.1573810, 0.1577505
9: -0.0688789, 0.0723558, -0.0667601, 0.0689582, -0.1378371, 0.1391159

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0465125, 0.0385643, -0.0442969, 0.0376780, -0.0841905, 0.0828613
1: -0.0418369, 0.0413297, -0.0398294, 0.0394113, -0.0812482, 0.0811592
2: -0.0445495, 0.1295948, -0.0422368, 0.1270103, -0.1715598, 0.1718317
3: -0.0245862, 0.0509914, -0.0232421, 0.0490188, -0.0736050, 0.0742335
4: -0.0542165, 0.0513486, -0.0525880, 0.0488775, -0.1030940, 0.1039366
5: -0.0286799, 0.0632065, -0.0264878, 0.0611526, -0.0898325, 0.0896943
6: -0.0956653, 0.0674427, -0.0933864, 0.0651521, -0.1608174, 0.1608291
7: 0.8483711, 1.0056781, 0.8516830, 1.0044756, -0.1561044, 0.1539951
8: -0.0822009, 0.0889652, -0.0804811, 0.0866854, -0.1688863, 0.1694463
9: -0.0737844, 0.0815988, -0.0719794, 0.0783329, -0.1521173, 0.1535782

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0419605, 0.0367033, -0.0344872, 0.0328873, -0.0748477, 0.0711905
1: -0.0378949, 0.0373017, -0.0328539, 0.0290420, -0.0669369, 0.0701556
2: -0.0397012, 0.1241682, -0.0298056, 0.1143140, -0.1540152, 0.1539738
3: -0.0217640, 0.0473668, -0.0168340, 0.0430660, -0.0648300, 0.0642008
4: -0.0508444, 0.0461601, -0.0442155, 0.0386686, -0.0895130, 0.0903755
5: -0.0242563, 0.0588941, -0.0190633, 0.0500517, -0.0743080, 0.0779575
6: -0.0908805, 0.0627528, -0.0824134, 0.0538600, -0.1447405, 0.1451662
7: 0.8553250, 1.0032721, 0.8694958, 0.9990575, -0.1437325, 0.1337762
8: -0.0786149, 0.0841784, -0.0714128, 0.0757376, -0.1543525, 0.1555913
9: -0.0700799, 0.0747414, -0.0630009, 0.0633814, -0.1334613, 0.1377424

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 68

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0498582, 0.0400651, -0.0464885, 0.0385547, -0.0884129, 0.0865536
1: -0.0449920, 0.0441827, -0.0418152, 0.0413090, -0.0863009, 0.0859979
2: -0.0480778, 0.1336188, -0.0445244, 0.1295668, -0.1776447, 0.1781431
3: -0.0269662, 0.0539249, -0.0245716, 0.0509700, -0.0779363, 0.0784965
4: -0.0570398, 0.0551882, -0.0541988, 0.0513218, -0.1083617, 0.1093870
5: -0.0320548, 0.0664665, -0.0286561, 0.0631842, -0.0952390, 0.0951227
6: -0.0992120, 0.0710214, -0.0956406, 0.0674178, -0.1666298, 0.1666620
7: 0.8432010, 1.0075297, 0.8484070, 1.0056651, -0.1624641, 0.1591228
8: -0.0852043, 0.0926447, -0.0821822, 0.0889404, -0.1741447, 0.1748269
9: -0.0765021, 0.0873128, -0.0737649, 0.0815634, -0.1580655, 0.1610778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0493371, 0.0397969, -0.0392951, 0.0354807, -0.0848178, 0.0790920
1: -0.0444713, 0.0437600, -0.0359801, 0.0346554, -0.0791268, 0.0797400
2: -0.0475104, 0.1329773, -0.0365308, 0.1207362, -0.1682466, 0.1695082
3: -0.0264732, 0.0534901, -0.0199995, 0.0459889, -0.0724621, 0.0734896
4: -0.0565209, 0.0545370, -0.0487206, 0.0430802, -0.0996011, 0.1032576
5: -0.0315259, 0.0658807, -0.0219893, 0.0560612, -0.0875871, 0.0878700
6: -0.0986077, 0.0704479, -0.0878775, 0.0599037, -0.1585114, 0.1583254
7: 0.8440285, 1.0072235, 0.8598840, 1.0019219, -0.1578934, 0.1473395
8: -0.0846476, 0.0919550, -0.0763075, 0.0811774, -0.1658249, 0.1682625
9: -0.0760828, 0.0862515, -0.0678119, 0.0705186, -0.1466014, 0.1540634

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0577033, 0.0441012, -0.0537349, 0.0420595, -0.0997629, 0.0978361
1: -0.0528291, 0.0505466, -0.0488648, 0.0473275, -0.1001566, 0.0994114
2: -0.0566185, 0.1432750, -0.0522982, 0.1383905, -0.1950090, 0.1955732
3: -0.0343882, 0.0604683, -0.0306339, 0.0571584, -0.0915466, 0.0911022
4: -0.0648521, 0.0649903, -0.0609004, 0.0600319, -0.1248841, 0.1258907
5: -0.0400164, 0.0752849, -0.0359891, 0.0708242, -0.1108405, 0.1112739
6: -0.1083082, 0.0796537, -0.1037069, 0.0752871, -0.1835953, 0.1833607
7: 0.8307452, 1.0121368, 0.8370458, 1.0098065, -0.1790614, 0.1750910
8: -0.0935853, 0.1030259, -0.0893459, 0.0977746, -0.1913600, 0.1923717
9: -0.0828136, 0.1032910, -0.0796210, 0.0952085, -0.1780221, 0.1829120

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0815048, 0.0563464, -0.0370655, 0.0342780, -0.1157828, 0.0934119
1: -0.0766064, 0.0698542, -0.0345304, 0.0320523, -0.1086587, 0.1043846
2: -0.0825301, 0.1725713, -0.0334122, 0.1177580, -0.2002882, 0.2059834
3: -0.0569056, 0.0803207, -0.0185315, 0.0446335, -0.1015391, 0.0988523
4: -0.0885539, 0.0947291, -0.0466315, 0.0410344, -0.1295883, 0.1413605
5: -0.0641711, 0.1020390, -0.0206325, 0.0532744, -0.1174455, 0.1226715
6: -0.1359052, 0.1058436, -0.0853436, 0.0571011, -0.1930063, 0.1911873
7: 0.7929554, 1.0261147, 0.8643413, 1.0005935, -0.2076381, 0.1617734
8: -0.1190127, 0.1345216, -0.0740377, 0.0786548, -0.1976675, 0.2085593
9: -0.1019622, 0.1517670, -0.0655809, 0.0672089, -0.1691711, 0.2173479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.12 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0887067, 0.0600516, -0.0427503, 0.0370486, -0.1257553, 0.1028018
1: -0.0838009, 0.0756963, -0.0384767, 0.0380490, -0.1218499, 0.1141730
2: -0.0903706, 0.1814357, -0.0405966, 0.1251750, -0.2155456, 0.2220323
3: -0.0637190, 0.0863276, -0.0222876, 0.0477560, -0.1114749, 0.1086152
4: -0.0957256, 0.1037275, -0.0514442, 0.0471227, -0.1428483, 0.1551717
5: -0.0714798, 0.1101344, -0.0249790, 0.0596942, -0.1311740, 0.1351134
6: -0.1442556, 0.1137682, -0.0917682, 0.0635574, -0.2078129, 0.2055364
7: 0.7815209, 1.0303441, 0.8540350, 1.0036533, -0.2221324, 0.1763091
8: -0.1267065, 0.1440517, -0.0792665, 0.0850665, -0.2117731, 0.2233182
9: -0.1077562, 0.1664350, -0.0707204, 0.0760137, -0.1837699, 0.2371554

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0638566, 0.0472669, -0.0370655, 0.0342780, -0.0981346, 0.0843324
1: -0.0589761, 0.0555381, -0.0345304, 0.0320523, -0.0910284, 0.0900685
2: -0.0633173, 0.1508488, -0.0334122, 0.1177580, -0.1810753, 0.1842610
3: -0.0402094, 0.0656007, -0.0185315, 0.0446335, -0.0848429, 0.0841322
4: -0.0709796, 0.0726785, -0.0466315, 0.0410344, -0.1120140, 0.1193100
5: -0.0462610, 0.0822015, -0.0206325, 0.0532744, -0.0995353, 0.1028340
6: -0.1154427, 0.0864244, -0.0853436, 0.0571011, -0.1725438, 0.1717680
7: 0.8209757, 1.0157505, 0.8643413, 1.0005935, -0.1796179, 0.1514092
8: -0.1001589, 0.1111683, -0.0740377, 0.0786548, -0.1788137, 0.1852060
9: -0.0877640, 0.1158232, -0.0655809, 0.0672089, -0.1549729, 0.1814042

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0707013, 0.0507883, -0.0427503, 0.0370486, -0.1077498, 0.0935385
1: -0.0658138, 0.0610905, -0.0384767, 0.0380490, -0.1038628, 0.0995672
2: -0.0707688, 0.1592737, -0.0405966, 0.1251750, -0.1959438, 0.1998703
3: -0.0466849, 0.0713097, -0.0222876, 0.0477560, -0.0944409, 0.0935973
4: -0.0777956, 0.0812306, -0.0514442, 0.0471227, -0.1249184, 0.1326748
5: -0.0532072, 0.0898953, -0.0249790, 0.0596942, -0.1129014, 0.1148743
6: -0.1233789, 0.0939560, -0.0917682, 0.0635574, -0.1869362, 0.1857242
7: 0.8101083, 1.0197703, 0.8540350, 1.0036533, -0.1935450, 0.1657352
8: -0.1074712, 0.1202257, -0.0792665, 0.0850665, -0.1925378, 0.1994922
9: -0.0932707, 0.1297637, -0.0707204, 0.0760137, -0.1692844, 0.2004840

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0339846, 0.0326161, -0.1147429, 0.0734465, -0.1074311, 0.1473591
1: -0.0325271, 0.0284551, -0.1098107, 0.0968167, -0.1293439, 0.1382658
2: -0.0291025, 0.1136425, -0.1187151, 0.2134828, -0.2425852, 0.2323576
3: -0.0165030, 0.0427604, -0.0883506, 0.1080440, -0.1245471, 0.1311110
4: -0.0437445, 0.0382073, -0.1216527, 0.1362585, -0.1800030, 0.1598601
5: -0.0187574, 0.0494234, -0.0979025, 0.1394006, -0.1581581, 0.1473259
6: -0.0818421, 0.0532282, -0.1744438, 0.1424172, -0.2242593, 0.2276720
7: 0.8705006, 0.9987579, 0.7401829, 1.0456343, -0.1751336, 0.2585750
8: -0.0709011, 0.0751689, -0.1545213, 0.1785046, -0.2494057, 0.2296902
9: -0.0624980, 0.0626352, -0.1287028, 0.2194627, -0.2819606, 0.1913380

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 68

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0339846, 0.0326161, -0.1384956, 0.0856667, -0.1196513, 0.1711117
1: -0.0325271, 0.0284551, -0.1335392, 0.1160847, -0.1486119, 0.1619944
2: -0.0291025, 0.1136425, -0.1445738, 0.2427189, -0.2718214, 0.2582164
3: -0.0165030, 0.0427604, -0.1108219, 0.1278557, -0.1443588, 0.1535823
4: -0.0437445, 0.0382073, -0.1453060, 0.1659364, -0.2096809, 0.1835134
5: -0.0187574, 0.0494234, -0.1220077, 0.1661000, -0.1848575, 0.1714311
6: -0.0818421, 0.0532282, -0.2019843, 0.1685534, -0.2503956, 0.2552125
7: 0.8705006, 0.9987579, 0.7024704, 1.0595833, -0.1890827, 0.2962875
8: -0.0709011, 0.0751689, -0.1798966, 0.2099359, -0.2808370, 0.2550655
9: -0.0624980, 0.0626352, -0.1478122, 0.2678394, -0.3303374, 0.2104474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 68

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0390926, 0.0353715, -0.1216495, 0.0769998, -0.1160924, 0.1570210
1: -0.0358485, 0.0344190, -0.1167102, 0.1024193, -0.1382678, 0.1511293
2: -0.0362477, 0.1204658, -0.1262341, 0.2219837, -0.2582314, 0.2466999
3: -0.0198662, 0.0458658, -0.0948846, 0.1138047, -0.1336709, 0.1407505
4: -0.0485309, 0.0428945, -0.1285305, 0.1448880, -0.1934189, 0.1714249
5: -0.0218661, 0.0558081, -0.1049116, 0.1471640, -0.1690301, 0.1607197
6: -0.0876474, 0.0596492, -0.1824518, 0.1500168, -0.2376642, 0.2421010
7: 0.8602889, 1.0018009, 0.7292173, 1.0496902, -0.1894014, 0.2725836
8: -0.0761014, 0.0809484, -0.1618997, 0.1876439, -0.2637452, 0.2428481
9: -0.0676093, 0.0702181, -0.1342593, 0.2335292, -0.3011385, 0.2044774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0390926, 0.0353715, -0.1453879, 0.0892125, -0.1283051, 0.1807594
1: -0.0358485, 0.0344190, -0.1404244, 0.1216757, -0.1575241, 0.1748434
2: -0.0362477, 0.1204658, -0.1520770, 0.2512023, -0.2874500, 0.2725427
3: -0.0198662, 0.0458658, -0.1173423, 0.1336044, -0.1534706, 0.1632081
4: -0.0485309, 0.0428945, -0.1521693, 0.1745479, -0.2230788, 0.1950638
5: -0.0218661, 0.0558081, -0.1290022, 0.1738473, -0.1957134, 0.1848104
6: -0.0876474, 0.0596492, -0.2099757, 0.1761373, -0.2637846, 0.2696249
7: 0.8602889, 1.0018009, 0.6915276, 1.0636309, -0.2033421, 0.3102733
8: -0.0761014, 0.0809484, -0.1872596, 0.2190561, -0.2951575, 0.2682079
9: -0.0676093, 0.0702181, -0.1533571, 0.2818767, -0.3494860, 0.2235753

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=56, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=29, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 33

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0570411, 0.0437605, -0.0909785, 0.0612203, -0.1182615, 0.1347390
1: -0.0521676, 0.0500095, -0.0860704, 0.0775392, -0.1297068, 0.1360798
2: -0.0558976, 0.1424600, -0.0928437, 0.1842320, -0.2401296, 0.2353037
3: -0.0337617, 0.0599160, -0.0658682, 0.0882226, -0.1219842, 0.1257842
4: -0.0641927, 0.0641629, -0.0979879, 0.1065659, -0.1707586, 0.1621508
5: -0.0393443, 0.0745405, -0.0737853, 0.1126880, -0.1520323, 0.1483258
6: -0.1075403, 0.0789251, -0.1468896, 0.1162680, -0.2238083, 0.2258147
7: 0.8317966, 1.0117480, 0.7779139, 1.0316782, -0.1998816, 0.2338340
8: -0.0928779, 0.1021496, -0.1291335, 0.1470578, -0.2399357, 0.2312831
9: -0.0822808, 0.1019422, -0.1095840, 0.1710619, -0.2533427, 0.2115261

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0643588, 0.0475252, -0.1008214, 0.0662843, -0.1306431, 0.1483466
1: -0.0594778, 0.0559455, -0.0959033, 0.0855237, -0.1450015, 0.1518488
2: -0.0638639, 0.1514670, -0.1035594, 0.1963473, -0.2602112, 0.2550264
3: -0.0406846, 0.0660196, -0.0751801, 0.0964324, -0.1371170, 0.1411997
4: -0.0714798, 0.0733060, -0.1077895, 0.1188643, -0.1903440, 0.1810955
5: -0.0467707, 0.0827660, -0.0837744, 0.1237521, -0.1705227, 0.1665403
6: -0.1160250, 0.0869771, -0.1583022, 0.1270987, -0.2431237, 0.2452793
7: 0.8201783, 1.0160456, 0.7622862, 1.0374585, -0.2172802, 0.2537594
8: -0.1006955, 0.1118328, -0.1396488, 0.1600826, -0.2607781, 0.2514817
9: -0.0881680, 0.1168461, -0.1175027, 0.1911089, -0.2792769, 0.2343488

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0404796, 0.0360558, -0.0909785, 0.0612203, -0.1016999, 0.1270343
1: -0.0368041, 0.0359003, -0.0860704, 0.0775392, -0.1143433, 0.1219707
2: -0.0380223, 0.1222803, -0.0928437, 0.1842320, -0.2222543, 0.2151241
3: -0.0207822, 0.0466371, -0.0658682, 0.0882226, -0.1090047, 0.1125053
4: -0.0497198, 0.0443550, -0.0979879, 0.1065659, -0.1562857, 0.1423429
5: -0.0229013, 0.0573939, -0.0737853, 0.1126880, -0.1355893, 0.1311792
6: -0.0892158, 0.0612440, -0.1468896, 0.1162680, -0.2054838, 0.2081336
7: 0.8577442, 1.0025570, 0.7779139, 1.0316782, -0.1739340, 0.2246431
8: -0.0773930, 0.0825132, -0.1291335, 0.1470578, -0.2244508, 0.2116467
9: -0.0688789, 0.0723558, -0.1095840, 0.1710619, -0.2399408, 0.1819398

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0465125, 0.0385643, -0.1008214, 0.0662843, -0.1127968, 0.1393857
1: -0.0418369, 0.0413297, -0.0959033, 0.0855237, -0.1273606, 0.1372330
2: -0.0445495, 0.1295948, -0.1035594, 0.1963473, -0.2408968, 0.2331542
3: -0.0245862, 0.0509914, -0.0751801, 0.0964324, -0.1210186, 0.1261715
4: -0.0542165, 0.0513486, -0.1077895, 0.1188643, -0.1730808, 0.1591382
5: -0.0286799, 0.0632065, -0.0837744, 0.1237521, -0.1524320, 0.1469809
6: -0.0956653, 0.0674427, -0.1583022, 0.1270987, -0.2227640, 0.2257449
7: 0.8483711, 1.0056781, 0.7622862, 1.0374585, -0.1890874, 0.2433919
8: -0.0822009, 0.0889652, -0.1396488, 0.1600826, -0.2422835, 0.2286140
9: -0.0737844, 0.0815988, -0.1175027, 0.1911089, -0.2648934, 0.1991015

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0520610, 0.0411983, -0.1083370, 0.0701509, -0.1222119, 0.1495353
1: -0.0471925, 0.0459696, -0.1034112, 0.0916203, -0.1388128, 0.1493808
2: -0.0504759, 0.1363300, -0.1117412, 0.2055979, -0.2560738, 0.2480712
3: -0.0290502, 0.0557621, -0.0822902, 0.1027010, -0.1317512, 0.1380523
4: -0.0592334, 0.0579405, -0.1152737, 0.1282546, -0.1874880, 0.1732142
5: -0.0342903, 0.0689425, -0.0914015, 0.1322000, -0.1664903, 0.1603440
6: -0.1017660, 0.0734452, -0.1670162, 0.1353684, -0.2371344, 0.2404614
7: 0.8397037, 1.0088235, 0.7503536, 1.0418723, -0.2021686, 0.2584699
8: -0.0875575, 0.0955596, -0.1476779, 0.1700278, -0.2575853, 0.2432375
9: -0.0782742, 0.0917992, -0.1235491, 0.2064158, -0.2846900, 0.2153484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0503015, 0.0402931, -0.1176037, 0.0749183, -0.1252198, 0.1578968
1: -0.0454348, 0.0445423, -0.1126685, 0.0991374, -0.1445721, 0.1572108
2: -0.0485603, 0.1341643, -0.1218296, 0.2170039, -0.2655641, 0.2559939
3: -0.0273856, 0.0542946, -0.0910570, 0.1104302, -0.1378158, 0.1453516
4: -0.0574813, 0.0557420, -0.1245016, 0.1398329, -0.1973141, 0.1802436
5: -0.0325047, 0.0669647, -0.1008057, 0.1426163, -0.1751209, 0.1677704
6: -0.0997259, 0.0715091, -0.1777608, 0.1455650, -0.2452909, 0.2492698
7: 0.8424975, 1.0077900, 0.7356408, 1.0473142, -0.2048167, 0.2721492
8: -0.0856778, 0.0932312, -0.1575774, 0.1822902, -0.2679680, 0.2508086
9: -0.0768587, 0.0882157, -0.1310044, 0.2252891, -0.3021478, 0.2192201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.06 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0597667, 0.0451627, -0.1153692, 0.0737688, -0.1335355, 0.1605320
1: -0.0548904, 0.0522204, -0.1104364, 0.0973248, -0.1522152, 0.1626568
2: -0.0588648, 0.1458147, -0.1193970, 0.2142537, -0.2731185, 0.2652116
3: -0.0363402, 0.0621893, -0.0889432, 0.1085665, -0.1449067, 0.1511325
4: -0.0669068, 0.0675683, -0.1222766, 0.1370411, -0.2039479, 0.1898449
5: -0.0421104, 0.0776042, -0.0985382, 0.1401046, -0.1822150, 0.1761423
6: -0.1107005, 0.0819242, -0.1751700, 0.1431064, -0.2538069, 0.2570942
7: 0.8274692, 1.0133487, 0.7391883, 1.0460019, -0.2185327, 0.2741604
8: -0.0957896, 0.1057563, -0.1551905, 0.1793334, -0.2751230, 0.2609468
9: -0.0844736, 0.1074934, -0.1292068, 0.2207384, -0.3052120, 0.2367001

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 255

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0579853, 0.0442463, -0.1244952, 0.0784638, -0.1364492, 0.1687415
1: -0.0531108, 0.0507754, -0.1195531, 0.1047277, -0.1578386, 0.1703285
2: -0.0569255, 0.1436222, -0.1293321, 0.2254864, -0.2824119, 0.2729542
3: -0.0346550, 0.0607036, -0.0975768, 0.1161783, -0.1508333, 0.1582803
4: -0.0651330, 0.0653426, -0.1313642, 0.1484436, -0.2135766, 0.1967069
5: -0.0403026, 0.0756018, -0.1077996, 0.1503628, -0.1906653, 0.1834014
6: -0.1086351, 0.0799641, -0.1857513, 0.1531481, -0.2617832, 0.2657154
7: 0.8302975, 1.0123026, 0.7246991, 1.0513614, -0.2210640, 0.2876035
8: -0.0938866, 0.1033990, -0.1649398, 0.1914094, -0.2852961, 0.2683389
9: -0.0830405, 0.1038653, -0.1365487, 0.2393251, -0.3223656, 0.2404140

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0815048, 0.0563464, -0.0893545, 0.0603848, -0.1418896, 0.1457009
1: -0.0766064, 0.0698542, -0.0844481, 0.0762218, -0.1528282, 0.1543023
2: -0.0825301, 0.1725713, -0.0910757, 0.1822331, -0.2647632, 0.2636470
3: -0.0569056, 0.0803207, -0.0643318, 0.0868680, -0.1437736, 0.1446525
4: -0.0885539, 0.0947291, -0.0963707, 0.1045369, -0.1930908, 0.1910998
5: -0.0641711, 0.1020390, -0.0721373, 0.1108625, -0.1750336, 0.1741763
6: -0.1359052, 0.1058436, -0.1450067, 0.1144810, -0.2503862, 0.2508503
7: 0.7929554, 1.0261147, 0.7804925, 1.0307244, -0.2377690, 0.2456222
8: -0.1190127, 0.1345216, -0.1273986, 0.1449088, -0.2639215, 0.2619202
9: -0.1019622, 0.1517670, -0.1082775, 0.1677544, -0.2697166, 0.2600444

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 33

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0887067, 0.0600516, -0.0990658, 0.0653810, -0.1540877, 0.1591173
1: -0.0838009, 0.0756963, -0.0941495, 0.0840995, -0.1679004, 0.1698458
2: -0.0903706, 0.1814357, -0.1016481, 0.1941863, -0.2845569, 0.2830838
3: -0.0637190, 0.0863276, -0.0735192, 0.0949680, -0.1586870, 0.1598468
4: -0.0957256, 0.1037275, -0.1060413, 0.1166707, -0.2123963, 0.2097688
5: -0.0714798, 0.1101344, -0.0819927, 0.1217786, -0.1932585, 0.1921271
6: -0.1442556, 0.1137682, -0.1562666, 0.1251668, -0.2694224, 0.2700348
7: 0.7815209, 1.0303441, 0.7650736, 1.0364275, -0.2549066, 0.2652705
8: -0.1267065, 0.1440517, -0.1377732, 0.1577595, -0.2844660, 0.2818249
9: -0.1077562, 0.1664350, -0.1160903, 0.1875332, -0.2952895, 0.2825254

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0638566, 0.0472669, -0.0893545, 0.0603848, -0.1242414, 0.1366214
1: -0.0589761, 0.0555381, -0.0844481, 0.0762218, -0.1351979, 0.1399862
2: -0.0633173, 0.1508488, -0.0910757, 0.1822331, -0.2455504, 0.2419245
3: -0.0402094, 0.0656007, -0.0643318, 0.0868680, -0.1270775, 0.1299325
4: -0.0709796, 0.0726785, -0.0963707, 0.1045369, -0.1755165, 0.1690492
5: -0.0462610, 0.0822015, -0.0721373, 0.1108625, -0.1571235, 0.1543388
6: -0.1154427, 0.0864244, -0.1450067, 0.1144810, -0.2299237, 0.2314311
7: 0.8209757, 1.0157505, 0.7804925, 1.0307244, -0.2097487, 0.2352580
8: -0.1001589, 0.1111683, -0.1273986, 0.1449088, -0.2450677, 0.2385668
9: -0.0877640, 0.1158232, -0.1082775, 0.1677544, -0.2555184, 0.2241007

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.18 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0707013, 0.0507883, -0.0990658, 0.0653810, -0.1360823, 0.1498541
1: -0.0658138, 0.0610905, -0.0941495, 0.0840995, -0.1499134, 0.1552399
2: -0.0707688, 0.1592737, -0.1016481, 0.1941863, -0.2649551, 0.2609218
3: -0.0466849, 0.0713097, -0.0735192, 0.0949680, -0.1416529, 0.1448289
4: -0.0777956, 0.0812306, -0.1060413, 0.1166707, -0.1944664, 0.1872718
5: -0.0532072, 0.0898953, -0.0819927, 0.1217786, -0.1749859, 0.1718880
6: -0.1233789, 0.0939560, -0.1562666, 0.1251668, -0.2485457, 0.2502226
7: 0.8101083, 1.0197703, 0.7650736, 1.0364275, -0.2263192, 0.2546967
8: -0.1074712, 0.1202257, -0.1377732, 0.1577595, -0.2652307, 0.2579989
9: -0.0932707, 0.1297637, -0.1160903, 0.1875332, -0.2808039, 0.2458540

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1147429, 0.0734465, -0.0339846, 0.0326161, -0.1473591, 0.1074311
1: -0.1098107, 0.0968167, -0.0325271, 0.0284551, -0.1382658, 0.1293439
2: -0.1187151, 0.2134828, -0.0291025, 0.1136425, -0.2323576, 0.2425852
3: -0.0883506, 0.1080440, -0.0165030, 0.0427604, -0.1311110, 0.1245471
4: -0.1216527, 0.1362585, -0.0437445, 0.0382073, -0.1598601, 0.1800030
5: -0.0979025, 0.1394006, -0.0187574, 0.0494234, -0.1473259, 0.1581581
6: -0.1744438, 0.1424172, -0.0818421, 0.0532282, -0.2276720, 0.2242593
7: 0.7401829, 1.0456343, 0.8705006, 0.9987579, -0.2585750, 0.1751336
8: -0.1545213, 0.1785046, -0.0709011, 0.0751689, -0.2296902, 0.2494057
9: -0.1287028, 0.2194627, -0.0624980, 0.0626352, -0.1913380, 0.2819606

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 75

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.04 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1147429, 0.0734465, -0.0717061, 0.0529639, -0.1677068, 0.1451526
1: -0.1098107, 0.0968167, -0.0570540, 0.0724969, -0.1823076, 0.1538708
2: -0.1187151, 0.2134828, -0.0818675, 0.1640305, -0.2827456, 0.2953503
3: -0.0883506, 0.1080440, -0.0413387, 0.0656932, -0.1540439, 0.1493828
4: -0.1216527, 0.1362585, -0.0790909, 0.0728204, -0.1944731, 0.2153494
5: -0.0979025, 0.1394006, -0.0417140, 0.0965727, -0.1944752, 0.1811146
6: -0.1744438, 0.1424172, -0.1247124, 0.1006455, -0.2750893, 0.2671296
7: 0.7401829, 1.0456343, 0.7950890, 1.0212306, -0.2810476, 0.2505453
8: -0.1545213, 0.1785046, -0.1093037, 0.1178484, -0.2723698, 0.2878083
9: -0.1287028, 0.2194627, -0.1002441, 0.1186330, -0.2473358, 0.3197068

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 75

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.97 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1384956, 0.0856667, -0.0339846, 0.0326161, -0.1711117, 0.1196513
1: -0.1335392, 0.1160847, -0.0325271, 0.0284551, -0.1619944, 0.1486119
2: -0.1445738, 0.2427189, -0.0291025, 0.1136425, -0.2582164, 0.2718214
3: -0.1108219, 0.1278557, -0.0165030, 0.0427604, -0.1535823, 0.1443588
4: -0.1453060, 0.1659364, -0.0437445, 0.0382073, -0.1835134, 0.2096809
5: -0.1220077, 0.1661000, -0.0187574, 0.0494234, -0.1714311, 0.1848575
6: -0.2019843, 0.1685534, -0.0818421, 0.0532282, -0.2552125, 0.2503956
7: 0.7024704, 1.0595833, 0.8705006, 0.9987579, -0.2962875, 0.1890827
8: -0.1798966, 0.2099359, -0.0709011, 0.0751689, -0.2550655, 0.2808370
9: -0.1478122, 0.2678394, -0.0624980, 0.0626352, -0.2104474, 0.3303374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 75

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.99 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.06 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1384956, 0.0856667, -0.0717061, 0.0529639, -0.1914595, 0.1573728
1: -0.1335392, 0.1160847, -0.0570540, 0.0724969, -0.2060361, 0.1731388
2: -0.1445738, 0.2427189, -0.0818675, 0.1640305, -0.3086043, 0.3245865
3: -0.1108219, 0.1278557, -0.0413387, 0.0656932, -0.1765152, 0.1691945
4: -0.1453060, 0.1659364, -0.0790909, 0.0728204, -0.2181264, 0.2450272
5: -0.1220077, 0.1661000, -0.0417140, 0.0965727, -0.2185805, 0.2078141
6: -0.2019843, 0.1685534, -0.1247124, 0.1006455, -0.3026298, 0.2932659
7: 0.7024704, 1.0595833, 0.7950890, 1.0212306, -0.3187602, 0.2644943
8: -0.1798966, 0.2099359, -0.1093037, 0.1178484, -0.2977450, 0.3192396
9: -0.1478122, 0.2678394, -0.1002441, 0.1186330, -0.2664452, 0.3680835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 75

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1216495, 0.0769998, -0.0390926, 0.0353715, -0.1570210, 0.1160924
1: -0.1167102, 0.1024193, -0.0358485, 0.0344190, -0.1511293, 0.1382678
2: -0.1262341, 0.2219837, -0.0362477, 0.1204658, -0.2466999, 0.2582314
3: -0.0948846, 0.1138047, -0.0198662, 0.0458658, -0.1407505, 0.1336709
4: -0.1285305, 0.1448880, -0.0485309, 0.0428945, -0.1714249, 0.1934189
5: -0.1049116, 0.1471640, -0.0218661, 0.0558081, -0.1607197, 0.1690301
6: -0.1824518, 0.1500168, -0.0876474, 0.0596492, -0.2421010, 0.2376642
7: 0.7292173, 1.0496902, 0.8602889, 1.0018009, -0.2725836, 0.1894014
8: -0.1618997, 0.1876439, -0.0761014, 0.0809484, -0.2428481, 0.2637452
9: -0.1342593, 0.2335292, -0.0676093, 0.0702181, -0.2044774, 0.3011385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 75

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1216495, 0.0769998, -0.0856271, 0.0557935, -0.1774430, 0.1626270
1: -0.1167102, 0.1024193, -0.0700596, 0.0786214, -0.1953316, 0.1724789
2: -0.1262341, 0.2219837, -0.0892052, 0.1798335, -0.3060676, 0.3111889
3: -0.0948846, 0.1138047, -0.0507140, 0.0688824, -0.1637670, 0.1645187
4: -0.1285305, 0.1448880, -0.0840062, 0.0993836, -0.2279141, 0.2288942
5: -0.1049116, 0.1471640, -0.0642105, 0.1031295, -0.2080411, 0.2113745
6: -0.1824518, 0.1500168, -0.1399629, 0.1072394, -0.2896912, 0.2899797
7: 0.7292173, 1.0496902, 0.7839932, 1.0243555, -0.2951382, 0.2656970
8: -0.1618997, 0.1876439, -0.1146441, 0.1332800, -0.2951797, 0.3022879
9: -0.1342593, 0.2335292, -0.1054932, 0.1450833, -0.2793426, 0.3390224

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 75

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.1453879, 0.0892125, -0.0390926, 0.0353715, -0.1807594, 0.1283051
1: -0.1404244, 0.1216757, -0.0358485, 0.0344190, -0.1748434, 0.1575241
2: -0.1520770, 0.2512023, -0.0362477, 0.1204658, -0.2725427, 0.2874500
3: -0.1173423, 0.1336044, -0.0198662, 0.0458658, -0.1632081, 0.1534706
4: -0.1521693, 0.1745479, -0.0485309, 0.0428945, -0.1950638, 0.2230788
5: -0.1290022, 0.1738473, -0.0218661, 0.0558081, -0.1848104, 0.1957134
6: -0.2099757, 0.1761373, -0.0876474, 0.0596492, -0.2696249, 0.2637846
7: 0.6915276, 1.0636309, 0.8602889, 1.0018009, -0.3102733, 0.2033421
8: -0.1872596, 0.2190561, -0.0761014, 0.0809484, -0.2682079, 0.2951575
9: -0.1533571, 0.2818767, -0.0676093, 0.0702181, -0.2235753, 0.3494860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 75

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1453879, 0.0892125, -0.0856271, 0.0557935, -0.2011814, 0.1748396
1: -0.1404244, 0.1216757, -0.0700596, 0.0786214, -0.2190458, 0.1917352
2: -0.1520770, 0.2512023, -0.0892052, 0.1798335, -0.3319105, 0.3404075
3: -0.1173423, 0.1336044, -0.0507140, 0.0688824, -0.1862246, 0.1843185
4: -0.1521693, 0.1745479, -0.0840062, 0.0993836, -0.2515529, 0.2585541
5: -0.1290022, 0.1738473, -0.0642105, 0.1031295, -0.2321317, 0.2380578
6: -0.2099757, 0.1761373, -0.1399629, 0.1072394, -0.3172151, 0.3161001
7: 0.6915276, 1.0636309, 0.7839932, 1.0243555, -0.3328279, 0.2796377
8: -0.1872596, 0.2190561, -0.1146441, 0.1332800, -0.3205396, 0.3337001
9: -0.1533571, 0.2818767, -0.1054932, 0.1450833, -0.2984405, 0.3873699

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 75

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0909785, 0.0612203, -0.0570411, 0.0437605, -0.1347390, 0.1182615
1: -0.0860704, 0.0775392, -0.0521676, 0.0500095, -0.1360798, 0.1297068
2: -0.0928437, 0.1842320, -0.0558976, 0.1424600, -0.2353037, 0.2401296
3: -0.0658682, 0.0882226, -0.0337617, 0.0599160, -0.1257842, 0.1219842
4: -0.0979879, 0.1065659, -0.0641927, 0.0641629, -0.1621508, 0.1707586
5: -0.0737853, 0.1126880, -0.0393443, 0.0745405, -0.1483258, 0.1520323
6: -0.1468896, 0.1162680, -0.1075403, 0.0789251, -0.2258147, 0.2238083
7: 0.7779139, 1.0316782, 0.8317966, 1.0117480, -0.2338340, 0.1998816
8: -0.1291335, 0.1470578, -0.0928779, 0.1021496, -0.2312831, 0.2399357
9: -0.1095840, 0.1710619, -0.0822808, 0.1019422, -0.2115261, 0.2533427

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 68

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 91

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.90 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0909785, 0.0612203, -0.1147429, 0.0734465, -0.1644250, 0.1759633
1: -0.0860704, 0.0775392, -0.1098107, 0.0968167, -0.1828871, 0.1873499
2: -0.0928437, 0.1842320, -0.1187151, 0.2134826, -0.3063264, 0.3029471
3: -0.0658682, 0.0882226, -0.0883506, 0.1080441, -0.1739123, 0.1765732
4: -0.0979879, 0.1065659, -0.1216528, 0.1362585, -0.2342464, 0.2282187
5: -0.0737853, 0.1126880, -0.0979025, 0.1394006, -0.2131860, 0.2105905
6: -0.1468896, 0.1162680, -0.1744438, 0.1424172, -0.2893068, 0.2907118
7: 0.7779139, 1.0316782, 0.7401828, 1.0456343, -0.2677203, 0.2914954
8: -0.1291335, 0.1470578, -0.1545213, 0.1785046, -0.3076381, 0.3015791
9: -0.1095840, 0.1710619, -0.1287028, 0.2194627, -0.3290466, 0.2997648

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 68

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 91

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.94 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.09 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1008214, 0.0662843, -0.0643588, 0.0475252, -0.1483466, 0.1306431
1: -0.0959033, 0.0855237, -0.0594778, 0.0559455, -0.1518488, 0.1450015
2: -0.1035594, 0.1963473, -0.0638639, 0.1514670, -0.2550264, 0.2602112
3: -0.0751801, 0.0964324, -0.0406846, 0.0660196, -0.1411997, 0.1371170
4: -0.1077895, 0.1188643, -0.0714798, 0.0733060, -0.1810955, 0.1903440
5: -0.0837744, 0.1237521, -0.0467707, 0.0827660, -0.1665403, 0.1705227
6: -0.1583022, 0.1270987, -0.1160250, 0.0869771, -0.2452793, 0.2431237
7: 0.7622862, 1.0374585, 0.8201783, 1.0160456, -0.2537594, 0.2172802
8: -0.1396488, 0.1600826, -0.1006955, 0.1118328, -0.2514817, 0.2607781
9: -0.1175027, 0.1911089, -0.0881680, 0.1168461, -0.2343488, 0.2792769

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 75

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 91

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.94 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1008214, 0.0662843, -0.1216495, 0.0769998, -0.1778212, 0.1879338
1: -0.0959033, 0.0855237, -0.1167102, 0.1024194, -0.1983227, 0.2022339
2: -0.1035594, 0.1963473, -0.1262341, 0.2219838, -0.3255432, 0.3225814
3: -0.0751801, 0.0964324, -0.0948846, 0.1138047, -0.1889848, 0.1913170
4: -0.1077895, 0.1188643, -0.1285304, 0.1448879, -0.2526775, 0.2473947
5: -0.0837744, 0.1237521, -0.1049116, 0.1471640, -0.2309384, 0.2286637
6: -0.1583022, 0.1270987, -0.1824518, 0.1500168, -0.3083190, 0.3095505
7: 0.7622862, 1.0374585, 0.7292173, 1.0496904, -0.2874042, 0.3082412
8: -0.1396488, 0.1600826, -0.1618997, 0.1876439, -0.3272927, 0.3219824
9: -0.1175027, 0.1911089, -0.1342593, 0.2335292, -0.3510319, 0.3253682

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 75

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 91

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.95 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0909785, 0.0612203, -0.0404796, 0.0360558, -0.1270343, 0.1016999
1: -0.0860704, 0.0775392, -0.0368041, 0.0359003, -0.1219707, 0.1143433
2: -0.0928437, 0.1842320, -0.0380223, 0.1222803, -0.2151241, 0.2222543
3: -0.0658682, 0.0882226, -0.0207822, 0.0466371, -0.1125053, 0.1090047
4: -0.0979879, 0.1065659, -0.0497198, 0.0443550, -0.1423429, 0.1562857
5: -0.0737853, 0.1126880, -0.0229013, 0.0573939, -0.1311792, 0.1355893
6: -0.1468896, 0.1162680, -0.0892158, 0.0612440, -0.2081336, 0.2054838
7: 0.7779139, 1.0316782, 0.8577442, 1.0025570, -0.2246431, 0.1739340
8: -0.1291335, 0.1470578, -0.0773930, 0.0825132, -0.2116467, 0.2244508
9: -0.1095840, 0.1710619, -0.0688789, 0.0723558, -0.1819398, 0.2399408

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 68

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 91

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.96 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0909785, 0.0612203, -0.0916359, 0.0566159, -0.1475944, 0.1528563
1: -0.0860704, 0.0775392, -0.0827224, 0.0804015, -0.1664719, 0.1602616
2: -0.0928437, 0.1842320, -0.0916491, 0.1822316, -0.2750754, 0.2758811
3: -0.0658682, 0.0882226, -0.0519613, 0.0911657, -0.1570338, 0.1401838
4: -0.0979879, 0.1065659, -0.0873828, 0.1016765, -0.1996644, 0.1939487
5: -0.0737853, 0.1126880, -0.0733248, 0.1050352, -0.1788206, 0.1860128
6: -0.1468896, 0.1162680, -0.1420774, 0.1140937, -0.2609833, 0.2583454
7: 0.7779139, 1.0316782, 0.7809203, 1.0301683, -0.2522544, 0.2507579
8: -0.1291335, 0.1470578, -0.1172260, 0.1353953, -0.2645288, 0.2642838
9: -0.1095840, 0.1710619, -0.1105471, 0.1481138, -0.2576977, 0.2816090

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 68

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 91

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.1008214, 0.0662843, -0.0465125, 0.0385643, -0.1393857, 0.1127968
1: -0.0959033, 0.0855237, -0.0418369, 0.0413297, -0.1372330, 0.1273606
2: -0.1035594, 0.1963473, -0.0445495, 0.1295948, -0.2331542, 0.2408968
3: -0.0751801, 0.0964324, -0.0245862, 0.0509914, -0.1261715, 0.1210186
4: -0.1077895, 0.1188643, -0.0542165, 0.0513486, -0.1591382, 0.1730808
5: -0.0837744, 0.1237521, -0.0286799, 0.0632065, -0.1469809, 0.1524320
6: -0.1583022, 0.1270987, -0.0956653, 0.0674427, -0.2257449, 0.2227640
7: 0.7622862, 1.0374585, 0.8483711, 1.0056781, -0.2433919, 0.1890874
8: -0.1396488, 0.1600826, -0.0822009, 0.0889652, -0.2286140, 0.2422835
9: -0.1175027, 0.1911089, -0.0737844, 0.0815988, -0.1991015, 0.2648934

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 75

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 91

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1008214, 0.0662843, -0.1031271, 0.0674705, -0.1682918, 0.1694113
1: -0.0959033, 0.0855237, -0.0982066, 0.0873940, -0.1832973, 0.1837303
2: -0.1035594, 0.1963473, -0.1060694, 0.1991852, -0.3027446, 0.3024167
3: -0.0751801, 0.0964324, -0.0773614, 0.0983554, -0.1735356, 0.1737937
4: -0.1077895, 0.1188643, -0.1100856, 0.1217451, -0.2295346, 0.2289499
5: -0.0837744, 0.1237521, -0.0861143, 0.1263437, -0.2101181, 0.2098663
6: -0.1583022, 0.1270987, -0.1609756, 0.1296357, -0.2879379, 0.2880743
7: 0.7622862, 1.0374585, 0.7586255, 1.0388125, -0.2765263, 0.2788330
8: -0.1396488, 0.1600826, -0.1421120, 0.1631337, -0.3027825, 0.3021946
9: -0.1175027, 0.1911089, -0.1193577, 0.1958048, -0.3133075, 0.3104666

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 75

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 91

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.96 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1083370, 0.0701509, -0.0520610, 0.0411983, -0.1495353, 0.1222119
1: -0.1034112, 0.0916203, -0.0471925, 0.0459696, -0.1493808, 0.1388128
2: -0.1117412, 0.2055979, -0.0504759, 0.1363300, -0.2480712, 0.2560738
3: -0.0822902, 0.1027010, -0.0290502, 0.0557621, -0.1380523, 0.1317512
4: -0.1152737, 0.1282546, -0.0592334, 0.0579405, -0.1732142, 0.1874880
5: -0.0914015, 0.1322000, -0.0342903, 0.0689425, -0.1603440, 0.1664903
6: -0.1670162, 0.1353684, -0.1017660, 0.0734452, -0.2404614, 0.2371344
7: 0.7503536, 1.0418723, 0.8397037, 1.0088235, -0.2584699, 0.2021686
8: -0.1476779, 0.1700278, -0.0875575, 0.0955596, -0.2432375, 0.2575853
9: -0.1235491, 0.2064158, -0.0782742, 0.0917992, -0.2153484, 0.2846900

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 75

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529614
time: 1.18 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.09 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1083370, 0.0701509, -0.1061754, 0.0690387, -0.1773757, 0.1763262
1: -0.1034112, 0.0916203, -0.1012518, 0.0898668, -0.1932780, 0.1928721
2: -0.1117412, 0.2055979, -0.1093879, 0.2029372, -0.3146784, 0.3149858
3: -0.0822902, 0.1027010, -0.0802452, 0.1008980, -0.1831882, 0.1829462
4: -0.1152737, 0.1282546, -0.1131211, 0.1255537, -0.2408274, 0.2413756
5: -0.0914015, 0.1322000, -0.0892078, 0.1297701, -0.2211716, 0.2214077
6: -0.1670162, 0.1353684, -0.1645100, 0.1329898, -0.3000061, 0.2998784
7: 0.7503536, 1.0418723, 0.7537858, 1.0406027, -0.2902490, 0.2880865
8: -0.1476779, 0.1700278, -0.1453685, 0.1671673, -0.3148452, 0.3153962
9: -0.1235491, 0.2064158, -0.1218101, 0.2020132, -0.3255623, 0.3282259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=54, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 75

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529614
time: 0.97 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.15 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1176037, 0.0749183, -0.0503015, 0.0402931, -0.1578968, 0.1252198
1: -0.1126685, 0.0991374, -0.0454348, 0.0445423, -0.1572108, 0.1445721
2: -0.1218296, 0.2170039, -0.0485603, 0.1341643, -0.2559939, 0.2655641
3: -0.0910570, 0.1104302, -0.0273856, 0.0542946, -0.1453516, 0.1378158
4: -0.1245016, 0.1398329, -0.0574813, 0.0557420, -0.1802436, 0.1973141
5: -0.1008057, 0.1426163, -0.0325047, 0.0669647, -0.1677704, 0.1751209
6: -0.1777608, 0.1455650, -0.0997259, 0.0715091, -0.2492698, 0.2452909
7: 0.7356408, 1.0473142, 0.8424975, 1.0077900, -0.2721492, 0.2048167
8: -0.1575774, 0.1822902, -0.0856778, 0.0932312, -0.2508086, 0.2679680
9: -0.1310044, 0.2252891, -0.0768587, 0.0882157, -0.2192201, 0.3021478

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 75

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529043, upper bound: 0.1528499
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.99 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1176037, 0.0749183, -0.1044018, 0.0681263, -0.1857300, 0.1793201
1: -0.1126685, 0.0991374, -0.0994801, 0.0884281, -0.2010966, 0.1986175
2: -0.1218296, 0.2170039, -0.1074572, 0.2007543, -0.3225839, 0.3244610
3: -0.0910570, 0.1104302, -0.0785674, 0.0994188, -0.1904758, 0.1889976
4: -0.1245016, 0.1398329, -0.1113550, 0.1233378, -0.2478394, 0.2511879
5: -0.1008057, 0.1426163, -0.0874079, 0.1277767, -0.2285824, 0.2300242
6: -0.1777608, 0.1455650, -0.1624537, 0.1310384, -0.3087991, 0.3080187
7: 0.7356408, 1.0473142, 0.7566016, 1.0395614, -0.3039206, 0.2907126
8: -0.1575774, 0.1822902, -0.1434738, 0.1648205, -0.3223980, 0.3257640
9: -0.1310044, 0.2252891, -0.1203833, 0.1984011, -0.3294055, 0.3456724

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=54, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 75

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529043, upper bound: 0.1528499
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.23 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1153692, 0.0737688, -0.0597667, 0.0451627, -0.1605320, 0.1335355
1: -0.1104364, 0.0973248, -0.0548904, 0.0522204, -0.1626568, 0.1522152
2: -0.1193970, 0.2142537, -0.0588648, 0.1458147, -0.2652116, 0.2731185
3: -0.0889432, 0.1085665, -0.0363402, 0.0621893, -0.1511325, 0.1449067
4: -0.1222766, 0.1370411, -0.0669068, 0.0675683, -0.1898449, 0.2039479
5: -0.0985382, 0.1401046, -0.0421104, 0.0776042, -0.1761423, 0.1822150
6: -0.1751700, 0.1431064, -0.1107005, 0.0819242, -0.2570942, 0.2538069
7: 0.7391883, 1.0460019, 0.8274692, 1.0133487, -0.2741604, 0.2185327
8: -0.1551905, 0.1793334, -0.0957896, 0.1057563, -0.2609468, 0.2751230
9: -0.1292068, 0.2207384, -0.0844736, 0.1074934, -0.2367001, 0.3052120

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 75

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.04 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.67 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1153692, 0.0737688, -0.1132846, 0.0726963, -0.1880655, 0.1870534
1: -0.1104364, 0.0973248, -0.1083538, 0.0956338, -0.2060702, 0.2056787
2: -0.1193970, 0.2142537, -0.1171275, 0.2116878, -0.3310847, 0.3313812
3: -0.0889432, 0.1085665, -0.0869710, 0.1068277, -0.1957709, 0.1955375
4: -0.1222766, 0.1370411, -0.1202006, 0.1344364, -0.2567130, 0.2572417
5: -0.0985382, 0.1401046, -0.0964226, 0.1377614, -0.2362996, 0.2365272
6: -0.1751700, 0.1431064, -0.1727529, 0.1408126, -0.3159826, 0.3158593
7: 0.7391883, 1.0460019, 0.7424983, 1.0447780, -0.3055897, 0.3035036
8: -0.1551905, 0.1793334, -0.1529634, 0.1765749, -0.3317654, 0.3322968
9: -0.1292068, 0.2207384, -0.1275297, 0.2164926, -0.3456993, 0.3482680

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 75

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.01 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.96 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.1244952, 0.0784638, -0.0579853, 0.0442463, -0.1687415, 0.1364492
1: -0.1195531, 0.1047277, -0.0531108, 0.0507754, -0.1703285, 0.1578386
2: -0.1293321, 0.2254864, -0.0569255, 0.1436222, -0.2729542, 0.2824119
3: -0.0975768, 0.1161783, -0.0346550, 0.0607036, -0.1582803, 0.1508333
4: -0.1313642, 0.1484436, -0.0651330, 0.0653426, -0.1967069, 0.2135766
5: -0.1077996, 0.1503628, -0.0403026, 0.0756018, -0.1834014, 0.1906653
6: -0.1857513, 0.1531481, -0.1086351, 0.0799641, -0.2657154, 0.2617832
7: 0.7246991, 1.0513614, 0.8302975, 1.0123026, -0.2876035, 0.2210640
8: -0.1649398, 0.1914094, -0.0938866, 0.1033990, -0.2683389, 0.2852961
9: -0.1365487, 0.2393251, -0.0830405, 0.1038653, -0.2404140, 0.3223656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 75

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529576, upper bound: 0.1529751
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1244952, 0.0784638, -0.1114451, 0.0717499, -0.1962451, 0.1899089
1: -0.1195531, 0.1047277, -0.1065162, 0.0941416, -0.2136946, 0.2112439
2: -0.1293321, 0.2254864, -0.1151249, 0.2094235, -0.3387556, 0.3406113
3: -0.0975768, 0.1161783, -0.0852307, 0.1052934, -0.2028701, 0.2014090
4: -0.1313642, 0.1484436, -0.1183688, 0.1321380, -0.2635023, 0.2668124
5: -0.1077996, 0.1503628, -0.0945558, 0.1356937, -0.2434933, 0.2449185
6: -0.1857513, 0.1531481, -0.1706201, 0.1387884, -0.3245397, 0.3237683
7: 0.7246991, 1.0513614, 0.7454189, 1.0436976, -0.3189985, 0.3059425
8: -0.1649398, 0.1914094, -0.1509982, 0.1741407, -0.3390805, 0.3424076
9: -0.1365487, 0.2393251, -0.1260497, 0.2127460, -0.3492947, 0.3653748

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 75

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529576, upper bound: 0.1529751
time: 1.08 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0893545, 0.0603848, -0.0815048, 0.0563464, -0.1457009, 0.1418896
1: -0.0844481, 0.0762218, -0.0766064, 0.0698542, -0.1543023, 0.1528282
2: -0.0910757, 0.1822331, -0.0825301, 0.1725713, -0.2636470, 0.2647632
3: -0.0643318, 0.0868680, -0.0569056, 0.0803207, -0.1446525, 0.1437736
4: -0.0963707, 0.1045369, -0.0885539, 0.0947291, -0.1910998, 0.1930908
5: -0.0721373, 0.1108625, -0.0641711, 0.1020390, -0.1741763, 0.1750336
6: -0.1450067, 0.1144810, -0.1359052, 0.1058436, -0.2508503, 0.2503862
7: 0.7804925, 1.0307244, 0.7929554, 1.0261147, -0.2456222, 0.2377690
8: -0.1273986, 0.1449088, -0.1190127, 0.1345216, -0.2619202, 0.2639215
9: -0.1082775, 0.1677544, -0.1019622, 0.1517670, -0.2600444, 0.2697166

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 68

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.04 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0893545, 0.0603848, -0.1374799, 0.0851441, -0.1744986, 0.1978647
1: -0.0844481, 0.0762218, -0.1325245, 0.1152608, -0.1997088, 0.2087463
2: -0.0910757, 0.1822331, -0.1434679, 0.2414687, -0.3325444, 0.3257011
3: -0.0643318, 0.0868680, -0.1098609, 0.1270085, -0.1913403, 0.1967289
4: -0.0963707, 0.1045369, -0.1442945, 0.1646672, -0.2610379, 0.2488314
5: -0.0721373, 0.1108625, -0.1209769, 0.1649583, -0.2370955, 0.2318394
6: -0.1450067, 0.1144810, -0.2008066, 0.1674357, -0.3124424, 0.3152876
7: 0.7804925, 1.0307244, 0.7040833, 1.0589868, -0.2784942, 0.3266411
8: -0.1273986, 0.1449088, -0.1788114, 0.2085917, -0.3359903, 0.3237202
9: -0.1082775, 0.1677544, -0.1469951, 0.2657706, -0.3740481, 0.3147495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 68

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.21 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0990658, 0.0653810, -0.0887067, 0.0600516, -0.1591173, 0.1540877
1: -0.0941495, 0.0840995, -0.0838009, 0.0756963, -0.1698458, 0.1679004
2: -0.1016481, 0.1941863, -0.0903706, 0.1814357, -0.2830838, 0.2845569
3: -0.0735192, 0.0949680, -0.0637190, 0.0863276, -0.1598468, 0.1586870
4: -0.1060413, 0.1166707, -0.0957256, 0.1037275, -0.2097688, 0.2123963
5: -0.0819927, 0.1217786, -0.0714798, 0.1101344, -0.1921271, 0.1932585
6: -0.1562666, 0.1251668, -0.1442556, 0.1137682, -0.2700348, 0.2694224
7: 0.7650736, 1.0364275, 0.7815209, 1.0303441, -0.2652705, 0.2549066
8: -0.1377732, 0.1577595, -0.1267065, 0.1440517, -0.2818249, 0.2844660
9: -0.1160903, 0.1875332, -0.1077562, 0.1664350, -0.2825254, 0.2952895

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 75

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0990658, 0.0653810, -0.1443755, 0.0886917, -0.1877575, 0.2097565
1: -0.0941495, 0.0840995, -0.1394131, 0.1208544, -0.2150039, 0.2235126
2: -0.1016481, 0.1941863, -0.1509749, 0.2499562, -0.3516043, 0.3451612
3: -0.0735192, 0.0949680, -0.1163845, 0.1327600, -0.2062792, 0.2113525
4: -0.1060413, 0.1166707, -0.1511612, 0.1732830, -0.2793243, 0.2678319
5: -0.0819927, 0.1217786, -0.1279749, 0.1727093, -0.2547020, 0.2497535
6: -0.1562666, 0.1251668, -0.2088018, 0.1750233, -0.3312899, 0.3339686
7: 0.7650736, 1.0364275, 0.6931350, 1.0630364, -0.2979628, 0.3432925
8: -0.1377732, 0.1577595, -0.1861781, 0.2177165, -0.3554897, 0.3439376
9: -0.1160903, 0.1875332, -0.1525427, 0.2798149, -0.3959052, 0.3400759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=56, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=29, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 75

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.97 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0893545, 0.0603848, -0.0638566, 0.0472669, -0.1366214, 0.1242414
1: -0.0844481, 0.0762218, -0.0589761, 0.0555381, -0.1399862, 0.1351979
2: -0.0910757, 0.1822331, -0.0633173, 0.1508488, -0.2419245, 0.2455504
3: -0.0643318, 0.0868680, -0.0402094, 0.0656007, -0.1299325, 0.1270775
4: -0.0963707, 0.1045369, -0.0709796, 0.0726785, -0.1690492, 0.1755165
5: -0.0721373, 0.1108625, -0.0462610, 0.0822015, -0.1543388, 0.1571235
6: -0.1450067, 0.1144810, -0.1154427, 0.0864244, -0.2314311, 0.2299237
7: 0.7804925, 1.0307244, 0.8209757, 1.0157505, -0.2352580, 0.2097487
8: -0.1273986, 0.1449088, -0.1001589, 0.1111683, -0.2385668, 0.2450677
9: -0.1082775, 0.1677544, -0.0877640, 0.1158232, -0.2241007, 0.2555184

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 68

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.99 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0893545, 0.0603848, -0.1175916, 0.0749120, -0.1642665, 0.1779764
1: -0.0844481, 0.0762218, -0.1126564, 0.0991275, -0.1835756, 0.1888782
2: -0.0910757, 0.1822331, -0.1218163, 0.2169889, -0.3080646, 0.3040494
3: -0.0643318, 0.0868680, -0.0910456, 0.1104201, -0.1747518, 0.1779136
4: -0.0963707, 0.1045369, -0.1244895, 0.1398178, -0.2361885, 0.2290264
5: -0.0721373, 0.1108625, -0.1007935, 0.1426027, -0.2147400, 0.2116560
6: -0.1450067, 0.1144810, -0.1777468, 0.1455517, -0.2905583, 0.2922277
7: 0.7804925, 1.0307244, 0.7356601, 1.0473074, -0.2668148, 0.2950643
8: -0.1273986, 0.1449088, -0.1575646, 0.1822741, -0.3096727, 0.3024734
9: -0.1082775, 0.1677544, -0.1309946, 0.2252645, -0.3335419, 0.2987490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 68

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0990658, 0.0653810, -0.0707013, 0.0507883, -0.1498541, 0.1360823
1: -0.0941495, 0.0840995, -0.0658138, 0.0610905, -0.1552399, 0.1499134
2: -0.1016481, 0.1941863, -0.0707688, 0.1592737, -0.2609218, 0.2649551
3: -0.0735192, 0.0949680, -0.0466849, 0.0713097, -0.1448289, 0.1416529
4: -0.1060413, 0.1166707, -0.0777956, 0.0812306, -0.1872718, 0.1944664
5: -0.0819927, 0.1217786, -0.0532072, 0.0898953, -0.1718880, 0.1749859
6: -0.1562666, 0.1251668, -0.1233789, 0.0939560, -0.2502226, 0.2485457
7: 0.7650736, 1.0364275, 0.8101083, 1.0197703, -0.2546967, 0.2263192
8: -0.1377732, 0.1577595, -0.1074712, 0.1202257, -0.2579989, 0.2652307
9: -0.1160903, 0.1875332, -0.0932707, 0.1297637, -0.2458540, 0.2808039

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 75

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0990658, 0.0653810, -0.1265041, 0.0794974, -0.1785631, 0.1918851
1: -0.0941495, 0.0840995, -0.1215599, 0.1063573, -0.2005067, 0.2056594
2: -0.1016481, 0.1941863, -0.1315190, 0.2279591, -0.3296072, 0.3257053
3: -0.0735192, 0.0949680, -0.0994773, 0.1178539, -0.1913731, 0.1944453
4: -0.1060413, 0.1166707, -0.1333647, 0.1509535, -0.2569948, 0.2500354
5: -0.0819927, 0.1217786, -0.1098382, 0.1526208, -0.2346135, 0.2316169
6: -0.1562666, 0.1251668, -0.1880805, 0.1553586, -0.3116252, 0.3132473
7: 0.7650736, 1.0364275, 0.7215098, 1.0525411, -0.2874675, 0.3149177
8: -0.1377732, 0.1577595, -0.1670860, 0.1940678, -0.3318410, 0.3248455
9: -0.1160903, 0.1875332, -0.1381649, 0.2434165, -0.3595068, 0.3256981

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 75

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.10 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.95 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.00 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529614
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529614
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529043, upper bound: 0.1528499
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529043, upper bound: 0.1528499
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529576, upper bound: 0.1529751
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529576, upper bound: 0.1529751
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0205606, 0.0219397, -0.0447873, 0.0378742, -0.0584347, 0.0667270
1: -0.0219141, 0.0113009, -0.0402738, 0.0398360, -0.0617500, 0.0515747
2: -0.0070595, 0.0872041, -0.0427486, 0.1275824, -0.1346419, 0.1299527
3: -0.0076867, 0.0307276, -0.0235396, 0.0494554, -0.0571422, 0.0542672
4: -0.0266113, 0.0258032, -0.0529485, 0.0494244, -0.0760357, 0.0787517
5: -0.0088441, 0.0346044, -0.0269730, 0.0616072, -0.0704514, 0.0615774
6: -0.0622602, 0.0316897, -0.0938909, 0.0656591, -0.1279193, 0.1255805
7: 0.9073486, 0.9890204, 0.8509499, 1.0047419, -0.0973933, 0.1380705
8: -0.0525140, 0.0529681, -0.0808618, 0.0871900, -0.1397040, 0.1338299
9: -0.0433657, 0.0408607, -0.0723790, 0.0790557, -0.1224215, 0.1132397

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=42, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0251656, 0.0277507, -0.0528957, 0.0416278, -0.0667934, 0.0806465
1: -0.0267177, 0.0181613, -0.0480264, 0.0466468, -0.0733644, 0.0661877
2: -0.0168033, 0.1015942, -0.0513846, 0.1373575, -0.1541608, 0.1529788
3: -0.0106638, 0.0372769, -0.0298399, 0.0564584, -0.0671222, 0.0671168
4: -0.0352928, 0.0302961, -0.0600647, 0.0589834, -0.0942762, 0.0903607
5: -0.0136510, 0.0383018, -0.0351374, 0.0698809, -0.0835319, 0.0734392
6: -0.0717265, 0.0418902, -0.1027338, 0.0743637, -0.1460902, 0.1446240
7: 0.8884658, 0.9936767, 0.8383783, 1.0093135, -0.1208476, 0.1552984
8: -0.0617222, 0.0649984, -0.0884493, 0.0966641, -0.1583863, 0.1534477
9: -0.0534725, 0.0497473, -0.0789458, 0.0934993, -0.1469718, 0.1286931

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0205606, 0.0219397, -0.0696754, 0.0502605, -0.0708210, 0.0916151
1: -0.0219141, 0.0113009, -0.0647890, 0.0602583, -0.0821723, 0.0760899
2: -0.0070595, 0.0872041, -0.0696519, 0.1580109, -0.1650704, 0.1568560
3: -0.0076867, 0.0307276, -0.0457143, 0.0704540, -0.0781407, 0.0764420
4: -0.0266113, 0.0258032, -0.0767740, 0.0799488, -0.1065601, 0.1025772
5: -0.0088441, 0.0346044, -0.0521661, 0.0887421, -0.0975862, 0.0867705
6: -0.0622602, 0.0316897, -0.1221894, 0.0928271, -0.1550874, 0.1538790
7: 0.9073486, 0.9890204, 0.8117372, 1.0191678, -0.1118192, 0.1772832
8: -0.0525140, 0.0529681, -0.1063752, 0.1188681, -0.1713821, 0.1593433
9: -0.0433657, 0.0408607, -0.0924453, 0.1276742, -0.1710399, 0.1333060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=42, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0251656, 0.0277507, -0.0777685, 0.0544242, -0.0795897, 0.1055192
1: -0.0267177, 0.0181613, -0.0728739, 0.0668233, -0.0935410, 0.0910352
2: -0.0168033, 0.1015942, -0.0784626, 0.1679723, -0.1847756, 0.1800568
3: -0.0106638, 0.0372769, -0.0533709, 0.0772043, -0.0878682, 0.0906478
4: -0.0352928, 0.0302961, -0.0848332, 0.0900607, -0.1253535, 0.1151293
5: -0.0136510, 0.0383018, -0.0603793, 0.0978393, -0.1114903, 0.0986812
6: -0.0717265, 0.0418902, -0.1315731, 0.1017324, -0.1734589, 0.1734633
7: 0.8884658, 0.9936767, 0.7988876, 1.0239204, -0.1354546, 0.1947891
8: -0.0617222, 0.0649984, -0.1150212, 0.1295775, -0.1912996, 0.1800196
9: -0.0534725, 0.0497473, -0.0989564, 0.1441574, -0.1976299, 0.1487036

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.28 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0224381, 0.0243322, -0.0518784, 0.0411044, -0.0635425, 0.0762106
1: -0.0238918, 0.0140830, -0.0470101, 0.0458215, -0.0697133, 0.0610931
2: -0.0110649, 0.0931288, -0.0502771, 0.1361054, -0.1471703, 0.1434059
3: -0.0088835, 0.0334241, -0.0288775, 0.0556099, -0.0644933, 0.0623016
4: -0.0301727, 0.0276530, -0.0590516, 0.0577124, -0.0878851, 0.0867046
5: -0.0108232, 0.0360929, -0.0341050, 0.0687373, -0.0795606, 0.0701979
6: -0.0661577, 0.0358588, -0.1015543, 0.0732443, -0.1394020, 0.1374131
7: 0.8995742, 0.9909337, 0.8399936, 1.0087162, -0.1091421, 0.1509401
8: -0.0562904, 0.0579212, -0.0873625, 0.0953179, -0.1516083, 0.1452837
9: -0.0475207, 0.0445194, -0.0781274, 0.0914274, -0.1389481, 0.1226468

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 33

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0303041, 0.0306308, -0.0602383, 0.0454053, -0.0757095, 0.0908691
1: -0.0301341, 0.0241580, -0.0553615, 0.0526030, -0.0827371, 0.0795195
2: -0.0239544, 0.1087262, -0.0593781, 0.1463951, -0.1703495, 0.1681044
3: -0.0140798, 0.0405229, -0.0367864, 0.0625827, -0.0766625, 0.0773093
4: -0.0402958, 0.0348302, -0.0673765, 0.0681576, -0.1084534, 0.1022067
5: -0.0165176, 0.0448231, -0.0425889, 0.0781344, -0.0946519, 0.0874120
6: -0.0776593, 0.0486018, -0.1112473, 0.0824431, -0.1601024, 0.1598491
7: 0.8778584, 0.9965653, 0.8267205, 1.0136255, -0.1357671, 0.1698448
8: -0.0671542, 0.0710047, -0.0962934, 0.1063803, -0.1735345, 0.1672981
9: -0.0588151, 0.0571716, -0.0848530, 0.1084538, -0.1672689, 0.1420246

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0224381, 0.0243322, -0.0769145, 0.0539848, -0.0764229, 0.1012468
1: -0.0238918, 0.0140830, -0.0720208, 0.0661306, -0.0900224, 0.0861037
2: -0.0110649, 0.0931288, -0.0775329, 0.1669213, -0.1779862, 0.1706617
3: -0.0088835, 0.0334241, -0.0525630, 0.0764921, -0.0853755, 0.0859871
4: -0.0301727, 0.0276530, -0.0839829, 0.0889938, -0.1191665, 0.1116359
5: -0.0108232, 0.0360929, -0.0595127, 0.0968794, -0.1077026, 0.0956056
6: -0.0661577, 0.0358588, -0.1305829, 0.1007928, -0.1669504, 0.1664417
7: 0.8995742, 0.9909337, 0.8002434, 1.0234189, -0.1238447, 0.1906903
8: -0.0562904, 0.0579212, -0.1141089, 0.1284474, -0.1847378, 0.1720301
9: -0.0475207, 0.0445194, -0.0982693, 0.1424181, -0.1899388, 0.1427887

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 33

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.08 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0303041, 0.0306308, -0.0849803, 0.0581344, -0.0884385, 0.1156111
1: -0.0301341, 0.0241580, -0.0800783, 0.0726735, -0.1028076, 0.1042363
2: -0.0239544, 0.1087262, -0.0863138, 0.1768490, -0.2008034, 0.1950400
3: -0.0140798, 0.0405229, -0.0601936, 0.0832195, -0.0972993, 0.1007164
4: -0.0402958, 0.0348302, -0.0920148, 0.0990715, -0.1393673, 0.1268450
5: -0.0165176, 0.0448231, -0.0676981, 0.1059457, -0.1224633, 0.1125212
6: -0.0776593, 0.0486018, -0.1399349, 0.1096679, -0.1873272, 0.1885366
7: 0.8778584, 0.9965653, 0.7874374, 1.0281556, -0.1502972, 0.2091280
8: -0.0671542, 0.0710047, -0.1227255, 0.1391206, -0.2062748, 0.1937302
9: -0.0588151, 0.0571716, -0.1047583, 0.1588454, -0.2176606, 0.1619299

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0335817, 0.0323988, -0.0290876, 0.0299746, -0.0635563, 0.0614864
1: -0.0322652, 0.0279848, -0.0293431, 0.0227378, -0.0550029, 0.0573279
2: -0.0285390, 0.1131044, -0.0222527, 0.1071013, -0.1356403, 0.1353571
3: -0.0162378, 0.0425155, -0.0132789, 0.0397833, -0.0560211, 0.0557944
4: -0.0433670, 0.0378377, -0.0391559, 0.0337140, -0.0770810, 0.0769936
5: -0.0185123, 0.0489199, -0.0157773, 0.0433026, -0.0618149, 0.0646971
6: -0.0813843, 0.0527219, -0.0762768, 0.0470726, -0.1284570, 0.1289987
7: 0.8713059, 0.9985181, 0.8802904, 0.9958407, -0.1245347, 0.1182277
8: -0.0704910, 0.0747131, -0.0659158, 0.0696283, -0.1401193, 0.1406289
9: -0.0620949, 0.0620371, -0.0575979, 0.0553657, -0.1174605, 0.1196350

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 68

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.08 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0449534, 0.0379406, -0.0352532, 0.0333004, -0.0782538, 0.0731937
1: -0.0404242, 0.0399797, -0.0333520, 0.0299363, -0.0703605, 0.0733317
2: -0.0429220, 0.1277761, -0.0308770, 0.1153371, -0.1582591, 0.1586531
3: -0.0236403, 0.0496033, -0.0173383, 0.0435316, -0.0671720, 0.0669416
4: -0.0530705, 0.0496097, -0.0449332, 0.0393714, -0.0924419, 0.0945429
5: -0.0271373, 0.0617612, -0.0195295, 0.0510091, -0.0781464, 0.0812907
6: -0.0940617, 0.0658308, -0.0832839, 0.0548229, -0.1488846, 0.1491147
7: 0.8507017, 1.0048318, 0.8679644, 0.9995139, -0.1488122, 0.1368674
8: -0.0809907, 0.0873609, -0.0721926, 0.0766042, -0.1575949, 0.1595535
9: -0.0725143, 0.0793006, -0.0637674, 0.0645184, -0.1370327, 0.1430680

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 68

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0383967, 0.0349961, -0.0343823, 0.0328307, -0.0712273, 0.0693784
1: -0.0353959, 0.0336065, -0.0327857, 0.0289195, -0.0643154, 0.0663923
2: -0.0352742, 0.1195362, -0.0296589, 0.1141738, -0.1494480, 0.1491951
3: -0.0194080, 0.0454427, -0.0167649, 0.0430022, -0.0624102, 0.0622076
4: -0.0478788, 0.0422559, -0.0441172, 0.0385723, -0.0864511, 0.0863731
5: -0.0214426, 0.0549382, -0.0189995, 0.0499206, -0.0713631, 0.0739377
6: -0.0868565, 0.0587744, -0.0822942, 0.0537282, -0.1405847, 0.1410686
7: 0.8616802, 1.0013866, 0.8697054, 0.9989949, -0.1373147, 0.1316813
8: -0.0753929, 0.0801609, -0.0713060, 0.0756189, -0.1510118, 0.1514670
9: -0.0669129, 0.0691850, -0.0628960, 0.0632256, -0.1301386, 0.1320810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 247

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0520881, 0.0412123, -0.0405691, 0.0360950, -0.0881831, 0.0817813
1: -0.0472196, 0.0459916, -0.0368700, 0.0359850, -0.0832046, 0.0828617
2: -0.0505054, 0.1363634, -0.0381238, 0.1223945, -0.1728999, 0.1744872
3: -0.0290758, 0.0557848, -0.0208415, 0.0466813, -0.0757571, 0.0766263
4: -0.0592604, 0.0579743, -0.0497877, 0.0444641, -0.1037245, 0.1077620
5: -0.0343178, 0.0689730, -0.0229832, 0.0574846, -0.0918024, 0.0919562
6: -0.1017974, 0.0734750, -0.0893165, 0.0613352, -0.1631326, 0.1627915
7: 0.8396606, 1.0088392, 0.8575980, 1.0026002, -0.1629395, 0.1512412
8: -0.0875865, 0.0955954, -0.0774668, 0.0826139, -0.1702004, 0.1730622
9: -0.0782960, 0.0918545, -0.0689515, 0.0725000, -0.1507961, 0.1608059

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 247

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0228216, 0.0248209, -0.0290876, 0.0299746, -0.0527962, 0.0539085
1: -0.0242957, 0.0146512, -0.0293431, 0.0227378, -0.0470335, 0.0439943
2: -0.0118829, 0.0943388, -0.0222527, 0.1071013, -0.1189842, 0.1165916
3: -0.0091279, 0.0339748, -0.0132789, 0.0397833, -0.0489112, 0.0472537
4: -0.0309001, 0.0280308, -0.0391559, 0.0337140, -0.0646141, 0.0671867
5: -0.0112275, 0.0363968, -0.0157773, 0.0433026, -0.0545300, 0.0521741
6: -0.0669537, 0.0367103, -0.0762768, 0.0470726, -0.1140263, 0.1129871
7: 0.8979864, 0.9913244, 0.8802904, 0.9958407, -0.0978543, 0.1110340
8: -0.0570617, 0.0589328, -0.0659158, 0.0696283, -0.1266900, 0.1248486
9: -0.0483693, 0.0452667, -0.0575979, 0.0553657, -0.1037350, 0.1028646

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 68

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0315121, 0.0312824, -0.0352532, 0.0333004, -0.0648125, 0.0665356
1: -0.0309195, 0.0255684, -0.0333520, 0.0299363, -0.0608558, 0.0589204
2: -0.0256440, 0.1103398, -0.0308770, 0.1153371, -0.1409811, 0.1412169
3: -0.0148752, 0.0412572, -0.0173383, 0.0435316, -0.0584068, 0.0585955
4: -0.0414277, 0.0359386, -0.0449332, 0.0393714, -0.0807991, 0.0808719
5: -0.0172528, 0.0463330, -0.0195295, 0.0510091, -0.0682618, 0.0658625
6: -0.0790322, 0.0501203, -0.0832839, 0.0548229, -0.1338551, 0.1334042
7: 0.8754434, 0.9972848, 0.8679644, 0.9995139, -0.1240705, 0.1293204
8: -0.0683840, 0.0723715, -0.0721926, 0.0766042, -0.1449882, 0.1445641
9: -0.0600239, 0.0589648, -0.0637674, 0.0645184, -0.1245424, 0.1227322

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 68

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0246613, 0.0271652, -0.0343823, 0.0328307, -0.0574920, 0.0615475
1: -0.0262336, 0.0173771, -0.0327857, 0.0289195, -0.0551531, 0.0501629
2: -0.0158076, 0.1001441, -0.0296589, 0.1141738, -0.1299814, 0.1298030
3: -0.0103005, 0.0366169, -0.0167649, 0.0430022, -0.0533027, 0.0533818
4: -0.0343896, 0.0298433, -0.0441172, 0.0385723, -0.0729620, 0.0739605
5: -0.0131667, 0.0378553, -0.0189995, 0.0499206, -0.0630872, 0.0568548
6: -0.0707725, 0.0407954, -0.0822942, 0.0537282, -0.1245007, 0.1230896
7: 0.8903686, 0.9931989, 0.8697054, 0.9989949, -0.1086262, 0.1234935
8: -0.0607619, 0.0637861, -0.0713060, 0.0756189, -0.1363808, 0.1350921
9: -0.0524405, 0.0488518, -0.0628960, 0.0632256, -0.1156661, 0.1117478

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 247

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0361957, 0.0338088, -0.0405691, 0.0360950, -0.0722906, 0.0743779
1: -0.0339648, 0.0310367, -0.0368700, 0.0359850, -0.0699498, 0.0679068
2: -0.0321954, 0.1165961, -0.0381238, 0.1223945, -0.1545898, 0.1547198
3: -0.0179588, 0.0441046, -0.0208415, 0.0466813, -0.0646401, 0.0649461
4: -0.0458164, 0.0402362, -0.0497877, 0.0444641, -0.0902805, 0.0900239
5: -0.0201031, 0.0521871, -0.0229832, 0.0574846, -0.0775877, 0.0751703
6: -0.0843550, 0.0560077, -0.0893165, 0.0613352, -0.1456902, 0.1453241
7: 0.8660803, 1.0000753, 0.8575980, 1.0026002, -0.1365198, 0.1424773
8: -0.0731521, 0.0776706, -0.0774668, 0.0826139, -0.1557660, 0.1551374
9: -0.0647105, 0.0659175, -0.0689515, 0.0725000, -0.1372105, 0.1348690

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 247

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0367655, 0.0341162, -0.0329199, 0.0320418, -0.0688073, 0.0670361
1: -0.0343352, 0.0317020, -0.0318349, 0.0272121, -0.0615474, 0.0635368
2: -0.0329924, 0.1173571, -0.0276133, 0.1122203, -0.1452127, 0.1449704
3: -0.0183339, 0.0444510, -0.0158021, 0.0421131, -0.0604471, 0.0602531
4: -0.0463503, 0.0407591, -0.0427469, 0.0372304, -0.0835807, 0.0835059
5: -0.0204498, 0.0528993, -0.0181095, 0.0480926, -0.0685425, 0.0710089
6: -0.0850026, 0.0567239, -0.0806322, 0.0518899, -0.1368925, 0.1373560
7: 0.8649412, 1.0004148, 0.8726290, 0.9981237, -0.1331825, 0.1277858
8: -0.0737322, 0.0783153, -0.0698172, 0.0739643, -0.1476965, 0.1481325
9: -0.0652807, 0.0667634, -0.0614326, 0.0610547, -0.1263354, 0.1281960

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 247

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.17 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0453684, 0.0381932, -0.0318147, 0.0314457, -0.0768140, 0.0700079
1: -0.0404052, 0.0405264, -0.0311163, 0.0259218, -0.0663270, 0.0716427
2: -0.0435647, 0.1285125, -0.0260674, 0.1107441, -0.1543088, 0.1545799
3: -0.0240233, 0.0490460, -0.0150744, 0.0414412, -0.0654646, 0.0641204
4: -0.0534325, 0.0503138, -0.0417113, 0.0362163, -0.0896488, 0.0920251
5: -0.0273745, 0.0623464, -0.0174369, 0.0467112, -0.0740857, 0.0797834
6: -0.0947110, 0.0662247, -0.0793761, 0.0505007, -0.1452117, 0.1456008
7: 0.8497580, 1.0049175, 0.8748384, 0.9974653, -0.1477073, 0.1300791
8: -0.0814267, 0.0880105, -0.0686921, 0.0727139, -0.1541406, 0.1567026
9: -0.0728437, 0.0802311, -0.0603267, 0.0594141, -0.1322578, 0.1405579

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 247

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.49 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0433057, 0.0372815, -0.0443909, 0.0377156, -0.0810213, 0.0816723
1: -0.0389313, 0.0385530, -0.0399146, 0.0394926, -0.0784239, 0.0784676
2: -0.0412022, 0.1258541, -0.0423348, 0.1271199, -0.1683221, 0.1681889
3: -0.0226407, 0.0481363, -0.0232991, 0.0491025, -0.0717432, 0.0714354
4: -0.0518595, 0.0477720, -0.0526571, 0.0489823, -0.1008418, 0.1004290
5: -0.0255071, 0.0602338, -0.0265808, 0.0612397, -0.0867469, 0.0868146
6: -0.0923669, 0.0641273, -0.0934831, 0.0652492, -0.1576161, 0.1576104
7: 0.8531647, 1.0039377, 0.8515426, 1.0045265, -0.1513618, 0.1523951
8: -0.0797118, 0.0856656, -0.0805540, 0.0867821, -0.1664938, 0.1662196
9: -0.0711719, 0.0768718, -0.0720560, 0.0784714, -0.1496433, 0.1489278

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 247

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0540474, 0.0422203, -0.0428752, 0.0371032, -0.0911506, 0.0850954
1: -0.0491769, 0.0475810, -0.0385687, 0.0381672, -0.0873441, 0.0861497
2: -0.0526383, 0.1387751, -0.0407382, 0.1253342, -0.1779726, 0.1795133
3: -0.0309295, 0.0574190, -0.0223704, 0.0478175, -0.0787470, 0.0797894
4: -0.0612115, 0.0604223, -0.0515390, 0.0472749, -0.1084865, 0.1119613
5: -0.0363061, 0.0711754, -0.0250932, 0.0598207, -0.0961269, 0.0962686
6: -0.1040691, 0.0756309, -0.0919086, 0.0636846, -0.1677537, 0.1675395
7: 0.8365498, 1.0099900, 0.8538307, 1.0037137, -0.1671640, 0.1561593
8: -0.0896797, 0.0981880, -0.0793696, 0.0852070, -0.1748867, 0.1775576
9: -0.0798723, 0.0958449, -0.0708217, 0.0762149, -0.1560873, 0.1666666

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 247

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0429082, 0.0371177, -0.0377385, 0.0346411, -0.0775493, 0.0748562
1: -0.0385931, 0.0381985, -0.0349680, 0.0328381, -0.0714312, 0.0731665
2: -0.0407757, 0.1253764, -0.0343535, 0.1186570, -0.1594327, 0.1597300
3: -0.0223923, 0.0478338, -0.0189746, 0.0450426, -0.0674349, 0.0668084
4: -0.0515642, 0.0473153, -0.0472620, 0.0416519, -0.0932161, 0.0945773
5: -0.0251235, 0.0598542, -0.0210420, 0.0541156, -0.0792391, 0.0808963
6: -0.0919459, 0.0637183, -0.0861085, 0.0579471, -0.1498929, 0.1498268
7: 0.8537768, 1.0037297, 0.8629959, 1.0009944, -0.1472177, 0.1407338
8: -0.0793969, 0.0852442, -0.0747228, 0.0794162, -0.1588131, 0.1599670
9: -0.0708485, 0.0762682, -0.0662544, 0.0682079, -0.1390565, 0.1425226

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 33

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0515278, 0.0409240, -0.0364057, 0.0339221, -0.0854499, 0.0773297
1: -0.0466598, 0.0455371, -0.0341014, 0.0312819, -0.0779418, 0.0796385
2: -0.0498953, 0.1356738, -0.0324892, 0.1168766, -0.1667719, 0.1681630
3: -0.0285458, 0.0553174, -0.0180971, 0.0442323, -0.0727781, 0.0734145
4: -0.0587024, 0.0572742, -0.0460132, 0.0404290, -0.0991314, 0.1032874
5: -0.0337492, 0.0683432, -0.0202309, 0.0524497, -0.0861989, 0.0885740
6: -0.1011478, 0.0728585, -0.0845937, 0.0562717, -0.1574195, 0.1574522
7: 0.8405502, 1.0085102, 0.8656604, 1.0002003, -0.1596500, 0.1428498
8: -0.0869879, 0.0948539, -0.0733659, 0.0779083, -0.1648962, 0.1682199
9: -0.0778453, 0.0907132, -0.0649207, 0.0662294, -0.1440746, 0.1556339

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 33

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0508671, 0.0405841, -0.0513611, 0.0408382, -0.0917053, 0.0919452
1: -0.0459998, 0.0450011, -0.0464933, 0.0454018, -0.0914016, 0.0914944
2: -0.0491761, 0.1348605, -0.0497139, 0.1354685, -0.1846446, 0.1845744
3: -0.0279207, 0.0547663, -0.0283880, 0.0551783, -0.0830991, 0.0831544
4: -0.0580445, 0.0564487, -0.0585364, 0.0570659, -0.1151104, 0.1149851
5: -0.0330786, 0.0676005, -0.0335799, 0.0681558, -0.1012344, 0.1011805
6: -0.1003817, 0.0721314, -0.1009544, 0.0726750, -0.1730567, 0.1730859
7: 0.8415993, 1.0081223, 0.8408149, 1.0084124, -0.1668131, 0.1673074
8: -0.0862821, 0.0939796, -0.0868098, 0.0946333, -0.1809154, 0.1807894
9: -0.0773137, 0.0893676, -0.0777111, 0.0903737, -0.1676874, 0.1670787

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 247

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0602622, 0.0454177, -0.0493916, 0.0398250, -0.1000873, 0.0948093
1: -0.0553854, 0.0526224, -0.0445259, 0.0438043, -0.0991897, 0.0971483
2: -0.0594042, 0.1464247, -0.0475698, 0.1330445, -0.1924487, 0.1939945
3: -0.0368091, 0.0626027, -0.0265249, 0.0535357, -0.0903448, 0.0891276
4: -0.0674004, 0.0681875, -0.0565753, 0.0546052, -0.1220056, 0.1247628
5: -0.0426133, 0.0781612, -0.0315813, 0.0659421, -0.1085553, 0.1097426
6: -0.1112751, 0.0824695, -0.0986710, 0.0705080, -0.1817831, 0.1811405
7: 0.8266824, 1.0136396, 0.8439417, 1.0072558, -0.1805733, 0.1696978
8: -0.0963191, 0.1064120, -0.0847059, 0.0920273, -0.1883464, 0.1911179
9: -0.0848723, 0.1085027, -0.0761267, 0.0863626, -0.1712349, 0.1846294

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 247

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0744607, 0.0527224, -0.0354930, 0.0334298, -0.1078905, 0.0882154
1: -0.0695694, 0.0641400, -0.0335079, 0.0302163, -0.0997857, 0.0976480
2: -0.0748615, 0.1639009, -0.0312125, 0.1156574, -0.1905189, 0.1951134
3: -0.0502415, 0.0744454, -0.0174962, 0.0436774, -0.0939189, 0.0919416
4: -0.0815393, 0.0859278, -0.0451579, 0.0395915, -0.1211308, 0.1310857
5: -0.0570224, 0.0941211, -0.0196754, 0.0513089, -0.1083313, 0.1137965
6: -0.1277378, 0.0980926, -0.0835565, 0.0551243, -0.1828621, 0.1816491
7: 0.8041396, 1.0219779, 0.8674850, 0.9996567, -0.1955171, 0.1544929
8: -0.1114874, 0.1252004, -0.0724368, 0.0768756, -0.1883630, 0.1976371
9: -0.0962951, 0.1374203, -0.0640074, 0.0648745, -0.1611696, 0.2014277

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 68

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0845291, 0.0579023, -0.0342220, 0.0327442, -0.1172733, 0.0921243
1: -0.0796276, 0.0723075, -0.0326815, 0.0287324, -0.1083600, 0.1049890
2: -0.0858226, 0.1762938, -0.0294346, 0.1139597, -0.1997823, 0.2057284
3: -0.0597668, 0.0828432, -0.0166594, 0.0429047, -0.1026715, 0.0995026
4: -0.0915655, 0.0985078, -0.0439670, 0.0384252, -0.1299907, 0.1424748
5: -0.0672402, 0.1054386, -0.0189020, 0.0497202, -0.1169604, 0.1243406
6: -0.1394118, 0.1091715, -0.0821120, 0.0535267, -0.1929385, 0.1912835
7: 0.7881536, 1.0278908, 0.8700260, 0.9988997, -0.2107460, 0.1578648
8: -0.1222436, 0.1385236, -0.0711429, 0.0754376, -0.1976812, 0.2096664
9: -0.1043954, 0.1579265, -0.0627356, 0.0629877, -0.1673830, 0.2206621

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 68

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0813255, 0.0562542, -0.0408402, 0.0362135, -0.1175391, 0.0970943
1: -0.0764273, 0.0697088, -0.0370697, 0.0362416, -0.1126689, 0.1067785
2: -0.0823350, 0.1723506, -0.0384312, 0.1227401, -0.2050751, 0.2107818
3: -0.0567360, 0.0801712, -0.0210213, 0.0468148, -0.1035508, 0.1011925
4: -0.0883754, 0.0945050, -0.0499936, 0.0447946, -0.1331699, 0.1444986
5: -0.0639891, 0.1018376, -0.0232312, 0.0577593, -0.1217484, 0.1250688
6: -0.1356973, 0.1056464, -0.0896212, 0.0616115, -0.1973088, 0.1952676
7: 0.7932401, 1.0260094, 0.8571551, 1.0027312, -0.2094911, 0.1688544
8: -0.1188212, 0.1342843, -0.0776906, 0.0829187, -0.2017399, 0.2119749
9: -0.1018180, 0.1514019, -0.0691714, 0.0729367, -0.1747547, 0.2205733

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0907857, 0.0611212, -0.0393919, 0.0355329, -0.1263186, 0.1005130
1: -0.0858778, 0.0773828, -0.0360430, 0.0347684, -0.1206462, 0.1134258
2: -0.0926339, 0.1839947, -0.0366662, 0.1208655, -0.2134994, 0.2206609
3: -0.0656858, 0.0880618, -0.0200632, 0.0460477, -0.1117335, 0.1081249
4: -0.0977959, 0.1063251, -0.0488113, 0.0431690, -0.1409649, 0.1551364
5: -0.0735897, 0.1124713, -0.0220482, 0.0561822, -0.1297719, 0.1345195
6: -0.1466661, 0.1160559, -0.0879875, 0.0600254, -0.2066915, 0.2040434
7: 0.7782202, 1.0315650, 0.8596905, 1.0019794, -0.2237592, 0.1718744
8: -0.1289276, 0.1468028, -0.0764060, 0.0812869, -0.2102145, 0.2232088
9: -0.1094289, 0.1706693, -0.0679088, 0.0706623, -0.1800912, 0.2385780

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 33

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0359440, 0.0336731, -0.0279410, 0.0293561, -0.0653001, 0.0616141
1: -0.0338012, 0.0307428, -0.0285976, 0.0213991, -0.0552002, 0.0593404
2: -0.0318434, 0.1162598, -0.0206488, 0.1055697, -0.1374131, 0.1369086
3: -0.0177931, 0.0439516, -0.0125240, 0.0390862, -0.0568793, 0.0564757
4: -0.0455805, 0.0400053, -0.0380815, 0.0326619, -0.0782424, 0.0780868
5: -0.0199499, 0.0518725, -0.0150795, 0.0418695, -0.0618194, 0.0669520
6: -0.0840689, 0.0556913, -0.0749737, 0.0456314, -0.1297003, 0.1306650
7: 0.8665834, 0.9999254, 0.8825825, 0.9951578, -0.1285744, 0.1173428
8: -0.0728959, 0.0773859, -0.0647485, 0.0683310, -0.1412269, 0.1421344
9: -0.0644587, 0.0655439, -0.0564505, 0.0536636, -0.1181222, 0.1219945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 68

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
time: 0.99 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.94 seconds
IS_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529614
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529614
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529043, upper bound: 0.1528499
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529043, upper bound: 0.1528499
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529576, upper bound: 0.1529751
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529576, upper bound: 0.1529751
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 7, lower bound: -0.1529751, upper bound: 0.1529751

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.78 + 597.79 = 601.57 seconds
