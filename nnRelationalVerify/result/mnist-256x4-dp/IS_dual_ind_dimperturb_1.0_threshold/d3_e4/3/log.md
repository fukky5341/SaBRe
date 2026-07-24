## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 106.6602947382


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797)
1: (-48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317)
2: (-65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747)
3: (-68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471)
4: (-64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390)
5: (-57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115)
6: (-54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235)
7: (-58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829)
8: (-71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617)
9: (-54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.53 + 11.35 = 12.88 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -106.7670618, upper bound: 106.7670618

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7509480, upper bound: 106.7484515
time: 8.19 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7477160, upper bound: 106.7477160
time: 8.16 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 16.51 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 16.51
Output dim: 0, lower bound: -106.7509480, upper bound: 106.7484515
IS_A2, status: Status.UNKNOWN, split count: 1, time: 16.51
Output dim: 0, lower bound: -106.7477160, upper bound: 106.7477160

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -43.2367897, 34.4949570, -57.1260300, 45.5063477, -88.7431259, 91.6209717
1: -34.9878044, 29.8007889, -46.5087395, 39.4619522, -74.4497528, 76.3095169
2: -46.5671349, 30.1880875, -61.8263855, 39.9733696, -86.5405045, 92.0144577
3: -49.2782364, 25.9414082, -65.3586578, 34.4015312, -83.6797638, 91.3000641
4: -46.0794792, 34.7557182, -60.8894615, 46.1131668, -92.1926422, 95.6451797
5: -41.5383759, 32.7077026, -55.0305443, 43.1037903, -84.6421432, 87.7382431
6: -39.4550514, 37.8932304, -52.1641197, 49.9888725, -89.4439163, 90.0573502
7: -42.1981354, 36.3016472, -55.9273071, 47.9200172, -90.1181335, 92.2289581
8: -51.0103607, 33.9446335, -67.8608627, 45.2444916, -96.2548523, 101.8054886
9: -38.9905930, 38.0316658, -51.5498238, 50.2817078, -89.2722931, 89.5814896

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=63, inp2_unstable=64, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=149, inp2_unstable=150, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=197, inp2_unstable=221, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7404419, upper bound: 106.7383882
time: 8.89 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7403021, upper bound: 106.7374048
time: 9.77 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -35.7654915, 28.4981251, -46.8274879, 37.3221436, -73.0876312, 75.3256149
1: -28.6854954, 24.5347595, -37.9242325, 32.2664948, -60.9519882, 62.4589844
2: -38.3050308, 24.8694439, -50.4714355, 32.6462212, -70.9512482, 75.3408661
3: -40.5462570, 21.3767605, -53.3877144, 28.1204891, -68.6667480, 74.7644730
4: -37.9453735, 28.6490612, -49.8108444, 37.6521950, -75.5975647, 78.4599075
5: -34.1815300, 27.0604343, -45.0441284, 35.3928070, -69.5743256, 72.1045532
6: -32.5468407, 31.3406620, -42.6719742, 40.9815521, -73.5283813, 74.0126343
7: -34.7955933, 30.0198460, -45.6488113, 39.2752914, -74.0708771, 75.6686554
8: -41.8391533, 27.7593937, -55.2702980, 36.7987862, -78.6379395, 83.0296936
9: -32.1872482, 31.2693481, -42.1782913, 41.1054382, -73.2926865, 73.4476242

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=63, inp2_unstable=64, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=157, inp2_unstable=149, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=175, inp2_unstable=204, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7329482, upper bound: 106.7281844
time: 9.56 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7435498, upper bound: 106.7435498
time: 7.94 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 19.01 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 19.01
Output dim: 0, lower bound: -106.7404419, upper bound: 106.7383882
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 19.01
Output dim: 0, lower bound: -106.7403021, upper bound: 106.7374048
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 19.01
Output dim: 0, lower bound: -106.7329482, upper bound: 106.7281844
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 19.01
Output dim: 0, lower bound: -106.7435498, upper bound: 106.7435498

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -39.4144402, 31.4517403, -41.7696991, 33.2865639, -72.7010040, 73.2214355
1: -31.8218117, 27.1506996, -33.7678795, 28.7669888, -60.5887947, 60.9185791
2: -42.3857384, 27.4988995, -44.9783859, 29.0818100, -71.4675369, 72.4772873
3: -44.8457909, 23.6152859, -47.5543365, 25.0169697, -69.8627548, 71.1696243
4: -41.9799309, 31.6541100, -44.4619865, 33.5388527, -75.5187683, 76.1160965
5: -37.8115425, 29.8467808, -40.1380577, 31.6360874, -69.4476166, 69.9848404
6: -35.9555817, 34.5581627, -38.0400887, 36.5754623, -72.5310364, 72.5982513
7: -38.4419250, 33.1212234, -40.7144432, 35.0862427, -73.5281677, 73.8356628
8: -46.3705750, 30.8078651, -49.1352615, 32.6204224, -78.9909973, 79.9431305
9: -35.5475807, 34.6308556, -37.6220818, 36.6176987, -72.1652679, 72.2529373

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=63, inp2_unstable=63, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=149, inp2_unstable=150, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=189, inp2_unstable=193, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7205639, upper bound: 106.7228340
time: 8.45 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7361426, upper bound: 106.7339479
time: 8.52 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -35.5550346, 28.3686314, -41.0954018, 32.7231026, -68.2781372, 69.4640350
1: -28.6053123, 24.4596920, -33.1540871, 28.2778568, -56.8831711, 57.6137733
2: -38.1526184, 24.7707939, -44.2174034, 28.5451202, -66.6977386, 68.9881973
3: -40.3334618, 21.2773552, -46.7726402, 24.5852566, -64.9187164, 68.0499878
4: -37.8037033, 28.5302238, -43.6781502, 32.9869766, -70.7906723, 72.2083664
5: -34.0482864, 26.9581890, -39.4650764, 31.0813084, -65.1295853, 66.4232483
6: -32.4063263, 31.1801186, -37.3933792, 35.9815826, -68.3879013, 68.5734863
7: -34.6294899, 29.8969402, -40.0162201, 34.5093307, -69.1388092, 69.9131393
8: -41.6798592, 27.6523018, -48.2439919, 31.9987011, -73.6785583, 75.8962936
9: -32.0586395, 31.1621895, -36.9952660, 35.9585495, -68.0171890, 68.1574554

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=63, inp2_unstable=63, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=149, inp2_unstable=158, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=180, inp2_unstable=191, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7205029, upper bound: 106.7225731
time: 9.42 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7358983, upper bound: 106.7327698
time: 8.09 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -30.1891232, 24.0460186, -32.8912849, 26.2247200, -56.4138374, 56.9373016
1: -24.0445480, 20.6514492, -26.3511429, 22.5753899, -46.6199379, 47.0025902
2: -32.2060280, 20.9289417, -35.1938286, 22.7152443, -54.9212723, 56.1227646
3: -34.0082436, 17.9537144, -37.1199646, 19.6106071, -53.6188507, 55.0736771
4: -31.9257584, 24.1541386, -34.7933502, 26.3842850, -58.3100357, 58.9474869
5: -28.7468357, 22.8470268, -31.4890842, 24.9431286, -53.6899643, 54.3361130
6: -27.3880329, 26.4726028, -29.8149204, 28.8785706, -56.2666016, 56.2875099
7: -29.3281002, 25.3119297, -31.9067802, 27.5848331, -56.9129333, 57.2187119
8: -35.1237335, 23.3037052, -38.2919464, 25.4591770, -60.5829048, 61.5956497
9: -27.1509094, 26.3086071, -29.6109352, 28.6589451, -55.8098526, 55.9195404

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=63, inp2_unstable=63, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=156, inp2_unstable=148, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=161, inp2_unstable=172, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7214853, upper bound: 106.7159808
time: 10.88 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7209231, upper bound: 106.7158192
time: 10.41 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -33.9287224, 27.0298824, -40.2161140, 32.0669556, -65.9956818, 67.2459869
1: -27.1591053, 23.2587318, -32.4377708, 27.6869431, -54.8460426, 55.6965027
2: -36.2973976, 23.5752411, -43.2416191, 27.9450035, -64.2424011, 66.8168640
3: -38.4010010, 20.2501183, -45.7074623, 24.0982819, -62.4992790, 65.9575806
4: -35.9663963, 27.1698589, -42.7176552, 32.2895584, -68.2559509, 69.8875122
5: -32.3917694, 25.6757030, -38.6177139, 30.4522324, -62.8439980, 64.2934189
6: -30.8582458, 29.7365932, -36.5928535, 35.2487755, -66.1070251, 66.3294449
7: -32.9985924, 28.4730682, -39.1207161, 33.7391357, -66.7377319, 67.5937805
8: -39.6190834, 26.2788525, -47.2089272, 31.3984394, -71.0175247, 73.4877777
9: -30.5329399, 29.6359177, -36.2138100, 35.2288857, -65.7618179, 65.8497314

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=63, inp2_unstable=63, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=157, inp2_unstable=147, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=169, inp2_unstable=189, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7331166, upper bound: 106.7321800
time: 9.92 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319687, upper bound: 106.7319687
time: 8.80 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 20.29 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 20.29
Output dim: 0, lower bound: -106.7205639, upper bound: 106.7228340
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 20.29
Output dim: 0, lower bound: -106.7361426, upper bound: 106.7339479
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 20.29
Output dim: 0, lower bound: -106.7205029, upper bound: 106.7225731
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 20.29
Output dim: 0, lower bound: -106.7358983, upper bound: 106.7327698
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 20.29
Output dim: 0, lower bound: -106.7214853, upper bound: 106.7159808
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 20.29
Output dim: 0, lower bound: -106.7209231, upper bound: 106.7158192
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 20.29
Output dim: 0, lower bound: -106.7331166, upper bound: 106.7321800
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 20.29
Output dim: 0, lower bound: -106.7319687, upper bound: 106.7319687

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -26.9957714, 21.5279961, -35.3646851, 28.2104492, -55.2062225, 56.8926735
1: -21.4769115, 18.5021420, -28.4536438, 24.3343830, -45.8112907, 46.9557838
2: -28.8063183, 18.6599503, -37.9704590, 24.5304356, -53.3367538, 56.6304092
3: -30.2951717, 16.0209675, -40.0986443, 21.1298409, -51.4250107, 56.1196136
4: -28.5431118, 21.6506538, -37.5689201, 28.3675213, -56.9106331, 59.2195702
5: -25.7069683, 20.4726715, -33.9173622, 26.8446159, -52.5515823, 54.3900337
6: -24.4729385, 23.7288704, -32.1499901, 31.0261974, -55.4991379, 55.8788605
7: -26.2193050, 22.6755047, -34.3932381, 29.7284260, -55.9477310, 57.0687408
8: -31.3422241, 20.8133163, -41.3485680, 27.4296513, -58.7718735, 62.1618843
9: -24.3412437, 23.5257301, -31.8787251, 30.9115620, -55.2528076, 55.4044418

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=63, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=146, inp2_unstable=148, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=181, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7099831, upper bound: 106.7142188
time: 9.41 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7098455, upper bound: 106.7131605
time: 9.00 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -33.3200150, 26.5823078, -39.6654892, 31.6172333, -64.9372406, 66.2477951
1: -26.7578011, 22.9192848, -32.0245438, 27.3153019, -54.0731010, 54.9438248
2: -35.7314110, 23.1726189, -42.6806641, 27.5884781, -63.3198891, 65.8532639
3: -37.7270775, 19.9066143, -45.1123810, 23.7436314, -61.4707108, 65.0189896
4: -35.4014511, 26.7457314, -42.2029762, 31.8370152, -67.2384644, 68.9487000
5: -31.8808517, 25.2663097, -38.0951347, 30.0579700, -61.9388199, 63.3614311
6: -30.3533936, 29.2491856, -36.1102371, 34.7541695, -65.1075516, 65.3594131
7: -32.4427376, 28.0131454, -38.6414757, 33.3279648, -65.7707062, 66.6546097
8: -38.9699593, 25.8568001, -46.5749245, 30.9052410, -69.8751984, 72.4317093
9: -30.0644608, 29.1836205, -35.7313843, 34.7485466, -64.8130035, 64.9150085

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=63, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=147, inp2_unstable=149, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=174, inp2_unstable=190, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 240

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7055833, upper bound: 106.7190603
time: 10.09 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7208024, upper bound: 106.7177235
time: 8.98 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -24.0892925, 19.2044907, -34.8429337, 27.7701416, -51.8594322, 54.0474243
1: -19.0461349, 16.4932728, -27.9839668, 23.9428101, -42.9889412, 44.4772377
2: -25.6340561, 16.6217728, -37.3679924, 24.1092129, -49.7432709, 53.9897652
3: -26.9205036, 14.2887573, -39.5014153, 20.7840137, -47.7045174, 53.7901726
4: -25.4096088, 19.3177166, -36.9599419, 27.9540386, -53.3636360, 56.2776566
5: -22.8679161, 18.2798080, -33.3967285, 26.4065285, -49.2744446, 51.6765366
6: -21.8131008, 21.1807365, -31.6456013, 30.5662041, -52.3793030, 52.8263321
7: -23.3755836, 20.2423000, -33.8698120, 29.2671738, -52.6427574, 54.1121140
8: -27.8913364, 18.5256157, -40.6608810, 26.9660225, -54.8573532, 59.1864891
9: -21.7051945, 20.9425716, -31.3918495, 30.4026031, -52.1077957, 52.3344193

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=63, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=146, inp2_unstable=157, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=139, inp2_unstable=177, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7098457, upper bound: 106.7140513
time: 10.10 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7097096, upper bound: 106.7127839
time: 9.34 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -29.8976784, 23.8478050, -39.0317345, 31.0910835, -60.9887581, 62.8795395
1: -23.9091663, 20.5317116, -31.4471092, 26.8558865, -50.7650452, 51.9788208
2: -31.9769135, 20.7696762, -41.9686928, 27.0829449, -59.0598526, 62.7383690
3: -33.7357178, 17.8150349, -44.3800774, 23.3375015, -57.0732155, 62.1951141
4: -31.7052631, 23.9926414, -41.4635010, 31.3260708, -63.0313301, 65.4561462
5: -28.5436974, 22.6941681, -37.4648056, 29.5392265, -58.0829124, 60.1589737
6: -27.2051868, 26.2437420, -35.5026627, 34.2005119, -61.4057007, 61.7463989
7: -29.0928135, 25.1457291, -37.9859352, 32.7875252, -61.8803406, 63.1316643
8: -34.8609428, 23.1047668, -45.7419701, 30.3264904, -65.1874237, 68.8467331
9: -26.9664726, 26.1284008, -35.1478310, 34.1319275, -61.0983963, 61.2762222

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=63, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=147, inp2_unstable=157, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=161, inp2_unstable=189, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7210084, upper bound: 106.7181599
time: 10.23 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7206317, upper bound: 106.7165899
time: 9.39 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -20.1126461, 16.0194283, -29.5011196, 23.5208721, -43.6335144, 45.5205460
1: -15.6518440, 13.7178354, -23.5418453, 20.2162914, -35.8681297, 37.2596817
2: -21.2493458, 13.8934927, -31.4927998, 20.3536701, -41.6030159, 45.3862839
3: -22.3775387, 11.9015656, -33.1789818, 17.5287094, -39.9062462, 45.0805473
4: -21.1285667, 16.0514183, -31.1617870, 23.6505070, -44.7790756, 47.2132034
5: -18.9544697, 15.2378941, -28.1792030, 22.3918915, -41.3463593, 43.4170914
6: -18.2389011, 17.6378098, -26.7079620, 25.9065208, -44.1454201, 44.3457680
7: -19.4972973, 16.8888149, -28.6029015, 24.7474670, -44.2447662, 45.4917145
8: -23.1870499, 15.3893232, -34.2479324, 22.7460823, -45.9331245, 49.6372566
9: -18.0501575, 17.4099998, -26.5524902, 25.6646862, -43.7148438, 43.9624901

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=63, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=154, inp2_unstable=147, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=121, inp2_unstable=160, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 254

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7124142, upper bound: 106.7083499
time: 9.18 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7122656, upper bound: 106.7066360
time: 8.01 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -20.5351601, 16.3365402, -26.7706451, 21.3403530, -41.8755112, 43.1071777
1: -15.9746342, 13.9935007, -21.2551384, 18.3198051, -34.2944336, 35.2486343
2: -21.6943054, 14.1537418, -28.5072823, 18.4390278, -40.1333313, 42.6610222
3: -22.8717899, 12.1380672, -29.9918556, 15.8683491, -38.7401390, 42.1299210
4: -21.5554943, 16.3949165, -28.2162704, 21.4503345, -43.0058289, 44.6111870
5: -19.3479347, 15.5180016, -25.5131283, 20.3349190, -39.6828537, 41.0311241
6: -18.6202106, 18.0147762, -24.1983891, 23.5040894, -42.1242943, 42.2131653
7: -19.9093761, 17.2184010, -25.9282341, 22.4500256, -42.3594017, 43.1466370
8: -23.6797295, 15.6935968, -30.9797535, 20.5834827, -44.2632103, 46.6733475
9: -18.4146080, 17.7703285, -24.0691071, 23.2338333, -41.6484413, 41.8394356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=63, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=160, inp2_unstable=147, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=120, inp2_unstable=149, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 111

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7116308, upper bound: 106.7080405
time: 9.12 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7115127, upper bound: 106.7064157
time: 9.14 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -22.9201889, 18.2385597, -36.4141846, 29.0323143, -51.9524956, 54.6527443
1: -18.0129681, 15.6695061, -29.2864952, 25.0455933, -43.0585632, 44.9560013
2: -24.3034382, 15.8932848, -39.0789642, 25.2695236, -49.5729599, 54.9722481
3: -25.6431770, 13.6194019, -41.2926331, 21.7853775, -47.4285507, 54.9120331
4: -24.1812000, 18.2996902, -38.6475906, 29.2134399, -53.3946342, 56.9472809
5: -21.6691113, 17.3736343, -34.9129791, 27.6055107, -49.2746201, 52.2866096
6: -20.8197842, 20.0805416, -33.1104927, 31.9296741, -52.7494545, 53.1910324
7: -22.2840919, 19.2735691, -35.3937988, 30.5721760, -52.8562622, 54.6673622
8: -26.5287151, 17.5605354, -42.6029205, 28.2946415, -54.8233566, 60.1634560
9: -20.5992908, 19.9207859, -32.7986603, 31.8436050, -52.4428825, 52.7194405

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=63, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=156, inp2_unstable=147, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=134, inp2_unstable=183, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7243811, upper bound: 106.7247474
time: 8.76 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7239363, upper bound: 106.7229655
time: 8.55 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -23.2609291, 18.4865112, -33.1956902, 26.4661980, -49.7271233, 51.6821976
1: -18.2581863, 15.8819284, -26.6132755, 22.8022003, -41.0603790, 42.4952049
2: -24.6476498, 16.0887527, -35.5451164, 23.0070190, -47.6546707, 51.6338692
3: -26.0274696, 13.8059540, -37.5401802, 19.8274899, -45.8549576, 51.3461342
4: -24.5050793, 18.5718517, -35.1749535, 26.6232643, -51.1283379, 53.7468033
5: -21.9763451, 17.5819817, -31.7809849, 25.1959648, -47.1723061, 49.3629684
6: -21.1096497, 20.3818245, -30.1598759, 29.1119728, -50.2216225, 50.5416946
7: -22.6100197, 19.5300560, -32.2521400, 27.8758163, -50.4858360, 51.7821960
8: -26.9089260, 17.8018475, -38.7172585, 25.6944828, -52.6034088, 56.5191040
9: -20.8871803, 20.2050629, -29.8933983, 28.9669590, -49.8541412, 50.0984573

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=63, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=161, inp2_unstable=147, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=173, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7231274, upper bound: 106.7244227
time: 8.69 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7227362, upper bound: 106.7227362
time: 8.25 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 18.48 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.48
Output dim: 0, lower bound: -106.7099831, upper bound: 106.7142188
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.48
Output dim: 0, lower bound: -106.7098455, upper bound: 106.7131605
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.48
Output dim: 0, lower bound: -106.7055833, upper bound: 106.7190603
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.48
Output dim: 0, lower bound: -106.7208024, upper bound: 106.7177235
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.48
Output dim: 0, lower bound: -106.7098457, upper bound: 106.7140513
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.48
Output dim: 0, lower bound: -106.7097096, upper bound: 106.7127839
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.48
Output dim: 0, lower bound: -106.7210084, upper bound: 106.7181599
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.48
Output dim: 0, lower bound: -106.7206317, upper bound: 106.7165899
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.48
Output dim: 0, lower bound: -106.7124142, upper bound: 106.7083499
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.48
Output dim: 0, lower bound: -106.7122656, upper bound: 106.7066360
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.48
Output dim: 0, lower bound: -106.7116308, upper bound: 106.7080405
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.48
Output dim: 0, lower bound: -106.7115127, upper bound: 106.7064157
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.48
Output dim: 0, lower bound: -106.7243811, upper bound: 106.7247474
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.48
Output dim: 0, lower bound: -106.7239363, upper bound: 106.7229655
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.48
Output dim: 0, lower bound: -106.7231274, upper bound: 106.7244227
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.48
Output dim: 0, lower bound: -106.7227362, upper bound: 106.7227362

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -25.4199505, 20.2968483, -28.9622650, 23.2001629, -48.6201057, 49.2591133
1: -20.1995220, 17.4381790, -23.2656231, 19.9802895, -40.1798096, 40.7038040
2: -27.1248112, 17.5865517, -31.0849590, 20.1585255, -47.2833366, 48.6715088
3: -28.5082645, 15.0946131, -32.7896004, 17.3009853, -45.8092499, 47.8842125
4: -26.8987598, 20.4150085, -30.8510494, 23.3263035, -50.2250633, 51.2660522
5: -24.1923218, 19.3153191, -27.7602749, 22.1553154, -46.3476295, 47.0755844
6: -23.0610428, 22.3635635, -26.3889866, 25.4674110, -48.5284500, 48.7525482
7: -24.7176323, 21.3948288, -28.2774162, 24.4914341, -49.2090607, 49.6722450
8: -29.5135212, 19.6060543, -33.8295097, 22.4497147, -51.9632339, 53.4355621
9: -22.9707584, 22.1816368, -26.2808723, 25.4192810, -48.3900375, 48.4625092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=62, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=146, inp2_unstable=147, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=149, inp2_unstable=168, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7082090, upper bound: 106.7125491
time: 9.22 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7080054, upper bound: 106.7124384
time: 9.25 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -23.4622898, 18.7623463, -30.6349583, 24.5389099, -48.0011978, 49.3973045
1: -18.5980072, 16.1029072, -24.6815300, 21.1372433, -39.7352448, 40.7844391
2: -25.0181789, 16.2490101, -32.9209938, 21.3874092, -46.4055862, 49.1699982
3: -26.2810631, 13.9287529, -34.7520943, 18.3368969, -44.6179581, 48.6808434
4: -24.8229065, 18.8751869, -32.7152023, 24.6609325, -49.4838371, 51.5903854
5: -22.3056717, 17.8700085, -29.3817616, 23.4308739, -45.7365456, 47.2517700
6: -21.3214970, 20.6502190, -27.9872856, 26.9206238, -48.2421188, 48.6375008
7: -22.8379784, 19.7985992, -29.9916687, 25.9449234, -48.7828979, 49.7902679
8: -27.2327518, 18.0997982, -35.8534698, 23.8238525, -51.0566025, 53.9532623
9: -21.2570553, 20.4954262, -27.8316765, 27.0107441, -48.2677994, 48.3270950

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=62, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=146, inp2_unstable=154, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=140, inp2_unstable=173, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7080435, upper bound: 106.7113979
time: 8.76 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7077708, upper bound: 106.7112561
time: 11.18 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -27.6762810, 22.0559826, -24.2131805, 19.2533932, -46.9296646, 46.2691650
1: -22.0445633, 18.9762573, -19.1659737, 16.5571918, -38.6017532, 38.1422310
2: -29.5364552, 19.2347164, -25.7258968, 16.8020802, -46.3385353, 44.9606094
3: -31.1606102, 16.4955292, -27.1809673, 14.4567547, -45.6173630, 43.6764908
4: -29.3129807, 22.1844959, -25.5748425, 19.3458099, -48.6587906, 47.7593384
5: -26.3504639, 21.0114727, -23.0028076, 18.4415054, -44.7919693, 44.0142822
6: -25.1709576, 24.2679520, -22.0053139, 21.1791019, -46.3500557, 46.2732506
7: -26.9274330, 23.2755241, -23.5212917, 20.3737831, -47.3012161, 46.7968102
8: -32.2206688, 21.3390980, -28.0580463, 18.5030479, -50.7237167, 49.3971443
9: -24.9396725, 24.1664772, -21.7437706, 21.0284920, -45.9681625, 45.9102402

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=62, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=147, inp2_unstable=146, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=143, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7132700, upper bound: 106.7100578
time: 9.98 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7105392, upper bound: 106.7093552
time: 8.96 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -26.8485527, 21.3862514, -26.3563786, 20.9650383, -47.8135910, 47.7426300
1: -21.3512936, 18.4002438, -20.9579296, 18.0417442, -39.3930359, 39.3581696
2: -28.6298599, 18.6467533, -28.0852833, 18.2711067, -46.9009666, 46.7320251
3: -30.1944160, 15.9961681, -29.6797352, 15.7406979, -45.9351120, 45.6759033
4: -28.4243259, 21.5134983, -27.9009323, 21.0905266, -49.5148506, 49.4144287
5: -25.5322781, 20.3795700, -25.0730686, 20.0448265, -45.5771027, 45.4526367
6: -24.4077778, 23.5417728, -23.9610462, 23.0857639, -47.4935341, 47.5028152
7: -26.1290379, 22.5799580, -25.6603603, 22.1858044, -48.3148422, 48.2403183
8: -31.2273426, 20.6694393, -30.6203499, 20.1767044, -51.4040451, 51.2897873
9: -24.1918945, 23.4311275, -23.7092209, 22.9282608, -47.1201515, 47.1403465

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=62, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=147, inp2_unstable=152, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=152, inp2_unstable=149, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7128576, upper bound: 106.7089504
time: 9.63 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7101499, upper bound: 106.7082230
time: 8.47 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -22.6305408, 18.0681286, -28.4823265, 22.7907085, -45.4212494, 46.5504532
1: -17.8616238, 15.5115385, -22.8337727, 19.6164970, -37.4781151, 38.3453102
2: -24.0807800, 15.6275969, -30.5261478, 19.7656574, -43.8464355, 46.1537399
3: -25.2804756, 13.4275217, -32.2367249, 16.9806309, -42.2611084, 45.6642418
4: -23.8839989, 18.1808300, -30.2816238, 22.9482021, -46.8321953, 48.4624557
5: -21.4714851, 17.2063980, -27.2806129, 21.7457066, -43.2171898, 44.4870110
6: -20.5227566, 19.9105396, -25.9122105, 25.0481548, -45.5709114, 45.8227425
7: -21.9835854, 19.0565033, -27.7845688, 24.0639896, -46.0475693, 46.8410721
8: -26.2071705, 17.4144802, -33.1903229, 22.0136490, -48.2208176, 50.6048050
9: -20.4412174, 19.6993656, -25.8251133, 24.9496746, -45.3908844, 45.5244751

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=62, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=146, inp2_unstable=155, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=164, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7080416, upper bound: 106.7122333
time: 9.94 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7078746, upper bound: 106.7121821
time: 8.01 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -20.8287792, 16.6630268, -29.8888187, 23.9223995, -44.7511787, 46.5518456
1: -16.3743076, 14.2810850, -24.0221996, 20.5952606, -36.9695625, 38.3032837
2: -22.1331749, 14.3963461, -32.0818214, 20.8046932, -42.9378586, 46.4781647
3: -23.2316475, 12.3504953, -33.9015999, 17.8556156, -41.0872574, 46.2520905
4: -21.9704037, 16.7523270, -31.8669682, 24.0574226, -46.0278206, 48.6192932
5: -19.7352371, 15.8867941, -28.6394844, 22.8225708, -42.5578003, 44.5262794
6: -18.9114017, 18.3239212, -27.2808781, 26.2680664, -45.1794662, 45.6047974
7: -20.2493744, 17.5859280, -29.2430573, 25.3000317, -45.5494080, 46.8289871
8: -24.0951309, 16.0252323, -34.9096794, 23.1828156, -47.2779350, 50.9349098
9: -18.8665276, 18.1383820, -27.1476059, 26.3008385, -45.1673660, 45.2859840

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=62, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=146, inp2_unstable=162, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=129, inp2_unstable=169, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7078763, upper bound: 106.7109210
time: 9.77 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7076558, upper bound: 106.7108598
time: 8.73 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -24.7049465, 19.6785145, -24.1821976, 19.2078876, -43.9128342, 43.8607101
1: -19.5662022, 16.9207382, -19.0962524, 16.5188103, -36.0850067, 36.0169907
2: -26.2885647, 17.1498852, -25.6688042, 16.7365417, -43.0251083, 42.8186836
3: -27.7131271, 14.7177601, -27.1417999, 14.4112377, -42.1243668, 41.8595581
4: -26.1153755, 19.7913895, -25.4913960, 19.3307686, -45.4461403, 45.2827759
5: -23.4541054, 18.7685490, -22.9535522, 18.3779659, -41.8320656, 41.7220993
6: -22.4489651, 21.6604195, -21.9489517, 21.1543331, -43.6032982, 43.6093636
7: -24.0289860, 20.7874870, -23.4716568, 20.3165665, -44.3455505, 44.2591400
8: -28.6801853, 18.9807720, -27.9843216, 18.4638596, -47.1440430, 46.9650879
9: -22.2490215, 21.5322380, -21.6967735, 20.9726925, -43.2217140, 43.2290077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=62, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=147, inp2_unstable=153, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=142, inp2_unstable=141, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7129620, upper bound: 106.7090198
time: 9.37 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7102897, upper bound: 106.7083145
time: 8.58 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -23.9339027, 19.0627651, -25.6319981, 20.3655682, -44.2994690, 44.6947594
1: -18.9196968, 16.3865185, -20.3196583, 17.5154705, -36.4351654, 36.7061729
2: -25.4448109, 16.6037769, -27.2632027, 17.7215366, -43.1663399, 43.8669815
3: -26.8162727, 14.2560320, -28.8204098, 15.2772751, -42.0935402, 43.0764351
4: -25.2835693, 19.1760254, -27.0671482, 20.5063000, -45.7898674, 46.2431717
5: -22.6965942, 18.1878834, -24.3450127, 19.4600830, -42.1566734, 42.5328979
6: -21.7449245, 20.9835949, -23.2691307, 22.4475002, -44.1924248, 44.2527237
7: -23.2801323, 20.1432381, -24.9240551, 21.5463200, -44.8264503, 45.0672913
8: -27.7625923, 18.3686695, -29.7253246, 19.5832386, -47.3458328, 48.0939941
9: -21.5539722, 20.8430729, -23.0257072, 22.2598877, -43.8138580, 43.8687820

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=62, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=147, inp2_unstable=155, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=138, inp2_unstable=142, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7126155, upper bound: 106.7076398
time: 9.95 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7098897, upper bound: 106.7069712
time: 9.02 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -18.8075047, 15.0088530, -23.6650734, 18.9625149, -37.7700195, 38.6739273
1: -14.5800047, 12.8323603, -18.7965221, 16.2648449, -30.8448486, 31.6288795
2: -19.8618279, 13.0068922, -25.2610950, 16.3804760, -36.2423019, 38.2679710
3: -20.9018784, 11.1269665, -26.5449467, 14.0629206, -34.9647942, 37.6719093
4: -19.7629299, 15.0287485, -25.0668964, 19.0588036, -38.8217278, 40.0956383
5: -17.7025928, 14.2782564, -22.5653572, 18.1010971, -35.8036880, 36.8436127
6: -17.0625248, 16.5056610, -21.4877472, 20.8362503, -37.8987732, 37.9934006
7: -18.2401180, 15.8249969, -23.0315990, 19.9861240, -38.2262421, 38.8565903
8: -21.6821022, 14.4072542, -27.4686718, 18.2667885, -39.9488869, 41.8759270
9: -16.9180813, 16.2837925, -21.4637337, 20.6823292, -37.6004105, 37.7475243

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=62, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=154, inp2_unstable=146, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=116, inp2_unstable=141, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7106326, upper bound: 106.7064994
time: 8.82 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7106188, upper bound: 106.7064985
time: 10.93 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -17.4854298, 13.9707317, -25.1708927, 20.1632290, -37.6486588, 39.1416245
1: -13.4781399, 11.9139013, -20.0825367, 17.2911072, -30.7692471, 31.9964371
2: -18.4306374, 12.1048613, -26.8974838, 17.4737148, -35.9043503, 39.0023384
3: -19.3837090, 10.3374834, -28.3205929, 14.9747286, -34.3584366, 38.6580696
4: -18.3505955, 13.9806976, -26.7338772, 20.2662029, -38.6167984, 40.7145691
5: -16.4279633, 13.2889347, -24.0267124, 19.2441349, -35.6720963, 37.3156471
6: -15.8587379, 15.3481302, -22.9111786, 22.1578960, -38.0166283, 38.2593040
7: -16.9451237, 14.7428160, -24.5448303, 21.3186855, -38.2638092, 39.2876472
8: -20.1367035, 13.4065838, -29.2463894, 19.4491482, -39.5858421, 42.6529655
9: -15.7521200, 15.1268806, -22.8770428, 22.0658970, -37.8180161, 38.0039215

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=62, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=154, inp2_unstable=154, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=147, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7103867, upper bound: 106.7047992
time: 9.37 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7103723, upper bound: 106.7048060
time: 10.16 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -19.2365551, 15.3298225, -21.2848225, 17.0581017, -36.2946548, 36.6146469
1: -14.9093456, 13.1173868, -16.7899303, 14.6225653, -29.5319099, 29.9073143
2: -20.3150253, 13.2715082, -22.6640396, 14.7114010, -35.0264282, 35.9355469
3: -21.4085693, 11.3678150, -23.7941170, 12.6392431, -34.0478134, 35.1619339
4: -20.1978836, 15.3765535, -22.4966602, 17.1539268, -37.3518066, 37.8732147
5: -18.1014957, 14.5632534, -20.2375927, 16.3011799, -34.4026718, 34.8008461
6: -17.4516754, 16.8906898, -19.3185196, 18.7366180, -36.1882935, 36.2092094
7: -18.6592026, 16.1602592, -20.6966209, 17.9891453, -36.6483421, 36.8568802
8: -22.1820450, 14.7171106, -24.6456165, 16.3955460, -38.5775909, 39.3627281
9: -17.2884407, 16.6511250, -19.3078785, 18.5537949, -35.8422356, 35.9590034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=62, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=160, inp2_unstable=146, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=115, inp2_unstable=133, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7098313, upper bound: 106.7061463
time: 8.81 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7097525, upper bound: 106.7061288
time: 10.64 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -17.8769836, 14.2649250, -22.6602268, 18.1570415, -36.0340271, 36.9251518
1: -13.7793379, 12.1739941, -17.9773865, 15.5574570, -29.3367882, 30.1513786
2: -18.8469219, 12.3467331, -24.1589565, 15.7144728, -34.5613937, 36.5056915
3: -19.8478107, 10.5571976, -25.4127979, 13.4654160, -33.3132248, 35.9699898
4: -18.7476730, 14.2996168, -24.0263309, 18.2572155, -37.0048866, 38.3259430
5: -16.7899952, 13.5467644, -21.5742111, 17.3412895, -34.1312790, 35.1209641
6: -16.2171993, 15.7013159, -20.6237526, 19.9470730, -36.1642723, 36.3250618
7: -17.3318195, 15.0500727, -22.0828228, 19.2153091, -36.5471268, 37.1328926
8: -20.5928059, 13.6930161, -26.2780209, 17.4777145, -38.0705185, 39.9710388
9: -16.0964985, 15.4670992, -20.6040764, 19.8208408, -35.9173317, 36.0711708

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=62, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=160, inp2_unstable=154, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=136, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7097194, upper bound: 106.7045577
time: 8.87 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7096454, upper bound: 106.7045541
time: 10.05 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -21.5100422, 17.1417160, -29.9080944, 23.9410992, -45.4511414, 47.0498123
1: -16.8581791, 14.7133417, -24.0071259, 20.6171780, -37.4753571, 38.7204666
2: -22.7965069, 14.9266481, -32.0795593, 20.8243179, -43.6208267, 47.0062065
3: -24.0395317, 12.7774601, -33.8640518, 17.8919334, -41.9314651, 46.6415100
4: -22.6942787, 17.1962700, -31.8186245, 24.0783310, -46.7726021, 49.0148888
5: -20.3199749, 16.3346748, -28.6504269, 22.8411503, -43.1611252, 44.9850998
6: -19.5581894, 18.8499832, -27.2581539, 26.2778893, -45.8360786, 46.1081390
7: -20.9286156, 18.1177120, -29.1738548, 25.2440434, -46.1726608, 47.2915649
8: -24.8915920, 16.4993610, -34.9496078, 23.2214489, -48.1130409, 51.4489670
9: -19.3694229, 18.7099724, -27.1044083, 26.2631588, -45.6325798, 45.8143768

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=62, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=147, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=126, inp2_unstable=171, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7225730, upper bound: 106.7230400
time: 7.96 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7226043, upper bound: 106.7230443
time: 8.30 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -20.0336037, 15.9910660, -31.5487061, 25.2508469, -45.2844505, 47.5397682
1: -15.6350908, 13.6979818, -25.4001465, 21.7515049, -37.3865967, 39.0981216
2: -21.2092533, 13.9165831, -33.8787117, 22.0292053, -43.2384567, 47.7952881
3: -22.3431625, 11.8874636, -35.7836189, 18.9035435, -41.2467041, 47.6710815
4: -21.1229687, 16.0329857, -33.6468506, 25.3857117, -46.5086784, 49.6798363
5: -18.8975964, 15.2457848, -30.2375355, 24.0887489, -42.9863434, 45.4833183
6: -18.2237930, 17.5612621, -28.8161564, 27.7057972, -45.9295883, 46.3774185
7: -19.4915123, 16.9113560, -30.8560123, 26.6683617, -46.1598663, 47.7673645
8: -23.1702576, 15.3852215, -36.9240570, 24.5690193, -47.7392769, 52.3092804
9: -18.0768032, 17.4198914, -28.6275196, 27.8177910, -45.8945885, 46.0474091

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=62, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=154, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=121, inp2_unstable=176, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7221237, upper bound: 106.7211809
time: 9.12 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7221397, upper bound: 106.7211940
time: 8.21 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -21.8575363, 17.4002094, -26.9483166, 21.5781918, -43.4357224, 44.3485260
1: -17.1099548, 14.9320650, -21.5482311, 18.5605888, -35.6705399, 36.4802971
2: -23.1501865, 15.1303892, -28.8378448, 18.7459450, -41.8961258, 43.9682350
3: -24.4361916, 12.9681587, -30.4225998, 16.0936108, -40.5298004, 43.3907509
4: -23.0282917, 17.4725780, -28.6244793, 21.7058086, -44.7340965, 46.0970497
5: -20.6384563, 16.5504436, -25.7740269, 20.6164932, -41.2549515, 42.3244705
6: -19.8580933, 19.1579952, -24.5453758, 23.6790218, -43.5371056, 43.7033691
7: -21.2626171, 18.3817635, -26.2883053, 22.7721958, -44.0348091, 44.6700668
8: -25.2884846, 16.7439289, -31.4036694, 20.8466434, -46.1351242, 48.1475983
9: -19.6634464, 19.0018654, -24.4330254, 23.6245308, -43.2879791, 43.4348907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=62, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=161, inp2_unstable=147, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=126, inp2_unstable=157, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7213963, upper bound: 106.7227157
time: 7.42 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7213420, upper bound: 106.7227009
time: 7.93 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 16.89 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 0, lower bound: -106.7082090, upper bound: 106.7125491
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 0, lower bound: -106.7080054, upper bound: 106.7124384
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 0, lower bound: -106.7080435, upper bound: 106.7113979
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 0, lower bound: -106.7077708, upper bound: 106.7112561
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 0, lower bound: -106.7132700, upper bound: 106.7100578
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 0, lower bound: -106.7105392, upper bound: 106.7093552
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 0, lower bound: -106.7128576, upper bound: 106.7089504
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 0, lower bound: -106.7101499, upper bound: 106.7082230
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 0, lower bound: -106.7080416, upper bound: 106.7122333
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 0, lower bound: -106.7078746, upper bound: 106.7121821
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 0, lower bound: -106.7078763, upper bound: 106.7109210
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 0, lower bound: -106.7076558, upper bound: 106.7108598
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 0, lower bound: -106.7129620, upper bound: 106.7090198
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 0, lower bound: -106.7102897, upper bound: 106.7083145
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 0, lower bound: -106.7126155, upper bound: 106.7076398
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 0, lower bound: -106.7098897, upper bound: 106.7069712
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 0, lower bound: -106.7106326, upper bound: 106.7064994
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 0, lower bound: -106.7106188, upper bound: 106.7064985
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 0, lower bound: -106.7103867, upper bound: 106.7047992
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 0, lower bound: -106.7103723, upper bound: 106.7048060
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 0, lower bound: -106.7098313, upper bound: 106.7061463
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 0, lower bound: -106.7097525, upper bound: 106.7061288
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 0, lower bound: -106.7097194, upper bound: 106.7045577
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 0, lower bound: -106.7096454, upper bound: 106.7045541
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 0, lower bound: -106.7225730, upper bound: 106.7230400
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 0, lower bound: -106.7226043, upper bound: 106.7230443
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 0, lower bound: -106.7221237, upper bound: 106.7211809
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 0, lower bound: -106.7221397, upper bound: 106.7211940
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 0, lower bound: -106.7213963, upper bound: 106.7227157
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 0, lower bound: -106.7213420, upper bound: 106.7227009
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.89
Output dim: 0, lower bound: -106.7227362, upper bound: 106.7227362

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 12.88 + 593.88 = 606.76 seconds
