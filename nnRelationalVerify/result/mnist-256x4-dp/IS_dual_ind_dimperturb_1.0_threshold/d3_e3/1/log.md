## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 2.411746211


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716)
1: (-0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653)
2: (-0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282)
3: (-1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487)
4: (-1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829)
5: (-1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234)
6: (-1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365)
7: (-1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017)
8: (-1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066)
9: (-1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.46 + 4.01 = 5.47 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -2.4863377, upper bound: 2.4863377

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 138

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4382318, upper bound: 2.4617431
time: 1.70 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4670166, upper bound: 2.4670167
time: 1.71 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 3.60 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 3.60
Output dim: 8, lower bound: -2.4382318, upper bound: 2.4617431
IS_A2, status: Status.UNKNOWN, split count: 1, time: 3.60
Output dim: 8, lower bound: -2.4670166, upper bound: 2.4670167

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.1493312, 0.2648046, -0.5499690, 0.6832625, -0.8325937, 0.8147736
1: -0.1491406, 0.1567767, -0.5121669, 0.5249491, -0.6740897, 0.6689436
2: -0.1610332, 0.2156616, -0.4431523, 0.7049769, -0.8660101, 0.6588140
3: -0.1086741, 0.1920385, -0.4242701, 0.6525689, -0.7612430, 0.6163086
4: -0.1606317, 0.1397428, -0.5540253, 0.5210850, -0.6817166, 0.6937681
5: -0.1443499, 0.1993984, -0.5044719, 0.5927411, -0.7370910, 0.7038703
6: -0.1449791, 0.1903342, -0.5048134, 0.5988603, -0.7438393, 0.6951476
7: -0.1878183, 0.1737633, -0.5440233, 0.6348198, -0.8226382, 0.7177866
8: 0.5766035, 1.0780331, -0.2623291, 1.1941999, -0.6175964, 1.3403622
9: -0.1896639, 0.2725694, -0.6070741, 0.6724687, -0.8621326, 0.8796436

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=66, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=62, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.3441655, upper bound: 2.3973992
time: 1.55 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.3431973, upper bound: 2.3742449
time: 1.50 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.3784711, 0.5267175, -0.8175827, 0.8354774, -1.2139485, 1.3443003
1: -0.3540550, 0.3617043, -0.7060200, 0.7148666, -1.0689216, 1.0677243
2: -0.3402389, 0.4843452, -0.6780660, 0.8963596, -1.2365985, 1.1624112
3: -0.2565480, 0.4747630, -0.7450740, 0.8218954, -1.0784434, 1.2198371
4: -0.3586341, 0.3588920, -0.8498437, 0.7276281, -1.0862622, 1.2087357
5: -0.3552166, 0.4340981, -0.7201439, 0.7827880, -1.1380045, 1.1542419
6: -0.3412417, 0.4440553, -0.7409224, 0.8005964, -1.1418381, 1.1849777
7: -0.3804360, 0.4354155, -0.7799670, 0.8755798, -1.2560159, 1.2153825
8: 0.1354375, 1.1458768, -0.7422276, 1.2464857, -1.1110482, 1.8881043
9: -0.4403357, 0.5109793, -0.8019977, 0.8782212, -1.3185568, 1.3129770

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=42, inp2_unstable=105, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4617431, upper bound: 2.4381191
time: 2.01 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4617431, upper bound: 2.4670167
time: 2.49 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 5.91 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 5.91
Output dim: 8, lower bound: -2.3441655, upper bound: 2.3973992
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 5.91
Output dim: 8, lower bound: -2.3431973, upper bound: 2.3742449
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.91
Output dim: 8, lower bound: -2.4617431, upper bound: 2.4381191
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 5.91
Output dim: 8, lower bound: -2.4617431, upper bound: 2.4670167

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.3784711, 0.5267175, -0.1493312, 0.2648046, -0.6432757, 0.6760488
1: -0.3540550, 0.3617043, -0.1491406, 0.1567767, -0.5108317, 0.5108449
2: -0.3402389, 0.4843452, -0.1610332, 0.2156616, -0.5559006, 0.6453784
3: -0.2565480, 0.4747630, -0.1086741, 0.1920385, -0.4485865, 0.5834371
4: -0.3586341, 0.3588920, -0.1606317, 0.1397428, -0.4983769, 0.5195237
5: -0.3552166, 0.4340981, -0.1443499, 0.1993984, -0.5546150, 0.5784479
6: -0.3412417, 0.4440553, -0.1449791, 0.1903342, -0.5315760, 0.5890343
7: -0.3804360, 0.4354155, -0.1878183, 0.1737633, -0.5541994, 0.6232338
8: 0.1354375, 1.1458768, 0.5766035, 1.0780331, -0.9425956, 0.5692732
9: -0.4403357, 0.5109793, -0.1896639, 0.2725694, -0.7129051, 0.7006432

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=66, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=42, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.3973992, upper bound: 2.3441655
time: 1.64 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.3742449, upper bound: 2.3431973
time: 1.65 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.3784711, 0.5267175, -0.3784711, 0.5267175, -0.9051886, 0.9051886
1: -0.3540550, 0.3617043, -0.3540550, 0.3617043, -0.7157593, 0.7157593
2: -0.3402389, 0.4843452, -0.3402389, 0.4843452, -0.8245841, 0.8245841
3: -0.2565480, 0.4747630, -0.2565480, 0.4747630, -0.7313111, 0.7313111
4: -0.3586341, 0.3588920, -0.3586341, 0.3588920, -0.7175261, 0.7175261
5: -0.3552166, 0.4340981, -0.3552166, 0.4340981, -0.7893147, 0.7893147
6: -0.3412417, 0.4440553, -0.3412417, 0.4440553, -0.7852970, 0.7852970
7: -0.3804360, 0.4354155, -0.3804360, 0.4354155, -0.8158515, 0.8158515
8: 0.1354375, 1.1458768, 0.1354375, 1.1458768, -1.0104393, 1.0104393
9: -0.4403357, 0.5109793, -0.4403357, 0.5109793, -0.9513149, 0.9513149

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=42, inp2_unstable=42, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.3973992, upper bound: 2.3846697
time: 1.68 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.3742449, upper bound: 2.3836684
time: 1.73 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.82 seconds
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 4.82
Output dim: 8, lower bound: -2.3973992, upper bound: 2.3441655
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 4.82
Output dim: 8, lower bound: -2.3742449, upper bound: 2.3431973
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 4.82
Output dim: 8, lower bound: -2.3973992, upper bound: 2.3846697
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 4.82
Output dim: 8, lower bound: -2.3742449, upper bound: 2.3836684

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 5.47 + 23.66 = 29.13 seconds
