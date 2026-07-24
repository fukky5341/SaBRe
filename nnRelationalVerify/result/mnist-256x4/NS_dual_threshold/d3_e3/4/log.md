## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.51975288


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692)
1: (-0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081)
2: (-0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965)
3: (-0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550)
4: (-0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409)
5: (-0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894)
6: (-0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582)
7: (-0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213)
8: (-0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886)
9: (-0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.76 + 3.05 = 3.81 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.5775032, upper bound: 0.5775021

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5661096, upper bound: 0.5557030
time: 3.59 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5556276, upper bound: 0.5556265
time: 1.78 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 5.43 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 5.43
Output dim: 0, lower bound: -0.5661096, upper bound: 0.5557030
NS_A2, status: Status.UNKNOWN, split count: 1, time: 5.43
Output dim: 0, lower bound: -0.5556276, upper bound: 0.5556265

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 0.6588491, 1.0098670, 0.4363009, 1.0305700, -0.3717209, 0.5735661
1: -0.0652878, 0.0725599, -0.0946363, 0.1140718, -0.1793597, 0.1671962
2: -0.0284445, 0.1199432, -0.0521084, 0.1664880, -0.1949326, 0.1720517
3: -0.0606395, 0.0949028, -0.0998731, 0.1264820, -0.1871214, 0.1947758
4: -0.0926590, 0.0525676, -0.1233485, 0.0905924, -0.1832514, 0.1759162
5: -0.0826193, 0.1864108, -0.1413220, 0.2247675, -0.3073867, 0.3277328
6: -0.0619781, 0.0912315, -0.0816977, 0.1669606, -0.2289386, 0.1729291
7: -0.0992174, 0.1211332, -0.1244795, 0.2396419, -0.3388593, 0.2456127
8: -0.0712993, 0.1051628, -0.1047466, 0.1465420, -0.2178413, 0.2099094
9: -0.0731309, 0.1082365, -0.1048256, 0.1604646, -0.2335954, 0.2130621

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 78

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5556276, upper bound: 0.5556265
time: 1.54 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5556276, upper bound: 0.5556265
time: 2.30 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 0.5589638, 1.0477290, 0.6151519, 1.0147618, -0.4557980, 0.4325771
1: -0.1970288, 0.1863454, -0.0708233, 0.0818076, -0.2788364, 0.2571687
2: -0.1132474, 0.2866830, -0.0332855, 0.1300832, -0.2433306, 0.3199686
3: -0.1250774, 0.1984940, -0.0683888, 0.1020437, -0.2271211, 0.2668828
4: -0.2207092, 0.1613820, -0.0988819, 0.0606914, -0.2814006, 0.2602639
5: -0.2249858, 0.2884768, -0.0951682, 0.1944770, -0.4194628, 0.3836450
6: -0.1418439, 0.2156764, -0.0663497, 0.1051258, -0.2469697, 0.2820261
7: -0.2380447, 0.2181820, -0.1049182, 0.1453690, -0.3834137, 0.3231003
8: -0.2150748, 0.2233016, -0.0785607, 0.1130246, -0.3280994, 0.3018623
9: -0.1826335, 0.2746370, -0.0800931, 0.1184551, -0.3010886, 0.3547301

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 245

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5549100, upper bound: 0.5553013
time: 1.76 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5549459, upper bound: 0.5549448
time: 8.96 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 11.48 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 11.48
Output dim: 0, lower bound: -0.5556276, upper bound: 0.5556265
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 11.48
Output dim: 0, lower bound: -0.5556276, upper bound: 0.5556265
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 11.48
Output dim: 0, lower bound: -0.5549100, upper bound: 0.5553013
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 11.48
Output dim: 0, lower bound: -0.5549459, upper bound: 0.5549448

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: 0.6588491, 1.0098670, 0.6588491, 1.0098670, -0.3510178, 0.3510178
1: -0.0652878, 0.0725599, -0.0652878, 0.0725599, -0.1378478, 0.1378478
2: -0.0284445, 0.1199432, -0.0284445, 0.1199432, -0.1483878, 0.1483878
3: -0.0606395, 0.0949028, -0.0606395, 0.0949028, -0.1555422, 0.1555422
4: -0.0926590, 0.0525676, -0.0926590, 0.0525676, -0.1452266, 0.1452266
5: -0.0826193, 0.1864108, -0.0826193, 0.1864108, -0.2690301, 0.2690301
6: -0.0619781, 0.0912315, -0.0619781, 0.0912315, -0.1532095, 0.1532095
7: -0.0992174, 0.1211332, -0.0992174, 0.1211332, -0.2203506, 0.2203506
8: -0.0712993, 0.1051628, -0.0712993, 0.1051628, -0.1764621, 0.1764621
9: -0.0731309, 0.1082365, -0.0731309, 0.1082365, -0.1813673, 0.1813673

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of NS_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5655256, upper bound: 0.5553728
time: 1.55 seconds

## Relational analysis of NS_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5655157, upper bound: 0.5550132
time: 1.85 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: 0.6588491, 1.0098670, 0.5589638, 1.0477290, -0.3888799, 0.4509032
1: -0.0652878, 0.0725599, -0.1970288, 0.1863454, -0.2516332, 0.2695887
2: -0.0284445, 0.1199432, -0.1132474, 0.2866830, -0.3151276, 0.2331906
3: -0.0606395, 0.0949028, -0.1250774, 0.1984940, -0.2591334, 0.2199802
4: -0.0926590, 0.0525676, -0.2207092, 0.1613820, -0.2540410, 0.2732768
5: -0.0826193, 0.1864108, -0.2249858, 0.2884768, -0.3710960, 0.4113967
6: -0.0619781, 0.0912315, -0.1418439, 0.2156764, -0.2776544, 0.2330754
7: -0.0992174, 0.1211332, -0.2380447, 0.2181820, -0.3173994, 0.3591779
8: -0.0712993, 0.1051628, -0.2150748, 0.2233016, -0.2946009, 0.3202376
9: -0.0731309, 0.1082365, -0.1826335, 0.2746370, -0.3477679, 0.2908700

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 93

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5656714, upper bound: 0.5549811
time: 1.78 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5655157, upper bound: 0.5550132
time: 1.81 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 0.5737605, 1.0451070, 0.6766188, 1.0094959, -0.4357353, 0.3684883
1: -0.1878469, 0.1780573, -0.0485740, 0.0534229, -0.2412698, 0.2266313
2: -0.1072832, 0.2750047, -0.0146381, 0.0998419, -0.2071252, 0.2896428
3: -0.1198316, 0.1912155, -0.0494041, 0.0772163, -0.1970479, 0.2406196
4: -0.2115348, 0.1537331, -0.0732921, 0.0324999, -0.2440347, 0.2270252
5: -0.2142647, 0.2806869, -0.0564635, 0.1676525, -0.3819171, 0.3371505
6: -0.1364323, 0.2052122, -0.0470036, 0.0818600, -0.2182923, 0.2522159
7: -0.2287775, 0.2076855, -0.0772785, 0.0987685, -0.3275460, 0.2849640
8: -0.2053939, 0.2141977, -0.0506771, 0.0901324, -0.2955263, 0.2648748
9: -0.1747209, 0.2630653, -0.0553269, 0.0747138, -0.2494347, 0.3183922

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 245

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5334189, upper bound: 0.5370422
time: 1.67 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5333943, upper bound: 0.5334403
time: 1.87 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 0.5775799, 1.0444301, 0.6039896, 1.0126011, -0.4350213, 0.4404405
1: -0.1854768, 0.1759178, -0.0803506, 0.0933281, -0.2788050, 0.2562684
2: -0.1057437, 0.2719904, -0.0407744, 0.1458090, -0.2515527, 0.3127648
3: -0.1184776, 0.1893367, -0.0739673, 0.1129888, -0.2314664, 0.2633040
4: -0.2091668, 0.1517586, -0.1085290, 0.0716918, -0.2808585, 0.2602877
5: -0.2114975, 0.2786761, -0.1097437, 0.2033951, -0.4148926, 0.3884198
6: -0.1350354, 0.2025111, -0.0743863, 0.1109426, -0.2459780, 0.2768974
7: -0.2263854, 0.2049762, -0.1176244, 0.1569286, -0.3833140, 0.3226005
8: -0.2028952, 0.2118476, -0.0924377, 0.1203881, -0.3232832, 0.3042853
9: -0.1726784, 0.2600786, -0.0903493, 0.1348855, -0.3075639, 0.3504279

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 245

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5334595, upper bound: 0.5366724
time: 1.58 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5334403, upper bound: 0.5334392
time: 1.75 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.08 seconds
NS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 4.08
Output dim: 0, lower bound: -0.5655256, upper bound: 0.5553728
NS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 4.08
Output dim: 0, lower bound: -0.5655157, upper bound: 0.5550132
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.08
Output dim: 0, lower bound: -0.5656714, upper bound: 0.5549811
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.08
Output dim: 0, lower bound: -0.5655157, upper bound: 0.5550132
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 4.08
Output dim: 0, lower bound: -0.5334189, upper bound: 0.5370422
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 4.08
Output dim: 0, lower bound: -0.5333943, upper bound: 0.5334403
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 4.08
Output dim: 0, lower bound: -0.5334595, upper bound: 0.5366724
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 4.08
Output dim: 0, lower bound: -0.5334403, upper bound: 0.5334392

## BFS NS instance: NS_A1_B1_B1

### Backsubstitution after applying NS history:
0: 0.6746894, 1.0091177, 0.7167201, 1.0067352, -0.3320458, 0.2923976
1: -0.0591800, 0.0639734, -0.0463979, 0.0461491, -0.1053291, 0.1103714
2: -0.0232218, 0.1118086, -0.0128021, 0.0926459, -0.1158677, 0.1246106
3: -0.0558084, 0.0872116, -0.0424021, 0.0732473, -0.1290557, 0.1296137
4: -0.0861012, 0.0444994, -0.0680437, 0.0284037, -0.1145049, 0.1125430
5: -0.0716894, 0.1795660, -0.0477926, 0.1608896, -0.2325790, 0.2273586
6: -0.0565628, 0.0844286, -0.0442496, 0.0696056, -0.1261685, 0.1286782
7: -0.0910453, 0.1089885, -0.0734423, 0.0820769, -0.1731222, 0.1824308
8: -0.0632034, 0.0992847, -0.0476022, 0.0823940, -0.1455974, 0.1468869
9: -0.0662497, 0.0968281, -0.0517316, 0.0674153, -0.1336650, 0.1485598

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5637773, upper bound: 0.5507492
time: 1.63 seconds

## Relational analysis of NS_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5500383, upper bound: 0.5506793
time: 1.77 seconds

## BFS NS instance: NS_A1_B1_B2

### Backsubstitution after applying NS history:
0: 0.6778433, 1.0090551, 0.6481054, 1.0113014, -0.3334581, 0.3609498
1: -0.0576316, 0.0618313, -0.0747998, 0.0842106, -0.1418422, 0.1366311
2: -0.0218932, 0.1098867, -0.0359052, 0.1357392, -0.1576324, 0.1457918
3: -0.0546856, 0.0852595, -0.0663267, 0.1058672, -0.1605528, 0.1515863
4: -0.0844423, 0.0424756, -0.1022819, 0.0636942, -0.1481366, 0.1447575
5: -0.0689919, 0.1778778, -0.0971737, 0.1952829, -0.2642748, 0.2750515
6: -0.0551660, 0.0829403, -0.0700125, 0.0975287, -0.1526947, 0.1529528
7: -0.0889138, 0.1062881, -0.1119681, 0.1323685, -0.2212822, 0.2182563
8: -0.0611643, 0.0978606, -0.0851876, 0.1125237, -0.1736881, 0.1830482
9: -0.0645238, 0.0939190, -0.0833151, 0.1246810, -0.1892048, 0.1772341

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5636310, upper bound: 0.5501322
time: 1.78 seconds

## Relational analysis of NS_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5500647, upper bound: 0.5500648
time: 2.89 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.7167201, 1.0067352, 0.5737605, 1.0451070, -0.3283869, 0.4329747
1: -0.0463979, 0.0461491, -0.1878469, 0.1780573, -0.2244552, 0.2339960
2: -0.0128021, 0.0926459, -0.1072832, 0.2750047, -0.2878068, 0.1999291
3: -0.0424021, 0.0732473, -0.1198316, 0.1912155, -0.2336176, 0.1930789
4: -0.0680437, 0.0284037, -0.2115348, 0.1537331, -0.2217767, 0.2399385
5: -0.0477926, 0.1608896, -0.2142647, 0.2806869, -0.3284795, 0.3751543
6: -0.0442496, 0.0696056, -0.1364323, 0.2052122, -0.2494618, 0.2060379
7: -0.0734423, 0.0820769, -0.2287775, 0.2076855, -0.2811278, 0.3108544
8: -0.0476022, 0.0823940, -0.2053939, 0.2141977, -0.2617998, 0.2877879
9: -0.0517316, 0.0674153, -0.1747209, 0.2630653, -0.3147970, 0.2421362

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 245

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A1_B2_A1_A1

### Relational analysis result of NS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5445990, upper bound: 0.5335029
time: 1.86 seconds

## Relational analysis of NS_A1_B2_A1_A2

### Relational analysis result of NS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5479874, upper bound: 0.5334931
time: 1.91 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.6481054, 1.0113014, 0.5775799, 1.0444301, -0.3963248, 0.4337215
1: -0.0747998, 0.0842106, -0.1854768, 0.1759178, -0.2507176, 0.2696875
2: -0.0359052, 0.1357392, -0.1057437, 0.2719904, -0.3078956, 0.2414829
3: -0.0663267, 0.1058672, -0.1184776, 0.1893367, -0.2556634, 0.2243448
4: -0.1022819, 0.0636942, -0.2091668, 0.1517586, -0.2540406, 0.2728610
5: -0.0971737, 0.1952829, -0.2114975, 0.2786761, -0.3758498, 0.4067804
6: -0.0700125, 0.0975287, -0.1350354, 0.2025111, -0.2725236, 0.2325641
7: -0.1119681, 0.1323685, -0.2263854, 0.2049762, -0.3169443, 0.3587539
8: -0.0851876, 0.1125237, -0.2028952, 0.2118476, -0.2970352, 0.3154189
9: -0.0833151, 0.1246810, -0.1726784, 0.2600786, -0.3433937, 0.2973594

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5443111, upper bound: 0.5335410
time: 1.68 seconds

## Relational analysis of NS_A1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5479874, upper bound: 0.5335379
time: 1.65 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: 0.5836221, 1.0433595, 0.8646507, 1.0046401, -0.4210180, 0.1787088
1: -0.1817275, 0.1725333, -0.0326158, 0.0074169, -0.1891444, 0.2051491
2: -0.1033083, 0.2672216, -0.0094829, 0.0492040, -0.1525123, 0.2767044
3: -0.1163354, 0.1863646, -0.0141850, 0.0470689, -0.1634043, 0.2005496
4: -0.2054204, 0.1486352, -0.0354038, 0.0102160, -0.2156364, 0.1840390
5: -0.2071195, 0.2754952, -0.0129675, 0.1182506, -0.3253700, 0.2884628
6: -0.1328256, 0.1982380, -0.0259889, 0.0249762, -0.1578018, 0.2242269
7: -0.2226010, 0.2006901, -0.0430410, 0.0208474, -0.2434484, 0.2437311
8: -0.1989420, 0.2081299, -0.0277846, 0.0426745, -0.2416166, 0.2359144
9: -0.1694473, 0.2553532, -0.0244922, 0.0332622, -0.2027095, 0.2798454

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5332409, upper bound: 0.5367603
time: 1.51 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5332409, upper bound: 0.5368123
time: 1.87 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: 0.6229223, 1.0363953, 0.8743206, 1.0051661, -0.3822438, 0.1620747
1: -0.1573407, 0.1505196, -0.0303707, 0.0043850, -0.1617257, 0.1808903
2: -0.0874675, 0.2362042, -0.0101775, 0.0426653, -0.1301329, 0.2463817
3: -0.1024024, 0.1670327, -0.0125586, 0.0426209, -0.1450232, 0.1795913
4: -0.1810533, 0.1283196, -0.0311137, 0.0078648, -0.1889181, 0.1594333
5: -0.1786443, 0.2548053, -0.0138412, 0.1111856, -0.2898299, 0.2686464
6: -0.1184525, 0.1704449, -0.0232394, 0.0210852, -0.1395377, 0.1936843
7: -0.1979870, 0.1728114, -0.0381420, 0.0167610, -0.2147480, 0.2109535
8: -0.1732299, 0.1839494, -0.0246886, 0.0385506, -0.2117805, 0.2086380
9: -0.1484311, 0.2246193, -0.0195093, 0.0310034, -0.1794345, 0.2441286

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5332171, upper bound: 0.5332006
time: 1.62 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5332171, upper bound: 0.5332614
time: 1.75 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: 0.5874733, 1.0426769, 0.8240218, 1.0048201, -0.4173468, 0.2186552
1: -0.1793378, 0.1703763, -0.0435468, 0.0297472, -0.2090850, 0.2139232
2: -0.1017561, 0.2641820, -0.0097750, 0.0780495, -0.1798057, 0.2739570
3: -0.1149701, 0.1844702, -0.0273816, 0.0652341, -0.1802042, 0.2118518
4: -0.2030327, 0.1466445, -0.0587616, 0.0223881, -0.2254208, 0.2054061
5: -0.2043292, 0.2734677, -0.0287694, 0.1471023, -0.3514315, 0.3022371
6: -0.1314172, 0.1955146, -0.0399969, 0.0421851, -0.1736022, 0.2355115
7: -0.2201891, 0.1979582, -0.0686231, 0.0375358, -0.2577249, 0.2665814
8: -0.1964224, 0.2057605, -0.0430873, 0.0656537, -0.2620761, 0.2488478
9: -0.1673880, 0.2523417, -0.0449966, 0.0557069, -0.2230949, 0.2973383

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A2_B2_B1_A1

### Relational analysis result of NS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5332805, upper bound: 0.5363793
time: 1.85 seconds

## Relational analysis of NS_A2_B2_B1_A2

### Relational analysis result of NS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5332805, upper bound: 0.5364255
time: 3.93 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: 0.6264586, 1.0357686, 0.8395842, 1.0052882, -0.3788297, 0.1961844
1: -0.1551462, 0.1485387, -0.0384021, 0.0193860, -0.1745322, 0.1869408
2: -0.0860421, 0.2334131, -0.0103387, 0.0661472, -0.1521893, 0.2437518
3: -0.1011486, 0.1652932, -0.0203753, 0.0585326, -0.1596813, 0.1856685
4: -0.1788608, 0.1264915, -0.0478977, 0.0164595, -0.1953203, 0.1743892
5: -0.1760820, 0.2529435, -0.0178746, 0.1364584, -0.3125404, 0.2708181
6: -0.1171591, 0.1679440, -0.0338291, 0.0351879, -0.1523470, 0.2017731
7: -0.1957721, 0.1703029, -0.0574022, 0.0313792, -0.2271513, 0.2277051
8: -0.1709162, 0.1817736, -0.0357636, 0.0554862, -0.2264025, 0.2175371
9: -0.1465400, 0.2218538, -0.0374321, 0.0416527, -0.1881927, 0.2592859

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A2_B2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5332625, upper bound: 0.5332006
time: 1.72 seconds

## Relational analysis of NS_A2_B2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5332625, upper bound: 0.5332614
time: 1.67 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.15 seconds
NS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -0.5637773, upper bound: 0.5507492
NS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -0.5500383, upper bound: 0.5506793
NS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -0.5636310, upper bound: 0.5501322
NS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -0.5500647, upper bound: 0.5500648
NS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -0.5445990, upper bound: 0.5335029
NS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -0.5479874, upper bound: 0.5334931
NS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -0.5443111, upper bound: 0.5335410
NS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -0.5479874, upper bound: 0.5335379
NS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -0.5332409, upper bound: 0.5367603
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -0.5332409, upper bound: 0.5368123
NS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -0.5332171, upper bound: 0.5332006
NS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -0.5332171, upper bound: 0.5332614
NS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -0.5332805, upper bound: 0.5363793
NS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -0.5332805, upper bound: 0.5364255
NS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -0.5332625, upper bound: 0.5332006
NS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -0.5332625, upper bound: 0.5332614

## BFS NS instance: NS_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: 0.7611310, 1.0050447, 0.7167201, 1.0067352, -0.2456042, 0.2883246
1: -0.0490225, 0.0444876, -0.0463979, 0.0461491, -0.0951715, 0.0908856
2: -0.0139519, 0.0938058, -0.0128021, 0.0926459, -0.1065978, 0.1066079
3: -0.0396139, 0.0736782, -0.0424021, 0.0732473, -0.1128612, 0.1160803
4: -0.0709643, 0.0302850, -0.0680437, 0.0284037, -0.0993680, 0.0983287
5: -0.0467331, 0.1609496, -0.0477926, 0.1608896, -0.2076227, 0.2087422
6: -0.0467248, 0.0596713, -0.0442496, 0.0696056, -0.1163304, 0.1039209
7: -0.0786732, 0.0630298, -0.0734423, 0.0820769, -0.1607500, 0.1364721
8: -0.0505896, 0.0811412, -0.0476022, 0.0823940, -0.1329836, 0.1287433
9: -0.0533829, 0.0726914, -0.0517316, 0.0674153, -0.1207982, 0.1244230

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_B1_A1_B1

### Relational analysis result of NS_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5500383, upper bound: 0.5506793
time: 1.64 seconds

## Relational analysis of NS_A1_B1_B1_A1_B2

### Relational analysis result of NS_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5500383, upper bound: 0.5506782
time: 1.67 seconds

## BFS NS instance: NS_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: 0.5971594, 1.0388936, 0.7799280, 1.0045586, -0.4073992, 0.2589656
1: -0.1573483, 0.1991967, -0.0412996, 0.0320383, -0.1893866, 0.2404963
2: -0.1059232, 0.2310522, -0.0092922, 0.0779000, -0.1838232, 0.2403444
3: -0.1166055, 0.2088947, -0.0302454, 0.0653563, -0.1819619, 0.2391401
4: -0.1892252, 0.1732421, -0.0566983, 0.0206598, -0.2098850, 0.2299404
5: -0.2288824, 0.2788445, -0.0309875, 0.1478216, -0.3767040, 0.3098319
6: -0.1465436, 0.1589732, -0.0379972, 0.0512577, -0.1978013, 0.1969704
7: -0.2329121, 0.2265615, -0.0640874, 0.0561015, -0.2890136, 0.2906489
8: -0.1949076, 0.1795728, -0.0406281, 0.0676625, -0.2625701, 0.2202009
9: -0.1718390, 0.2797996, -0.0439553, 0.0515768, -0.2234157, 0.3237549

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 93

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A1_B1_B1_A2_A1

### Relational analysis result of NS_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5317786, upper bound: 0.5290795
time: 1.69 seconds

## Relational analysis of NS_A1_B1_B1_A2_A2

### Relational analysis result of NS_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5289799, upper bound: 0.5290601
time: 1.57 seconds

## BFS NS instance: NS_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: 0.7634771, 1.0050331, 0.6481054, 1.0113014, -0.2478243, 0.3569278
1: -0.0482551, 0.0431929, -0.0747998, 0.0842106, -0.1324657, 0.1179927
2: -0.0134475, 0.0921844, -0.0359052, 0.1357392, -0.1491867, 0.1280896
3: -0.0386268, 0.0728286, -0.0663267, 0.1058672, -0.1444940, 0.1391553
4: -0.0695279, 0.0293175, -0.1022819, 0.0636942, -0.1332221, 0.1315995
5: -0.0451079, 0.1596043, -0.0971737, 0.1952829, -0.2403908, 0.2567781
6: -0.0458526, 0.0587223, -0.0700125, 0.0975287, -0.1433813, 0.1287348
7: -0.0772264, 0.0621416, -0.1119681, 0.1323685, -0.2095948, 0.1741097
8: -0.0495957, 0.0797459, -0.0851876, 0.1125237, -0.1621195, 0.1649335
9: -0.0524307, 0.0705753, -0.0833151, 0.1246810, -0.1771117, 0.1538904

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_B2_A1_B1

### Relational analysis result of NS_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5500648, upper bound: 0.5500636
time: 2.40 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2

### Relational analysis result of NS_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5500648, upper bound: 0.5500647
time: 1.72 seconds

## BFS NS instance: NS_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: 0.5997576, 1.0384640, 0.7204868, 1.0081012, -0.4083437, 0.3179771
1: -0.1559889, 0.1972319, -0.0622184, 0.0649214, -0.2209104, 0.2594503
2: -0.1047590, 0.2293372, -0.0252234, 0.1134129, -0.2181718, 0.2545606
3: -0.1156103, 0.2071825, -0.0524300, 0.0901507, -0.2057609, 0.2596125
4: -0.1877715, 0.1714288, -0.0885356, 0.0470654, -0.2348370, 0.2599643
5: -0.2265297, 0.2773669, -0.0713071, 0.1792795, -0.4058092, 0.3486740
6: -0.1453156, 0.1576020, -0.0598799, 0.0747656, -0.2200812, 0.2174819
7: -0.2310322, 0.2241908, -0.0979270, 0.0893457, -0.3203779, 0.3221178
8: -0.1931145, 0.1783184, -0.0681796, 0.0973127, -0.2904271, 0.2464980
9: -0.1703293, 0.2772464, -0.0681340, 0.1018943, -0.2722237, 0.3453804

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_B2_A2_A1

### Relational analysis result of NS_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5497842, upper bound: 0.5486508
time: 1.82 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2

### Relational analysis result of NS_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5487180, upper bound: 0.5487169
time: 1.90 seconds

## BFS NS instance: NS_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: 0.8701054, 1.0045376, 0.5836221, 1.0433595, -0.1732541, 0.4209155
1: -0.0313567, 0.0057155, -0.1817275, 0.1725333, -0.2038900, 0.1874430
2: -0.0093475, 0.0455170, -0.1033083, 0.2672216, -0.2765690, 0.1488253
3: -0.0132594, 0.0445744, -0.1163354, 0.1863646, -0.1996240, 0.1609097
4: -0.0329978, 0.0088574, -0.2054204, 0.1486352, -0.1816331, 0.2142778
5: -0.0127973, 0.1142885, -0.2071195, 0.2754952, -0.2882925, 0.3214080
6: -0.0244469, 0.0227540, -0.1328256, 0.1982380, -0.2226849, 0.1555796
7: -0.0402653, 0.0185557, -0.2226010, 0.2006901, -0.2409554, 0.2411567
8: -0.0260483, 0.0403385, -0.1989420, 0.2081299, -0.2341781, 0.2392805
9: -0.0216764, 0.0319954, -0.1694473, 0.2553532, -0.2770296, 0.2014427

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A1_B2_A1_A1_B1

### Relational analysis result of NS_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5440922, upper bound: 0.5333300
time: 1.78 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2

### Relational analysis result of NS_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5442616, upper bound: 0.5333300
time: 1.82 seconds

## BFS NS instance: NS_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: 0.8773322, 1.0050526, 0.6229223, 1.0363953, -0.1590631, 0.3821303
1: -0.0295352, 0.0032750, -0.1573407, 0.1505196, -0.1800548, 0.1606157
2: -0.0100276, 0.0406022, -0.0874675, 0.2362042, -0.2462318, 0.1280698
3: -0.0122027, 0.0409655, -0.1024024, 0.1670327, -0.1792354, 0.1433679
4: -0.0295172, 0.0077309, -0.1810533, 0.1283196, -0.1578368, 0.1887842
5: -0.0136526, 0.1085565, -0.1786443, 0.2548053, -0.2684579, 0.2872008
6: -0.0222162, 0.0203790, -0.1184525, 0.1704449, -0.1926611, 0.1388314
7: -0.0368426, 0.0152402, -0.1979870, 0.1728114, -0.2096540, 0.2132272
8: -0.0235365, 0.0374476, -0.1732299, 0.1839494, -0.2074858, 0.2106775
9: -0.0180498, 0.0301627, -0.1484311, 0.2246193, -0.2426691, 0.1785939

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A1_B2_A1_A2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5475985, upper bound: 0.5333187
time: 1.85 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5477609, upper bound: 0.5333198
time: 1.69 seconds

## BFS NS instance: NS_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: 0.8299456, 1.0047139, 0.5874733, 1.0426769, -0.2127313, 0.4172406
1: -0.0413114, 0.0259852, -0.1793378, 0.1703763, -0.2116877, 0.2053230
2: -0.0095803, 0.0733582, -0.1017561, 0.2641820, -0.2737623, 0.1751143
3: -0.0245481, 0.0628008, -0.1149701, 0.1844702, -0.2090182, 0.1777710
4: -0.0546108, 0.0195503, -0.2030327, 0.1466445, -0.2012553, 0.2225830
5: -0.0241464, 0.1432376, -0.2043292, 0.2734677, -0.2976142, 0.3475668
6: -0.0374648, 0.0395560, -0.1314172, 0.1955146, -0.2329794, 0.1709731
7: -0.0644040, 0.0353004, -0.2201891, 0.1979582, -0.2623622, 0.2554894
8: -0.0401990, 0.0616628, -0.1964224, 0.2057605, -0.2459596, 0.2580853
9: -0.0422500, 0.0495758, -0.1673880, 0.2523417, -0.2945917, 0.2169637

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A1_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5438352, upper bound: 0.5333702
time: 2.10 seconds

## Relational analysis of NS_A1_B2_A2_A1_B2

### Relational analysis result of NS_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5440398, upper bound: 0.5333691
time: 1.50 seconds

## BFS NS instance: NS_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: 0.8433097, 1.0051452, 0.6264586, 1.0357686, -0.1924589, 0.3786866
1: -0.0375421, 0.0167516, -0.1551462, 0.1485387, -0.1860808, 0.1718978
2: -0.0101499, 0.0636290, -0.0860421, 0.2334131, -0.2435630, 0.1496711
3: -0.0190561, 0.0568288, -0.1011486, 0.1652932, -0.1843493, 0.1579775
4: -0.0457446, 0.0155316, -0.1788608, 0.1264915, -0.1722361, 0.1943923
5: -0.0157932, 0.1337522, -0.1760820, 0.2529435, -0.2687367, 0.3098342
6: -0.0325084, 0.0336702, -0.1171591, 0.1679440, -0.2004524, 0.1508293
7: -0.0549368, 0.0298139, -0.1957721, 0.1703029, -0.2252397, 0.2255860
8: -0.0345777, 0.0531542, -0.1709162, 0.1817736, -0.2163512, 0.2240704
9: -0.0355089, 0.0398762, -0.1465400, 0.2218538, -0.2573627, 0.1864162

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5475985, upper bound: 0.5333651
time: 1.75 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2

### Relational analysis result of NS_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5477609, upper bound: 0.5333651
time: 1.71 seconds

## BFS NS instance: NS_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: 0.6426894, 1.0328927, 0.8700899, 1.0043930, -0.3617036, 0.1628028
1: -0.1450746, 0.1394472, -0.0313603, 0.0057204, -0.1507950, 0.1708076
2: -0.0794999, 0.2206031, -0.0091566, 0.0455276, -0.1250274, 0.2297597
3: -0.0953944, 0.1573093, -0.0132621, 0.0445815, -0.1399759, 0.1705714
4: -0.1687972, 0.1181012, -0.0330047, 0.0088613, -0.1776585, 0.1511059
5: -0.1643219, 0.2443987, -0.0125571, 0.1142998, -0.2786217, 0.2569559
6: -0.1112231, 0.1564657, -0.0244513, 0.0227604, -0.1339835, 0.1809170
7: -0.1856067, 0.1587892, -0.0402733, 0.0185622, -0.2041690, 0.1990624
8: -0.1602973, 0.1717871, -0.0260533, 0.0403452, -0.2006424, 0.1978404
9: -0.1378604, 0.2091607, -0.0216845, 0.0319990, -0.1698595, 0.2308452

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A2_B1_B1_A1_B1

### Relational analysis result of NS_A2_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5030596, upper bound: 0.5192727
time: 1.52 seconds

## Relational analysis of NS_A2_B1_B1_A1_B2

### Relational analysis result of NS_A2_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5029390, upper bound: 0.5080555
time: 1.58 seconds

## BFS NS instance: NS_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: 0.6080372, 1.0390331, 0.8676173, 1.0045234, -0.3964862, 0.1714157
1: -0.1665774, 0.1588573, -0.0319311, 0.0064916, -0.1730690, 0.1907884
2: -0.0934673, 0.2479521, -0.0093289, 0.0471988, -0.1406660, 0.2572809
3: -0.1076796, 0.1743547, -0.0136816, 0.0457123, -0.1533919, 0.1880363
4: -0.1902825, 0.1360143, -0.0340953, 0.0094771, -0.1997596, 0.1701095
5: -0.1894295, 0.2626417, -0.0127738, 0.1160958, -0.3055252, 0.2754155
6: -0.1238963, 0.1809718, -0.0251503, 0.0237677, -0.1476640, 0.2061221
7: -0.2073096, 0.1833707, -0.0415314, 0.0196010, -0.2269106, 0.2249021
8: -0.1829684, 0.1931080, -0.0268403, 0.0414041, -0.2243725, 0.2199483
9: -0.1563912, 0.2362599, -0.0229608, 0.0325732, -0.1889644, 0.2592208

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A2_B1_B1_A2_A1

### Relational analysis result of NS_A2_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5195513, upper bound: 0.5092310
time: 1.55 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2

### Relational analysis result of NS_A2_B1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5029390, upper bound: 0.5081358
time: 1.34 seconds

## BFS NS instance: NS_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: 0.6809140, 1.0261191, 0.8791367, 1.0049136, -0.3239995, 0.1469824
1: -0.1213551, 0.1180361, -0.0290346, 0.0027014, -0.1240565, 0.1470706
2: -0.0640926, 0.1904346, -0.0098441, 0.0393660, -0.1034586, 0.2002787
3: -0.0818427, 0.1385065, -0.0119894, 0.0399744, -0.1218172, 0.1504959
4: -0.1450970, 0.0983416, -0.0285606, 0.0076525, -0.1527495, 0.1269021
5: -0.1366262, 0.2242749, -0.0134218, 0.1069812, -0.2436074, 0.2376968
6: -0.0972432, 0.1294332, -0.0216354, 0.0199558, -0.1171990, 0.1510686
7: -0.1616664, 0.1316736, -0.0361059, 0.0143290, -0.1759953, 0.1677796
8: -0.1352888, 0.1482683, -0.0228848, 0.0367868, -0.1720756, 0.1711532
9: -0.1174194, 0.1792679, -0.0172179, 0.0296591, -0.1470785, 0.1964858

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A2_B1_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5029848, upper bound: 0.5186512
time: 1.54 seconds

## Relational analysis of NS_A2_B1_B2_A1_B2

### Relational analysis result of NS_A2_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5028530, upper bound: 0.5033955
time: 1.15 seconds

## BFS NS instance: NS_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: 0.6472589, 1.0320829, 0.8769382, 1.0050516, -0.3577927, 0.1551448
1: -0.1422392, 0.1368878, -0.0296445, 0.0034203, -0.1456594, 0.1665323
2: -0.0776582, 0.2169967, -0.0100263, 0.0408721, -0.1185303, 0.2270229
3: -0.0937744, 0.1550616, -0.0122493, 0.0411822, -0.1349566, 0.1673108
4: -0.1659643, 0.1157391, -0.0297261, 0.0077484, -0.1737127, 0.1454652
5: -0.1610111, 0.2419932, -0.0136510, 0.1089006, -0.2699117, 0.2556442
6: -0.1095519, 0.1532343, -0.0223501, 0.0204714, -0.1300233, 0.1755844
7: -0.1827449, 0.1555478, -0.0370127, 0.0154392, -0.1981842, 0.1925605
8: -0.1573077, 0.1689758, -0.0236872, 0.0375920, -0.1948997, 0.1926630
9: -0.1354169, 0.2055874, -0.0182408, 0.0302728, -0.1656897, 0.2238282

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A2_B1_B2_A2_A1

### Relational analysis result of NS_A2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5187747, upper bound: 0.5035744
time: 1.59 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2

### Relational analysis result of NS_A2_B1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5028530, upper bound: 0.5034608
time: 1.50 seconds

## BFS NS instance: NS_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: 0.6459141, 1.0323212, 0.8302952, 1.0045614, -0.3586473, 0.2020260
1: -0.1430736, 0.1376408, -0.0411794, 0.0257630, -0.1688366, 0.1788202
2: -0.0782001, 0.2180582, -0.0093788, 0.0730812, -0.1512813, 0.2274370
3: -0.0942512, 0.1557229, -0.0243808, 0.0626572, -0.1569084, 0.1801037
4: -0.1667979, 0.1164343, -0.0543657, 0.0193828, -0.1861807, 0.1708000
5: -0.1619854, 0.2427009, -0.0238735, 0.1430095, -0.3049949, 0.2665744
6: -0.1100436, 0.1541851, -0.0373153, 0.0394008, -0.1494444, 0.1915004
7: -0.1835870, 0.1565016, -0.0641549, 0.0351684, -0.2187554, 0.2206565
8: -0.1581875, 0.1698031, -0.0400286, 0.0614272, -0.2196147, 0.2098317
9: -0.1361360, 0.2066389, -0.0420879, 0.0492138, -0.1853498, 0.2487268

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A2_B2_B1_A1_B1

### Relational analysis result of NS_A2_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5031026, upper bound: 0.5187707
time: 1.88 seconds

## Relational analysis of NS_A2_B2_B1_A1_B2

### Relational analysis result of NS_A2_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5029826, upper bound: 0.5067585
time: 1.47 seconds

## BFS NS instance: NS_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: 0.6118456, 1.0383583, 0.8274871, 1.0047071, -0.3928615, 0.2108712
1: -0.1642140, 0.1567242, -0.0422392, 0.0275464, -0.1917605, 0.1989633
2: -0.0919322, 0.2449462, -0.0095713, 0.0753053, -0.1672375, 0.2545175
3: -0.1063294, 0.1724814, -0.0257241, 0.0638107, -0.1701401, 0.1982054
4: -0.1879212, 0.1340455, -0.0563335, 0.0207281, -0.2086493, 0.1903790
5: -0.1866700, 0.2606366, -0.0260651, 0.1448416, -0.3315116, 0.2867017
6: -0.1225035, 0.1782784, -0.0385156, 0.0406472, -0.1631507, 0.2167940
7: -0.2049245, 0.1806690, -0.0661550, 0.0362282, -0.2411527, 0.2468240
8: -0.1804768, 0.1907647, -0.0413978, 0.0633191, -0.2437959, 0.2321625
9: -0.1543545, 0.2332817, -0.0433899, 0.0521204, -0.2064748, 0.2766716

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A2_B2_B1_A2_A1

### Relational analysis result of NS_A2_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5193515, upper bound: 0.5076082
time: 1.73 seconds

## Relational analysis of NS_A2_B2_B1_A2_A2

### Relational analysis result of NS_A2_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5029826, upper bound: 0.5068534
time: 1.52 seconds

## BFS NS instance: NS_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: 0.6838112, 1.0256059, 0.8454104, 1.0050251, -0.3212140, 0.1801955
1: -0.1195573, 0.1164131, -0.0370572, 0.0152664, -0.1348237, 0.1534704
2: -0.0629248, 0.1881480, -0.0099912, 0.0622091, -0.1251339, 0.1981392
3: -0.0808155, 0.1370813, -0.0183123, 0.0558681, -0.1366836, 0.1553936
4: -0.1433007, 0.0968438, -0.0445305, 0.0150083, -0.1583090, 0.1413743
5: -0.1345268, 0.2227496, -0.0146195, 0.1322263, -0.2667531, 0.2373692
6: -0.0961836, 0.1273843, -0.0317637, 0.0328144, -0.1289980, 0.1591480
7: -0.1598517, 0.1296182, -0.0535467, 0.0289312, -0.1887830, 0.1831649
8: -0.1333932, 0.1464858, -0.0339090, 0.0518392, -0.1852324, 0.1803948
9: -0.1158700, 0.1770019, -0.0344245, 0.0388745, -0.1547446, 0.2114264

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 93

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A2_B2_B2_A1_B1

### Relational analysis result of NS_A2_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5054308, upper bound: 0.5184584
time: 1.49 seconds

## Relational analysis of NS_A2_B2_B2_A1_B2

### Relational analysis result of NS_A2_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5052892, upper bound: 0.5050141
time: 1.43 seconds

## BFS NS instance: NS_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: 0.6507534, 1.0314637, 0.8428187, 1.0051773, -0.3544239, 0.1886450
1: -0.1400707, 0.1349303, -0.0376554, 0.0170988, -0.1571695, 0.1725858
2: -0.0762496, 0.2142388, -0.0101922, 0.0639608, -0.1402104, 0.2244310
3: -0.0925355, 0.1533426, -0.0192300, 0.0570533, -0.1495888, 0.1725726
4: -0.1637974, 0.1139327, -0.0460283, 0.0156539, -0.1794513, 0.1599610
5: -0.1584792, 0.2401533, -0.0160675, 0.1341088, -0.2925880, 0.2562207
6: -0.1082738, 0.1507629, -0.0326824, 0.0338702, -0.1421441, 0.1834453
7: -0.1805563, 0.1530689, -0.0552617, 0.0300201, -0.2105763, 0.2083306
8: -0.1550215, 0.1668256, -0.0347339, 0.0534615, -0.2084830, 0.2015595
9: -0.1335482, 0.2028544, -0.0357623, 0.0401103, -0.1736585, 0.2386167

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A2_B2_B2_A2_B1

### Relational analysis result of NS_A2_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5051993, upper bound: 0.5184764
time: 1.55 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2

### Relational analysis result of NS_A2_B2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5050582, upper bound: 0.5050582
time: 1.55 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.89 seconds
NS_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.5500383, upper bound: 0.5506793
NS_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.5500383, upper bound: 0.5506782
NS_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.5317786, upper bound: 0.5290795
NS_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.5289799, upper bound: 0.5290601
NS_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.5500648, upper bound: 0.5500636
NS_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.5500648, upper bound: 0.5500647
NS_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.5497842, upper bound: 0.5486508
NS_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.5487180, upper bound: 0.5487169
NS_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.5440922, upper bound: 0.5333300
NS_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.5442616, upper bound: 0.5333300
NS_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.5475985, upper bound: 0.5333187
NS_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.5477609, upper bound: 0.5333198
NS_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.5438352, upper bound: 0.5333702
NS_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.5440398, upper bound: 0.5333691
NS_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.5475985, upper bound: 0.5333651
NS_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.5477609, upper bound: 0.5333651
NS_A2_B1_B1_A1_B1, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.5030596, upper bound: 0.5192727
NS_A2_B1_B1_A1_B2, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.5029390, upper bound: 0.5080555
NS_A2_B1_B1_A2_A1, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.5195513, upper bound: 0.5092310
NS_A2_B1_B1_A2_A2, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.5029390, upper bound: 0.5081358
NS_A2_B1_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.5029848, upper bound: 0.5186512
NS_A2_B1_B2_A1_B2, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.5028530, upper bound: 0.5033955
NS_A2_B1_B2_A2_A1, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.5187747, upper bound: 0.5035744
NS_A2_B1_B2_A2_A2, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.5028530, upper bound: 0.5034608
NS_A2_B2_B1_A1_B1, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.5031026, upper bound: 0.5187707
NS_A2_B2_B1_A1_B2, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.5029826, upper bound: 0.5067585
NS_A2_B2_B1_A2_A1, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.5193515, upper bound: 0.5076082
NS_A2_B2_B1_A2_A2, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.5029826, upper bound: 0.5068534
NS_A2_B2_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.5054308, upper bound: 0.5184584
NS_A2_B2_B2_A1_B2, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.5052892, upper bound: 0.5050141
NS_A2_B2_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.5051993, upper bound: 0.5184764
NS_A2_B2_B2_A2_B2, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 0, lower bound: -0.5050582, upper bound: 0.5050582

## BFS NS instance: NS_A1_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.7611310, 1.0050447, 0.7983336, 1.0043157, -0.2431847, 0.2067111
1: -0.0490225, 0.0444876, -0.0405003, 0.0287419, -0.0777643, 0.0849879
2: -0.0139519, 0.0938058, -0.0090006, 0.0748188, -0.0887707, 0.1028064
3: -0.0396139, 0.0736782, -0.0273666, 0.0637051, -0.1033190, 0.1010448
4: -0.0709643, 0.0302850, -0.0545702, 0.0192968, -0.0902612, 0.0848552
5: -0.0467331, 0.1609496, -0.0272462, 0.1450194, -0.1917525, 0.1881958
6: -0.0467248, 0.0596713, -0.0369248, 0.0464897, -0.0932144, 0.0965961
7: -0.0786732, 0.0630298, -0.0626681, 0.0485557, -0.1272289, 0.1256979
8: -0.0505896, 0.0811412, -0.0394594, 0.0643672, -0.1149568, 0.1206005
9: -0.0533829, 0.0726914, -0.0424477, 0.0487658, -0.1021488, 0.1151390

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A1_B1_B1_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5506999, upper bound: 0.5339829
time: 1.78 seconds

## Relational analysis of NS_A1_B1_B1_A1_B1_B2

### Relational analysis result of NS_A1_B1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5542444, upper bound: 0.5399651
time: 1.69 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.7611310, 1.0050447, 0.6878760, 1.0323187, -0.2711877, 0.3171687
1: -0.0490225, 0.0444876, -0.0949228, 0.1162096, -0.1652320, 0.1394105
2: -0.0139519, 0.0938058, -0.0433007, 0.1858696, -0.1998215, 0.1371065
3: -0.0396139, 0.0736782, -0.0925049, 0.1211566, -0.1607705, 0.1661831
4: -0.0709643, 0.0302850, -0.1541585, 0.0876065, -0.1585708, 0.1844435
5: -0.0467331, 0.1609496, -0.1350161, 0.2359241, -0.2826572, 0.2959658
6: -0.0467248, 0.0596713, -0.0981902, 0.1026080, -0.1493327, 0.1578615
7: -0.0786732, 0.0630298, -0.1655904, 0.0889120, -0.1675852, 0.2286202
8: -0.0505896, 0.0811412, -0.1094653, 0.1573737, -0.2079633, 0.1906064
9: -0.0533829, 0.0726914, -0.1081205, 0.1966150, -0.2499979, 0.1808118

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 93

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A1_B1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5467199, upper bound: 0.5332929
time: 1.97 seconds

## Relational analysis of NS_A1_B1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5466415, upper bound: 0.5291475
time: 1.70 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: 0.6796408, 1.0252547, 0.8074253, 1.0044392, -0.3247984, 0.2178294
1: -0.1141921, 0.1368213, -0.0395570, 0.0259400, -0.1401322, 0.1763783
2: -0.0689618, 0.1766047, -0.0091785, 0.0719572, -0.1409190, 0.1857833
3: -0.0850131, 0.1545413, -0.0252921, 0.0620541, -0.1470672, 0.1798334
4: -0.1430787, 0.1156779, -0.0520088, 0.0182762, -0.1613549, 0.1676867
5: -0.1541908, 0.2319359, -0.0244923, 0.1423328, -0.2965236, 0.2564282
6: -0.1075609, 0.1154401, -0.0357414, 0.0436228, -0.1511837, 0.1511815
7: -0.1732325, 0.1512975, -0.0603685, 0.0448162, -0.2180487, 0.2116660
8: -0.1379820, 0.1397513, -0.0377960, 0.0617466, -0.1997286, 0.1775474
9: -0.1239123, 0.1987516, -0.0407132, 0.0457042, -0.1696166, 0.2394648

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 93

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_B1_A2_A1_A1

### Relational analysis result of NS_A1_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5313355, upper bound: 0.5282933
time: 1.59 seconds

## Relational analysis of NS_A1_B1_B1_A2_A1_A2

### Relational analysis result of NS_A1_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5303027, upper bound: 0.5283570
time: 1.71 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: 0.6843632, 1.0244740, 0.8544952, 1.0042895, -0.3199263, 0.1699789
1: -0.1117213, 0.1332502, -0.0349601, 0.0105845, -0.1223057, 0.1682104
2: -0.0668457, 0.1734875, -0.0090199, 0.0560685, -0.1229142, 0.1825074
3: -0.0832043, 0.1514294, -0.0159083, 0.0517134, -0.1349177, 0.1673377
4: -0.1404366, 0.1123821, -0.0398832, 0.0127456, -0.1531822, 0.1522653
5: -0.1499145, 0.2292501, -0.0123853, 0.1256274, -0.2755419, 0.2416354
6: -0.1053290, 0.1129477, -0.0288597, 0.0291134, -0.1344425, 0.1418074
7: -0.1698157, 0.1469884, -0.0482087, 0.0251144, -0.1949301, 0.1951971
8: -0.1347228, 0.1374715, -0.0310172, 0.0470239, -0.1817467, 0.1684888
9: -0.1211684, 0.1941112, -0.0297348, 0.0356207, -0.1567890, 0.2238460

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_B1_A2_A2_A1

### Relational analysis result of NS_A1_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5287414, upper bound: 0.5282835
time: 1.77 seconds

## Relational analysis of NS_A1_B1_B1_A2_A2_A2

### Relational analysis result of NS_A1_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5282396, upper bound: 0.5283386
time: 1.60 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.7634771, 1.0050331, 0.7399384, 1.0076054, -0.2441283, 0.2650947
1: -0.0482551, 0.0431929, -0.0600745, 0.0609631, -0.1092181, 0.1032674
2: -0.0134475, 0.0921844, -0.0231938, 0.1101008, -0.1235483, 0.1153782
3: -0.0386268, 0.0728286, -0.0494173, 0.0871750, -0.1258018, 0.1222459
4: -0.0695279, 0.0293175, -0.0859832, 0.0440337, -0.1135616, 0.1153007
5: -0.0451079, 0.1596043, -0.0658135, 0.1759739, -0.2210818, 0.2254179
6: -0.0458526, 0.0587223, -0.0581260, 0.0694934, -0.1153460, 0.1168483
7: -0.0772264, 0.0621416, -0.0958181, 0.0781591, -0.1553854, 0.1579597
8: -0.0495957, 0.0797459, -0.0656617, 0.0939140, -0.1435097, 0.1454075
9: -0.0524307, 0.0705753, -0.0652722, 0.0976700, -0.1501007, 0.1358475

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A1_B1_B2_A1_B1_B1

### Relational analysis result of NS_A1_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5504173, upper bound: 0.5333815
time: 1.65 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5540427, upper bound: 0.5391467
time: 1.91 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.7634771, 1.0050331, 0.5552647, 1.0411532, -0.2776761, 0.4497684
1: -0.0482551, 0.0431929, -0.1726215, 0.2095309, -0.2577860, 0.2158144
2: -0.0134475, 0.0921844, -0.1120469, 0.3017985, -0.3152460, 0.2042313
3: -0.0386268, 0.0728286, -0.1281740, 0.2178999, -0.2565267, 0.2010026
4: -0.0695279, 0.0293175, -0.2005732, 0.1827793, -0.2523072, 0.2298907
5: -0.0451079, 0.1596043, -0.2424278, 0.2866163, -0.3317242, 0.4020321
6: -0.0458526, 0.0587223, -0.1530022, 0.1732270, -0.2190796, 0.2117245
7: -0.0772264, 0.0621416, -0.2450728, 0.2390312, -0.3162575, 0.3072144
8: -0.0495957, 0.0797459, -0.2290019, 0.1861704, -0.2357661, 0.3087478
9: -0.0524307, 0.0705753, -0.1867765, 0.2932273, -0.3456580, 0.2573518

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 93

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A1_B1_B2_A1_B2_B1

### Relational analysis result of NS_A1_B1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5464146, upper bound: 0.5323510
time: 2.04 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2_B2

### Relational analysis result of NS_A1_B1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5464019, upper bound: 0.5291066
time: 2.52 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: 0.6442509, 1.0311066, 0.7477425, 1.0065479, -0.3622971, 0.2833641
1: -0.1327090, 0.1635846, -0.0568351, 0.0560646, -0.1887735, 0.2204197
2: -0.0848207, 0.1999663, -0.0203570, 0.1057998, -0.1906206, 0.2203233
3: -0.0985683, 0.1778624, -0.0468615, 0.0828744, -0.1814428, 0.2247240
4: -0.1628785, 0.1403766, -0.0823310, 0.0397989, -0.2026775, 0.2227076
5: -0.1862383, 0.2520628, -0.0600198, 0.1722096, -0.3584478, 0.3120826
6: -0.1242870, 0.1341186, -0.0550755, 0.0659224, -0.1902094, 0.1891942
7: -0.1988390, 0.1835907, -0.0911959, 0.0721606, -0.2709996, 0.2747867
8: -0.1624068, 0.1568373, -0.0612111, 0.0907205, -0.2531273, 0.2180484
9: -0.1444760, 0.2335264, -0.0617158, 0.0912775, -0.2357535, 0.2952422

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 93

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A1_B1_B2_A2_A1_A1

### Relational analysis result of NS_A1_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5314445, upper bound: 0.5282316
time: 1.88 seconds

## Relational analysis of NS_A1_B1_B2_A2_A1_A2

### Relational analysis result of NS_A1_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5287801, upper bound: 0.5282245
time: 1.64 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: 0.6195401, 1.0351928, 0.7401538, 1.0072223, -0.3876822, 0.2950390
1: -0.1456383, 0.1822718, -0.0589376, 0.0594705, -0.2051088, 0.2412094
2: -0.0958941, 0.2162785, -0.0222455, 0.1087631, -0.2046572, 0.2385239
3: -0.1080332, 0.1941463, -0.0487834, 0.0857776, -0.1938108, 0.2429297
4: -0.1767038, 0.1576225, -0.0848012, 0.0425553, -0.2192591, 0.2424237
5: -0.2086155, 0.2661162, -0.0640751, 0.1748716, -0.3834871, 0.3301913
6: -0.1359659, 0.1471609, -0.0570751, 0.0687709, -0.2047369, 0.2042360
7: -0.2167187, 0.2061394, -0.0941336, 0.0771500, -0.2938686, 0.3002729
8: -0.1794614, 0.1687676, -0.0641214, 0.0930550, -0.2725164, 0.2328890
9: -0.1588346, 0.2578078, -0.0640727, 0.0955597, -0.2543943, 0.3218805

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 93

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A1_B1_B2_A2_A2_A1

### Relational analysis result of NS_A1_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5304170, upper bound: 0.5282931
time: 7.48 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2_A2

### Relational analysis result of NS_A1_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5282776, upper bound: 0.5282765
time: 1.72 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: 0.8753143, 1.0042808, 0.6426894, 1.0328927, -0.1575784, 0.3615914
1: -0.0300950, 0.0040187, -0.1450746, 0.1394472, -0.1695423, 0.1490933
2: -0.0090083, 0.0419846, -0.0794999, 0.2206031, -0.2296115, 0.1214845
3: -0.0124411, 0.0420747, -0.0953944, 0.1573093, -0.1697505, 0.1374691
4: -0.0305869, 0.0078206, -0.1687972, 0.1181012, -0.1486881, 0.1766178
5: -0.0123707, 0.1103182, -0.1643219, 0.2443987, -0.2567694, 0.2746400
6: -0.0229018, 0.0208522, -0.1112231, 0.1564657, -0.1793675, 0.1320752
7: -0.0377133, 0.0162592, -0.1856067, 0.1587892, -0.1965025, 0.2018659
8: -0.0243084, 0.0381867, -0.1602973, 0.1717871, -0.1960955, 0.1984840
9: -0.0190277, 0.0307260, -0.1378604, 0.2091607, -0.2281884, 0.1685865

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A1_B2_A1_A1_B1_B1

### Relational analysis result of NS_A1_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5220156, upper bound: 0.5239308
time: 1.59 seconds

## Relational analysis of NS_A1_B2_A1_A1_B1_B2

### Relational analysis result of NS_A1_B2_A1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5180828, upper bound: 0.5030726
time: 1.44 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: 0.8731177, 1.0044266, 0.6080372, 1.0390331, -0.1659153, 0.3963894
1: -0.0306614, 0.0047760, -0.1665774, 0.1588573, -0.1895187, 0.1713534
2: -0.0092010, 0.0434808, -0.0934673, 0.2479521, -0.2571531, 0.1369481
3: -0.0127483, 0.0431967, -0.1076796, 0.1743547, -0.1871029, 0.1508763
4: -0.0316691, 0.0081071, -0.1902825, 0.1360143, -0.1676834, 0.1983896
5: -0.0126130, 0.1121003, -0.1894295, 0.2626417, -0.2752548, 0.3015298
6: -0.0235953, 0.0215268, -0.1238963, 0.1809718, -0.2045672, 0.1454232
7: -0.0387324, 0.0172900, -0.2073096, 0.1833707, -0.2221031, 0.2245996
8: -0.0250894, 0.0390484, -0.1829684, 0.1931080, -0.2181974, 0.2220168
9: -0.0201214, 0.0312958, -0.1563912, 0.2362599, -0.2563813, 0.1876870

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A1_B2_A1_A1_B2_B1

### Relational analysis result of NS_A1_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5222578, upper bound: 0.5239297
time: 2.15 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2_B2

### Relational analysis result of NS_A1_B2_A1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5183926, upper bound: 0.5030737
time: 1.87 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: 0.8821917, 1.0047950, 0.6809140, 1.0261191, -0.1439274, 0.3238809
1: -0.0281871, 0.0019954, -0.1213551, 0.1180361, -0.1462232, 0.1233505
2: -0.0096875, 0.0372733, -0.0640926, 0.1904346, -0.2001221, 0.1013659
3: -0.0116284, 0.0382987, -0.0818427, 0.1385065, -0.1501349, 0.1201414
4: -0.0269412, 0.0075249, -0.1450970, 0.0983416, -0.1252827, 0.1526219
5: -0.0132249, 0.1043144, -0.1366262, 0.2242749, -0.2374998, 0.2409405
6: -0.0207458, 0.0192395, -0.0972432, 0.1294332, -0.1501790, 0.1164827
7: -0.0349802, 0.0127864, -0.1616664, 0.1316736, -0.1666538, 0.1744528
8: -0.0218940, 0.0356680, -0.1352888, 0.1482683, -0.1701624, 0.1709569
9: -0.0159329, 0.0288065, -0.1174194, 0.1792679, -0.1952008, 0.1462259

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B2_A1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5268245, upper bound: 0.5030988
time: 1.70 seconds

## Relational analysis of NS_A1_B2_A1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203080, upper bound: 0.5030436
time: 1.66 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: 0.8799546, 1.0049415, 0.6472589, 1.0320829, -0.1521283, 0.3576826
1: -0.0288077, 0.0025124, -0.1422392, 0.1368878, -0.1656954, 0.1447515
2: -0.0098809, 0.0388057, -0.0776582, 0.2169967, -0.2268775, 0.1164639
3: -0.0118928, 0.0395258, -0.0937744, 0.1550616, -0.1669543, 0.1333002
4: -0.0281270, 0.0076183, -0.1659643, 0.1157391, -0.1438661, 0.1735826
5: -0.0134682, 0.1062672, -0.1610111, 0.2419932, -0.2554614, 0.2672783
6: -0.0213973, 0.0197640, -0.1095519, 0.1532343, -0.1746316, 0.1293159
7: -0.0358046, 0.0139160, -0.1827449, 0.1555478, -0.1913524, 0.1966610
8: -0.0226195, 0.0364873, -0.1573077, 0.1689758, -0.1915953, 0.1937950
9: -0.0168738, 0.0294309, -0.1354169, 0.2055874, -0.2224613, 0.1648478

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A1_B2_A1_A2_B2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5245738, upper bound: 0.5236537
time: 1.90 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5205231, upper bound: 0.5030436
time: 1.68 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: 0.8358850, 1.0044501, 0.6459141, 1.0323212, -0.1964362, 0.3585359
1: -0.0393129, 0.0220015, -0.1430736, 0.1376408, -0.1769536, 0.1650751
2: -0.0092320, 0.0687284, -0.0782001, 0.2180582, -0.2272901, 0.1469285
3: -0.0217750, 0.0602244, -0.0942512, 0.1557229, -0.1774979, 0.1544755
4: -0.0502155, 0.0173809, -0.1667979, 0.1164343, -0.1666498, 0.1841788
5: -0.0199928, 0.1391454, -0.1619854, 0.2427009, -0.2626937, 0.3011308
6: -0.0351404, 0.0367722, -0.1100436, 0.1541851, -0.1893255, 0.1468158
7: -0.0599598, 0.0329333, -0.1835870, 0.1565016, -0.2164613, 0.2165203
8: -0.0371408, 0.0578018, -0.1581875, 0.1698031, -0.2069439, 0.2159892
9: -0.0393417, 0.0438005, -0.1361360, 0.2066389, -0.2459806, 0.1799365

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A1_B2_A2_A1_B1_B1

### Relational analysis result of NS_A1_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5215143, upper bound: 0.5237289
time: 1.99 seconds

## Relational analysis of NS_A1_B2_A2_A1_B1_B2

### Relational analysis result of NS_A1_B2_A2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5176995, upper bound: 0.5031165
time: 1.58 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: 0.8334101, 1.0046064, 0.6118456, 1.0383583, -0.2049482, 0.3927608
1: -0.0400424, 0.0237515, -0.1642140, 0.1567242, -0.1967665, 0.1879655
2: -0.0094383, 0.0706261, -0.0919322, 0.2449462, -0.2543844, 0.1625583
3: -0.0229016, 0.0613562, -0.1063294, 0.1724814, -0.1953829, 0.1676856
4: -0.0521463, 0.0179974, -0.1879212, 0.1340455, -0.1861919, 0.2059186
5: -0.0215188, 0.1409431, -0.1866700, 0.2606366, -0.2821554, 0.3276131
6: -0.0360178, 0.0379951, -0.1225035, 0.1782784, -0.2142962, 0.1604986
7: -0.0619026, 0.0339731, -0.2049245, 0.1806690, -0.2425716, 0.2388976
8: -0.0384843, 0.0593510, -0.1804768, 0.1907647, -0.2292490, 0.2398278
9: -0.0406193, 0.0460488, -0.1543545, 0.2332817, -0.2739010, 0.2004033

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A1_B2_A2_A1_B2_B1

### Relational analysis result of NS_A1_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5217242, upper bound: 0.5237289
time: 1.83 seconds

## Relational analysis of NS_A1_B2_A2_A1_B2_B2

### Relational analysis result of NS_A1_B2_A2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5179350, upper bound: 0.5031165
time: 1.86 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: 0.8491482, 1.0048802, 0.6838112, 1.0256059, -0.1764578, 0.3210690
1: -0.0361944, 0.0126234, -0.1195573, 0.1164131, -0.1526076, 0.1321807
2: -0.0097999, 0.0596827, -0.0629248, 0.1881480, -0.1979479, 0.1226075
3: -0.0169888, 0.0541588, -0.0808155, 0.1370813, -0.1540701, 0.1349743
4: -0.0423703, 0.0140774, -0.1433007, 0.0968438, -0.1392140, 0.1573781
5: -0.0133662, 0.1295114, -0.1345268, 0.2227496, -0.2361159, 0.2640381
6: -0.0304387, 0.0312918, -0.0961836, 0.1273843, -0.1578230, 0.1274754
7: -0.0510733, 0.0273609, -0.1598517, 0.1296182, -0.1806915, 0.1872126
8: -0.0327193, 0.0494996, -0.1333932, 0.1464858, -0.1792051, 0.1828928
9: -0.0324951, 0.0370922, -0.1158700, 0.1770019, -0.2094970, 0.1529623

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5265443, upper bound: 0.5055614
time: 1.78 seconds

## Relational analysis of NS_A1_B2_A2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5166133, upper bound: 0.5054495
time: 1.76 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: 0.8465400, 1.0050348, 0.6507534, 1.0314637, -0.1849237, 0.3542814
1: -0.0367965, 0.0144676, -0.1400707, 0.1349303, -0.1717268, 0.1545383
2: -0.0100041, 0.0614455, -0.0762496, 0.2142388, -0.2242429, 0.1376951
3: -0.0179123, 0.0553516, -0.0925355, 0.1533426, -0.1712549, 0.1478871
4: -0.0438777, 0.0147270, -0.1637974, 0.1139327, -0.1578104, 0.1785244
5: -0.0139885, 0.1314059, -0.1584792, 0.2401533, -0.2541417, 0.2898850
6: -0.0313633, 0.0323542, -0.1082738, 0.1507629, -0.1821262, 0.1406281
7: -0.0527992, 0.0284566, -0.1805563, 0.1530689, -0.2058681, 0.2090129
8: -0.0335494, 0.0511322, -0.1550215, 0.1668256, -0.2003750, 0.2061537
9: -0.0338413, 0.0383359, -0.1335482, 0.2028544, -0.2366958, 0.1718841

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5265972, upper bound: 0.5053340
time: 1.82 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5166573, upper bound: 0.5052255
time: 1.59 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.34 seconds
NS_A1_B1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -0.5506999, upper bound: 0.5339829
NS_A1_B1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -0.5542444, upper bound: 0.5399651
NS_A1_B1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -0.5467199, upper bound: 0.5332929
NS_A1_B1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -0.5466415, upper bound: 0.5291475
NS_A1_B1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -0.5313355, upper bound: 0.5282933
NS_A1_B1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -0.5303027, upper bound: 0.5283570
NS_A1_B1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -0.5287414, upper bound: 0.5282835
NS_A1_B1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -0.5282396, upper bound: 0.5283386
NS_A1_B1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -0.5504173, upper bound: 0.5333815
NS_A1_B1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -0.5540427, upper bound: 0.5391467
NS_A1_B1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -0.5464146, upper bound: 0.5323510
NS_A1_B1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -0.5464019, upper bound: 0.5291066
NS_A1_B1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -0.5314445, upper bound: 0.5282316
NS_A1_B1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -0.5287801, upper bound: 0.5282245
NS_A1_B1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -0.5304170, upper bound: 0.5282931
NS_A1_B1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -0.5282776, upper bound: 0.5282765
NS_A1_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -0.5220156, upper bound: 0.5239308
NS_A1_B2_A1_A1_B1_B2, status: Status.VERIFIED, split count: 6, time: 4.34
Output dim: 0, lower bound: -0.5180828, upper bound: 0.5030726
NS_A1_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -0.5222578, upper bound: 0.5239297
NS_A1_B2_A1_A1_B2_B2, status: Status.VERIFIED, split count: 6, time: 4.34
Output dim: 0, lower bound: -0.5183926, upper bound: 0.5030737
NS_A1_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -0.5268245, upper bound: 0.5030988
NS_A1_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -0.5203080, upper bound: 0.5030436
NS_A1_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -0.5245738, upper bound: 0.5236537
NS_A1_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -0.5205231, upper bound: 0.5030436
NS_A1_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -0.5215143, upper bound: 0.5237289
NS_A1_B2_A2_A1_B1_B2, status: Status.VERIFIED, split count: 6, time: 4.34
Output dim: 0, lower bound: -0.5176995, upper bound: 0.5031165
NS_A1_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -0.5217242, upper bound: 0.5237289
NS_A1_B2_A2_A1_B2_B2, status: Status.VERIFIED, split count: 6, time: 4.34
Output dim: 0, lower bound: -0.5179350, upper bound: 0.5031165
NS_A1_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -0.5265443, upper bound: 0.5055614
NS_A1_B2_A2_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 4.34
Output dim: 0, lower bound: -0.5166133, upper bound: 0.5054495
NS_A1_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -0.5265972, upper bound: 0.5053340
NS_A1_B2_A2_A2_B2_A2, status: Status.VERIFIED, split count: 6, time: 4.34
Output dim: 0, lower bound: -0.5166573, upper bound: 0.5052255

## BFS NS instance: NS_A1_B1_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: 0.8202676, 1.0047050, 0.8663340, 1.0040419, -0.1837744, 0.1383710
1: -0.0449635, 0.0321313, -0.0322273, 0.0068918, -0.0518553, 0.0643586
2: -0.0106995, 0.0810227, -0.0086929, 0.0480662, -0.0587657, 0.0897155
3: -0.0291773, 0.0667761, -0.0138994, 0.0462992, -0.0754765, 0.0806755
4: -0.0613920, 0.0241864, -0.0346613, 0.0097968, -0.0711888, 0.0588478
5: -0.0316991, 0.1495515, -0.0119740, 0.1170280, -0.1487270, 0.1615255
6: -0.0416015, 0.0438512, -0.0255130, 0.0242904, -0.0658919, 0.0693643
7: -0.0712969, 0.0389525, -0.0421844, 0.0201403, -0.0914372, 0.0811369
8: -0.0449176, 0.0681828, -0.0272488, 0.0419537, -0.0868712, 0.0954316
9: -0.0467372, 0.0595922, -0.0236233, 0.0328713, -0.0796084, 0.0832155

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_B1_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5624914, upper bound: 0.5534159
time: 1.99 seconds

## Relational analysis of NS_A1_B1_B1_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5621319, upper bound: 0.5535158
time: 1.93 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: 0.8166971, 1.0047272, 0.8543697, 1.0041176, -0.1874205, 0.1503575
1: -0.0451527, 0.0328187, -0.0349891, 0.0106236, -0.0557763, 0.0678078
2: -0.0108656, 0.0817207, -0.0087930, 0.0561533, -0.0670189, 0.0905137
3: -0.0297570, 0.0671584, -0.0159296, 0.0517708, -0.0815278, 0.0830880
4: -0.0618878, 0.0244954, -0.0399386, 0.0127768, -0.0746646, 0.0644340
5: -0.0325617, 0.1501864, -0.0120999, 0.1257185, -0.1582802, 0.1622864
6: -0.0418531, 0.0447444, -0.0288952, 0.0291646, -0.0710176, 0.0736395
7: -0.0716337, 0.0403606, -0.0482726, 0.0251670, -0.0968008, 0.0886332
8: -0.0451924, 0.0689171, -0.0310572, 0.0470776, -0.0922700, 0.0999743
9: -0.0470873, 0.0602508, -0.0297996, 0.0356498, -0.0827371, 0.0900503

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_B1_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5630520, upper bound: 0.5626789
time: 1.96 seconds

## Relational analysis of NS_A1_B1_B1_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5626223, upper bound: 0.5627622
time: 1.80 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: 0.7892460, 1.0048916, 0.7425401, 1.0191224, -0.2298763, 0.2623516
1: -0.0469866, 0.0385002, -0.0742948, 0.0814940, -0.1284806, 0.1127950
2: -0.0123552, 0.0876365, -0.0298398, 0.1425786, -0.1549339, 0.1174764
3: -0.0345384, 0.0703684, -0.0663572, 0.0987031, -0.1332415, 0.1367256
4: -0.0662785, 0.0272925, -0.1158556, 0.0614206, -0.1276990, 0.1431481
5: -0.0395991, 0.1554725, -0.0923569, 0.2002612, -0.2398604, 0.2478293
6: -0.0441830, 0.0520657, -0.0748249, 0.0783476, -0.1225306, 0.1268906
7: -0.0749561, 0.0516683, -0.1266570, 0.0682839, -0.1432399, 0.1783253
8: -0.0477673, 0.0749568, -0.0828138, 0.1205471, -0.1683144, 0.1577706
9: -0.0501407, 0.0662418, -0.0827755, 0.1400388, -0.1901795, 0.1490173

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_B1_A1_B2_B1_A1

### Relational analysis result of NS_A1_B1_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5463741, upper bound: 0.5318182
time: 1.93 seconds

## Relational analysis of NS_A1_B1_B1_A1_B2_B1_A2

### Relational analysis result of NS_A1_B1_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5461564, upper bound: 0.5318775
time: 1.79 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: 0.8355923, 1.0047078, 0.7463353, 1.0182060, -0.1826137, 0.2583725
1: -0.0393991, 0.0222086, -0.0728626, 0.0790838, -0.1184830, 0.0950712
2: -0.0095722, 0.0689528, -0.0289052, 0.1395731, -0.1491452, 0.0978581
3: -0.0219083, 0.0603582, -0.0645418, 0.0971442, -0.1190525, 0.1249000
4: -0.0504439, 0.0174538, -0.1131963, 0.0596025, -0.1100463, 0.1306501
5: -0.0201733, 0.1393579, -0.0893951, 0.1977852, -0.2179585, 0.2287530
6: -0.0352442, 0.0369168, -0.0732028, 0.0766632, -0.1119073, 0.1101195
7: -0.0601895, 0.0330563, -0.1239540, 0.0668518, -0.1270413, 0.1570102
8: -0.0372997, 0.0579850, -0.0809634, 0.1179903, -0.1552901, 0.1389484
9: -0.0394928, 0.0440665, -0.0810159, 0.1361108, -0.1756036, 0.1250823

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_B1_A1_B2_B2_B1

### Relational analysis result of NS_A1_B1_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5461173, upper bound: 0.5289160
time: 1.67 seconds

## Relational analysis of NS_A1_B1_B1_A1_B2_B2_B2

### Relational analysis result of NS_A1_B1_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5461226, upper bound: 0.5284352
time: 2.69 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A1_A1

### Backsubstitution after applying NS history:
0: 0.7233108, 1.0180336, 0.8298602, 1.0041044, -0.2807935, 0.1881734
1: -0.0913430, 0.1037963, -0.0381349, 0.0198870, -0.1112300, 0.1419312
2: -0.0493925, 0.1477774, -0.0087623, 0.0664386, -0.1158311, 0.1565397
3: -0.0682864, 0.1257637, -0.0211951, 0.0585981, -0.1268844, 0.1469589
4: -0.1186463, 0.0852003, -0.0477147, 0.0164541, -0.1351004, 0.1329149
5: -0.1146450, 0.2070999, -0.0188956, 0.1366661, -0.2513111, 0.2259955
6: -0.0869213, 0.0923915, -0.0335343, 0.0372032, -0.1241245, 0.1259257
7: -0.1416350, 0.1114487, -0.0565267, 0.0354500, -0.1770850, 0.1679754
8: -0.1078426, 0.1186678, -0.0354983, 0.0561685, -0.1640111, 0.1541661
9: -0.0985374, 0.1558403, -0.0372102, 0.0416245, -0.1401619, 0.1930505

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 93

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_B1_A2_A1_A1_B1

### Relational analysis result of NS_A1_B1_B1_A2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5090392, upper bound: 0.5178166
time: 1.45 seconds

## Relational analysis of NS_A1_B1_B1_A2_A1_A1_B2

### Relational analysis result of NS_A1_B1_B1_A2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5082473, upper bound: 0.5042918
time: 1.34 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A1_A2

### Backsubstitution after applying NS history:
0: 0.6986473, 1.0221119, 0.8240388, 1.0042655, -0.3056183, 0.1980731
1: -0.1042474, 0.1224479, -0.0387173, 0.0220596, -0.1263071, 0.1611652
2: -0.0604447, 0.1640581, -0.0089697, 0.0684483, -0.1288930, 0.1730278
3: -0.0777331, 0.1420164, -0.0224640, 0.0599226, -0.1376557, 0.1644804
4: -0.1324450, 0.1024131, -0.0493108, 0.0171612, -0.1496062, 0.1517239
5: -0.1369792, 0.2211264, -0.0207836, 0.1388004, -0.2757796, 0.2419101
6: -0.0985779, 0.1054085, -0.0344612, 0.0390213, -0.1375992, 0.1398698
7: -0.1594803, 0.1339541, -0.0581650, 0.0379162, -0.1973965, 0.1921191
8: -0.1248644, 0.1305752, -0.0363304, 0.0581534, -0.1830178, 0.1669056
9: -0.1128684, 0.1800752, -0.0386191, 0.0429779, -0.1558463, 0.2186943

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 93

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_B1_A2_A1_A2_B1

### Relational analysis result of NS_A1_B1_B1_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5080748, upper bound: 0.5177051
time: 1.50 seconds

## Relational analysis of NS_A1_B1_B1_A2_A1_A2_B2

### Relational analysis result of NS_A1_B1_B1_A2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5074322, upper bound: 0.5043469
time: 1.37 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A2_A1

### Backsubstitution after applying NS history:
0: 0.7278847, 1.0172775, 0.8600718, 1.0040331, -0.2761484, 0.1572056
1: -0.0889499, 0.1003376, -0.0336728, 0.0088450, -0.0977949, 0.1340104
2: -0.0473429, 0.1447582, -0.0086814, 0.0522990, -0.0996419, 0.1534396
3: -0.0665346, 0.1227498, -0.0149620, 0.0491631, -0.1156976, 0.1377117
4: -0.1160873, 0.0820083, -0.0374234, 0.0113566, -0.1274439, 0.1194317
5: -0.1105033, 0.2044986, -0.0119595, 0.1215766, -0.2320799, 0.2164581
6: -0.0847597, 0.0899775, -0.0272833, 0.0268416, -0.1116013, 0.1172607
7: -0.1383257, 0.1072751, -0.0453710, 0.0227713, -0.1610970, 0.1526461
8: -0.1046860, 0.1164597, -0.0292421, 0.0446356, -0.1493216, 0.1457018
9: -0.0958798, 0.1513461, -0.0268560, 0.0343256, -0.1302053, 0.1782020

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_B1_A2_A2_A1_B1

### Relational analysis result of NS_A1_B1_B1_A2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5052731, upper bound: 0.5172062
time: 1.39 seconds

## Relational analysis of NS_A1_B1_B1_A2_A2_A1_B2

### Relational analysis result of NS_A1_B1_B1_A2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5051384, upper bound: 0.5042331
time: 1.43 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A2_A2

### Backsubstitution after applying NS history:
0: 0.7041814, 1.0211968, 0.8575732, 1.0041779, -0.2999965, 0.1636236
1: -0.1013520, 0.1182628, -0.0342496, 0.0096244, -0.1109764, 0.1525125
2: -0.0579648, 0.1604051, -0.0088726, 0.0539880, -0.1119528, 0.1692777
3: -0.0756135, 0.1383697, -0.0153860, 0.0503058, -0.1259193, 0.1537557
4: -0.1293488, 0.0985510, -0.0385256, 0.0119789, -0.1413277, 0.1370765
5: -0.1319679, 0.2179792, -0.0121999, 0.1233916, -0.2553596, 0.2301791
6: -0.0959625, 0.1024879, -0.0279896, 0.0278596, -0.1238220, 0.1304775
7: -0.1554762, 0.1289043, -0.0466425, 0.0238211, -0.1792973, 0.1755468
8: -0.1210451, 0.1279034, -0.0300375, 0.0457057, -0.1667507, 0.1579408
9: -0.1096529, 0.1746373, -0.0281459, 0.0349058, -0.1445587, 0.2027832

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_B1_A2_A2_A2_A1

### Relational analysis result of NS_A1_B1_B1_A2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5173235, upper bound: 0.5044173
time: 1.60 seconds

## Relational analysis of NS_A1_B1_B1_A2_A2_A2_A2

### Relational analysis result of NS_A1_B1_B1_A2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5039722, upper bound: 0.5042736
time: 1.55 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: 0.8222907, 1.0047038, 0.8283659, 1.0042260, -0.1819353, 0.1763379
1: -0.0442001, 0.0308464, -0.0419075, 0.0269883, -0.0711884, 0.0727539
2: -0.0102013, 0.0794204, -0.0089361, 0.0746092, -0.0848105, 0.0883565
3: -0.0282096, 0.0659452, -0.0253037, 0.0634497, -0.0916593, 0.0912488
4: -0.0599745, 0.0232172, -0.0557176, 0.0203071, -0.0802816, 0.0789349
5: -0.0301202, 0.1482317, -0.0253792, 0.1442682, -0.1743884, 0.1736109
6: -0.0407367, 0.0429533, -0.0381400, 0.0402572, -0.0809939, 0.0810933
7: -0.0698559, 0.0381890, -0.0655291, 0.0358965, -0.1057524, 0.1037181
8: -0.0439313, 0.0668197, -0.0409693, 0.0627271, -0.1066584, 0.1077890
9: -0.0457991, 0.0574985, -0.0429824, 0.0512108, -0.0970099, 0.1004809

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_B2_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5624674, upper bound: 0.5530632
time: 2.72 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5621157, upper bound: 0.5531623
time: 1.62 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: 0.8190148, 1.0047238, 0.8150617, 1.0043045, -0.1852897, 0.1896621
1: -0.0443865, 0.0314959, -0.0469280, 0.0354376, -0.0798241, 0.0784239
2: -0.0103617, 0.0800856, -0.0119815, 0.0851454, -0.0955071, 0.0920671
3: -0.0287546, 0.0663082, -0.0316676, 0.0689145, -0.0976691, 0.0979758
4: -0.0604517, 0.0235172, -0.0650399, 0.0266803, -0.0871320, 0.0885571
5: -0.0309337, 0.1488336, -0.0357618, 0.1529480, -0.1838817, 0.1845953
6: -0.0409817, 0.0437814, -0.0438267, 0.0461617, -0.0871434, 0.0876080
7: -0.0701895, 0.0394799, -0.0750048, 0.0409170, -0.1111065, 0.1144848
8: -0.0441996, 0.0675131, -0.0474558, 0.0716901, -0.1158897, 0.1149689
9: -0.0461350, 0.0581364, -0.0491510, 0.0649804, -0.1111154, 0.1072874

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_B2_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5630906, upper bound: 0.5625753
time: 1.55 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5626636, upper bound: 0.5626636
time: 3.27 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: 0.7916230, 1.0048839, 0.6434450, 1.0283749, -0.2367519, 0.3614389
1: -0.0462106, 0.0371745, -0.1290509, 0.1510914, -0.1973020, 0.1662255
2: -0.0118450, 0.0859890, -0.0774177, 0.2269461, -0.2387912, 0.1634067
3: -0.0335310, 0.0695080, -0.0961285, 0.1669761, -0.2005071, 0.1656365
4: -0.0648247, 0.0263080, -0.1559084, 0.1288472, -0.1936719, 0.1822164
5: -0.0379520, 0.1541075, -0.1719969, 0.2426674, -0.2806194, 0.3261045
6: -0.0433007, 0.0510944, -0.1164792, 0.1297210, -0.1730218, 0.1675736
7: -0.0734933, 0.0507625, -0.1882809, 0.1685160, -0.2420094, 0.2390433
8: -0.0467620, 0.0735395, -0.1661425, 0.1488616, -0.1956236, 0.2396820
9: -0.0491769, 0.0641009, -0.1391714, 0.2172933, -0.2664702, 0.2032723

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_B2_A1_B2_B1_B1

### Relational analysis result of NS_A1_B1_B2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5458400, upper bound: 0.5320671
time: 1.68 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2_B1_B2

### Relational analysis result of NS_A1_B1_B2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5458433, upper bound: 0.5308755
time: 2.34 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: 0.8372595, 1.0047067, 0.6582043, 1.0262364, -0.1889769, 0.3465024
1: -0.0389387, 0.0210296, -0.1217582, 0.1413098, -0.1802485, 0.1427878
2: -0.0095708, 0.0677185, -0.0716216, 0.2144174, -0.2239882, 0.1393400
3: -0.0211984, 0.0595958, -0.0907648, 0.1584525, -0.1796509, 0.1503606
4: -0.0492412, 0.0170386, -0.1484325, 0.1198201, -0.1690614, 0.1654712
5: -0.0191734, 0.1381469, -0.1602083, 0.2353113, -0.2544847, 0.2983553
6: -0.0346531, 0.0361351, -0.1103660, 0.1224393, -0.1570924, 0.1465011
7: -0.0589405, 0.0323558, -0.1787751, 0.1567136, -0.2156541, 0.2111309
8: -0.0365035, 0.0569414, -0.1556213, 0.1426169, -0.1791204, 0.2125628
9: -0.0386321, 0.0427611, -0.1312034, 0.2045837, -0.2432158, 0.1739645

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_B2_A1_B2_B2_B1

### Relational analysis result of NS_A1_B1_B2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5458351, upper bound: 0.5288768
time: 1.79 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2_B2_B2

### Relational analysis result of NS_A1_B1_B2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5458376, upper bound: 0.5283731
time: 1.73 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A1_A1

### Backsubstitution after applying NS history:
0: 0.7255796, 1.0176586, 0.7769074, 1.0053364, -0.2797568, 0.2407512
1: -0.0901560, 0.1020805, -0.0533449, 0.0482001, -0.1383561, 0.1554254
2: -0.0483758, 0.1462797, -0.0166665, 0.0993945, -0.1477703, 0.1629462
3: -0.0674173, 0.1242687, -0.0416836, 0.0769033, -0.1443207, 0.1659524
4: -0.1173769, 0.0836168, -0.0770141, 0.0351513, -0.1525282, 0.1606309
5: -0.1125905, 0.2058097, -0.0512992, 0.1658313, -0.2784218, 0.2571088
6: -0.0858491, 0.0911941, -0.0511420, 0.0579241, -0.1437732, 0.1423361
7: -0.1399934, 0.1093786, -0.0861526, 0.0572444, -0.1972378, 0.1955312
8: -0.1062768, 0.1175725, -0.0558566, 0.0848098, -0.1910866, 0.1734291
9: -0.0972191, 0.1536109, -0.0574962, 0.0823966, -0.1796158, 0.2111071

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 93

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_B2_A2_A1_A1_B1

### Relational analysis result of NS_A1_B1_B2_A2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5091478, upper bound: 0.5178166
time: 1.64 seconds

## Relational analysis of NS_A1_B1_B2_A2_A1_A1_B2

### Relational analysis result of NS_A1_B1_B2_A2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5082986, upper bound: 0.5040271
time: 1.39 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A1_A2

### Backsubstitution after applying NS history:
0: 0.7295126, 1.0170083, 0.8226043, 1.0041847, -0.2746722, 0.1944040
1: -0.0880982, 0.0991066, -0.0440817, 0.0306474, -0.1187456, 0.1431883
2: -0.0466135, 0.1436836, -0.0101241, 0.0791721, -0.1257856, 0.1538078
3: -0.0659111, 0.1216771, -0.0280597, 0.0658164, -0.1317275, 0.1497368
4: -0.1151767, 0.0808723, -0.0597548, 0.0230671, -0.1382438, 0.1406271
5: -0.1090292, 0.2035730, -0.0298756, 0.1480272, -0.2570564, 0.2334486
6: -0.0839904, 0.0891183, -0.0406027, 0.0428142, -0.1268046, 0.1297211
7: -0.1371479, 0.1057900, -0.0696327, 0.0380707, -0.1752186, 0.1754227
8: -0.1035625, 0.1156738, -0.0437784, 0.0666087, -0.1701712, 0.1594521
9: -0.0949340, 0.1497466, -0.0456538, 0.0571740, -0.1521080, 0.1954003

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_B2_A2_A1_A2_B1

### Relational analysis result of NS_A1_B1_B2_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5053137, upper bound: 0.5172026
time: 1.65 seconds

## Relational analysis of NS_A1_B1_B2_A2_A1_A2_B2

### Relational analysis result of NS_A1_B1_B2_A2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5051687, upper bound: 0.5039678
time: 1.42 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A2_A1

### Backsubstitution after applying NS history:
0: 0.7014718, 1.0216448, 0.7697109, 1.0060059, -0.3045341, 0.2519339
1: -0.1027697, 0.1203121, -0.0550593, 0.0516090, -0.1543787, 0.1753715
2: -0.0591790, 0.1621938, -0.0184283, 0.1020743, -0.1612534, 0.1806222
3: -0.0766513, 0.1401552, -0.0435583, 0.0795593, -0.1562107, 0.1837135
4: -0.1308648, 0.1004419, -0.0794724, 0.0374892, -0.1683540, 0.1799143
5: -0.1344216, 0.2195202, -0.0547802, 0.1684853, -0.3029069, 0.2743005
6: -0.0972431, 0.1039179, -0.0531300, 0.0602246, -0.1574677, 0.1570479
7: -0.1574368, 0.1313768, -0.0888955, 0.0615135, -0.2189503, 0.2202724
8: -0.1229151, 0.1292116, -0.0584222, 0.0870736, -0.2099887, 0.1876338
9: -0.1112273, 0.1772999, -0.0595005, 0.0865759, -0.1978031, 0.2368004

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 93

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_B2_A2_A2_A1_B1

### Relational analysis result of NS_A1_B1_B2_A2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5081830, upper bound: 0.5177046
time: 1.68 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2_A1_B2

### Relational analysis result of NS_A1_B1_B2_A2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5074701, upper bound: 0.5040792
time: 1.85 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A2_A2

### Backsubstitution after applying NS history:
0: 0.7063016, 1.0208461, 0.8198258, 1.0043418, -0.2980402, 0.2010203
1: -0.1002425, 0.1166594, -0.0451302, 0.0324119, -0.1326544, 0.1617896
2: -0.0570146, 0.1590054, -0.0108083, 0.0813725, -0.1383871, 0.1698137
3: -0.0748013, 0.1369724, -0.0293887, 0.0669576, -0.1417589, 0.1663611
4: -0.1281625, 0.0970710, -0.0617017, 0.0243981, -0.1525605, 0.1587727
5: -0.1300479, 0.2167732, -0.0320438, 0.1498398, -0.2798877, 0.2488170
6: -0.0949603, 0.1013687, -0.0417903, 0.0440473, -0.1390076, 0.1431590
7: -0.1539420, 0.1269695, -0.0716116, 0.0391192, -0.1930612, 0.1985811
8: -0.1195817, 0.1268797, -0.0451330, 0.0684804, -0.1880621, 0.1720127
9: -0.1084208, 0.1725539, -0.0469420, 0.0600496, -0.1684704, 0.2194960

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_B2_A2_A2_A2_A1

### Relational analysis result of NS_A1_B1_B2_A2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5171246, upper bound: 0.5041586
time: 1.37 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2_A2_A2

### Relational analysis result of NS_A1_B1_B2_A2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5040038, upper bound: 0.5040038
time: 1.36 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: 0.8753143, 1.0042808, 0.6590844, 1.0299875, -0.1546732, 0.3451964
1: -0.0300950, 0.0040187, -0.1349010, 0.1302636, -0.1603587, 0.1389197
2: -0.0090083, 0.0419846, -0.0728916, 0.2076634, -0.2166717, 0.1148762
3: -0.0124411, 0.0420747, -0.0895819, 0.1492444, -0.1616855, 0.1316566
4: -0.0305869, 0.0078206, -0.1586319, 0.1096260, -0.1402129, 0.1664525
5: -0.0123707, 0.1103182, -0.1524427, 0.2357673, -0.2481380, 0.2627609
6: -0.0229018, 0.0208522, -0.1052269, 0.1448711, -0.1677729, 0.1260791
7: -0.0377133, 0.0162592, -0.1753382, 0.1471590, -0.1848723, 0.1915974
8: -0.0243084, 0.0381867, -0.1495708, 0.1616996, -0.1860080, 0.1877575
9: -0.0190277, 0.0307260, -0.1290930, 0.1963392, -0.2153669, 0.1598190

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A1_A1_B1_B1_A1

### Relational analysis result of NS_A1_B2_A1_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4875081, upper bound: 0.4880029
time: 1.54 seconds

## Relational analysis of NS_A1_B2_A1_A1_B1_B1_A2

### Relational analysis result of NS_A1_B2_A1_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4757259, upper bound: 0.4803642
time: 1.32 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: 0.8731177, 1.0044266, 0.6245476, 1.0361073, -0.1629896, 0.3798790
1: -0.0306614, 0.0047760, -0.1563321, 0.1496092, -0.1802705, 0.1611080
2: -0.0092010, 0.0434808, -0.0868123, 0.2349214, -0.2441224, 0.1302932
3: -0.0127483, 0.0431967, -0.1018262, 0.1662332, -0.1789815, 0.1450229
4: -0.0316691, 0.0081071, -0.1800455, 0.1274794, -0.1591485, 0.1881526
5: -0.0126130, 0.1121003, -0.1774667, 0.2539495, -0.2665626, 0.2895670
6: -0.0235953, 0.0215268, -0.1178580, 0.1692956, -0.1928909, 0.1393848
7: -0.0387324, 0.0172900, -0.1969690, 0.1716584, -0.2103908, 0.2142590
8: -0.0250894, 0.0390484, -0.1721665, 0.1829494, -0.2080388, 0.2112148
9: -0.0201214, 0.0312958, -0.1475619, 0.2233482, -0.2434696, 0.1788577

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A1_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_A1_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4882955, upper bound: 0.4880040
time: 1.80 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_A1_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4769896, upper bound: 0.4804103
time: 1.50 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.8885377, 1.0043397, 0.6809140, 1.0261191, -0.1375814, 0.3234257
1: -0.0264265, 0.0005289, -0.1213551, 0.1180361, -0.1444626, 0.1218840
2: -0.0090861, 0.0329261, -0.0640926, 0.1904346, -0.1995207, 0.0970187
3: -0.0108785, 0.0348175, -0.0818427, 0.1385065, -0.1493850, 0.1166602
4: -0.0235771, 0.0072597, -0.1450970, 0.0983416, -0.1219187, 0.1523568
5: -0.0124685, 0.0987744, -0.1366262, 0.2242749, -0.2367435, 0.2354006
6: -0.0188978, 0.0177513, -0.0972432, 0.1294332, -0.1483310, 0.1149945
7: -0.0326415, 0.0095821, -0.1616664, 0.1316736, -0.1643151, 0.1712485
8: -0.0198358, 0.0333439, -0.1352888, 0.1482683, -0.1681041, 0.1686328
9: -0.0132638, 0.0270352, -0.1174194, 0.1792679, -0.1925316, 0.1444546

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 242

### Candidate
type: A, layer: 1, pos: 210

## Relational analysis of NS_A1_B2_A1_A2_B1_A1_A1

### Relational analysis result of NS_A1_B2_A1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5266434, upper bound: 0.5024419
time: 1.71 seconds

## Relational analysis of NS_A1_B2_A1_A2_B1_A1_A2

### Relational analysis result of NS_A1_B2_A1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5266546, upper bound: 0.5025889
time: 1.79 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.8285141, 1.0040193, 0.6943018, 1.0237468, -0.1952327, 0.3097174
1: -0.0430785, 0.0144001, -0.1130477, 0.1105370, -0.1536154, 0.1274478
2: -0.0086632, 0.0740446, -0.0586964, 0.1798685, -0.1885317, 0.1327410
3: -0.0179716, 0.0677438, -0.0770964, 0.1319211, -0.1498927, 0.1448402
4: -0.0553958, 0.0097674, -0.1367964, 0.0914209, -0.1468168, 0.1465637
5: -0.0160868, 0.1511736, -0.1269259, 0.2172269, -0.2333137, 0.2780995
6: -0.0363773, 0.0318268, -0.0923469, 0.1199654, -0.1563427, 0.1241737
7: -0.0547615, 0.0398907, -0.1532815, 0.1221767, -0.1769382, 0.1931722
8: -0.0393038, 0.0553260, -0.1265299, 0.1400312, -0.1793350, 0.1818559
9: -0.0385105, 0.0437884, -0.1102602, 0.1687982, -0.2073087, 0.1540486

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 242

### Candidate
type: A, layer: 1, pos: 210

## Relational analysis of NS_A1_B2_A1_A2_B1_A2_A1

### Relational analysis result of NS_A1_B2_A1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5198610, upper bound: 0.5024076
time: 1.74 seconds

## Relational analysis of NS_A1_B2_A1_A2_B1_A2_A2

### Relational analysis result of NS_A1_B2_A1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5199137, upper bound: 0.5025229
time: 1.64 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: 0.8799546, 1.0049415, 0.6637397, 1.0291625, -0.1492079, 0.3412018
1: -0.0288077, 0.0025124, -0.1320123, 0.1276561, -0.1564637, 0.1345247
2: -0.0098809, 0.0388057, -0.0710151, 0.2039894, -0.2138703, 0.1098209
3: -0.0118928, 0.0395258, -0.0879315, 0.1469546, -0.1588473, 0.1274572
4: -0.0281270, 0.0076183, -0.1557456, 0.1072196, -0.1353466, 0.1633639
5: -0.0134682, 0.1062672, -0.1490698, 0.2333165, -0.2467847, 0.2553370
6: -0.0213973, 0.0197640, -0.1035243, 0.1415790, -0.1629763, 0.1232883
7: -0.0358046, 0.0139160, -0.1724228, 0.1438566, -0.1796612, 0.1863388
8: -0.0226195, 0.0364873, -0.1465252, 0.1588354, -0.1814550, 0.1830125
9: -0.0168738, 0.0294309, -0.1266036, 0.1926987, -0.2095726, 0.1560345

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_A1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4877903, upper bound: 0.4860400
time: 1.35 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_A2

### Relational analysis result of NS_A1_B2_A1_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4648138, upper bound: 0.4655482
time: 1.21 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: 0.8853946, 1.0046178, 0.5170766, 1.0551512, -0.1697566, 0.4875412
1: -0.0272985, 0.0012553, -0.2230208, 0.2098081, -0.2371067, 0.2242761
2: -0.0094534, 0.0350792, -0.1301309, 0.3197421, -0.3291954, 0.1652101
3: -0.0112499, 0.0365418, -0.1399275, 0.2190984, -0.2303484, 0.1764693
4: -0.0252434, 0.0073911, -0.2466801, 0.1830348, -0.2082782, 0.2540712
5: -0.0129304, 0.1015184, -0.2553353, 0.3105287, -0.3234591, 0.3568537
6: -0.0198131, 0.0184884, -0.1571631, 0.2452991, -0.2651122, 0.1756515
7: -0.0337998, 0.0111692, -0.2642790, 0.2478956, -0.2816954, 0.2754483
8: -0.0208552, 0.0344951, -0.2424794, 0.2490738, -0.2699290, 0.2769745
9: -0.0145858, 0.0279125, -0.2050331, 0.3073939, -0.3219798, 0.2329457

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 242

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5201075, upper bound: 0.5024261
time: 1.70 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5201711, upper bound: 0.5025229
time: 1.68 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: 0.8358850, 1.0044501, 0.6622137, 1.0294329, -0.1935478, 0.3422364
1: -0.0393129, 0.0220015, -0.1329592, 0.1285109, -0.1678238, 0.1549608
2: -0.0092320, 0.0687284, -0.0716303, 0.2051938, -0.2144258, 0.1403587
3: -0.0217750, 0.0602244, -0.0884725, 0.1477053, -0.1694803, 0.1486969
4: -0.0502155, 0.0173809, -0.1566917, 0.1080084, -0.1582239, 0.1740726
5: -0.0199928, 0.1391454, -0.1501756, 0.2341199, -0.2541127, 0.2893209
6: -0.0351404, 0.0367722, -0.1040825, 0.1426581, -0.1777985, 0.1408546
7: -0.0599598, 0.0329333, -0.1733784, 0.1449393, -0.2048990, 0.2063117
8: -0.0371408, 0.0578018, -0.1475236, 0.1597742, -0.1969151, 0.2053254
9: -0.0393417, 0.0438005, -0.1274197, 0.1938922, -0.2332339, 0.1712202

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A2_A1_B1_B1_A1

### Relational analysis result of NS_A1_B2_A2_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4869676, upper bound: 0.4877185
time: 1.41 seconds

## Relational analysis of NS_A1_B2_A2_A1_B1_B1_A2

### Relational analysis result of NS_A1_B2_A2_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4751341, upper bound: 0.4796515
time: 1.36 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: 0.8334101, 1.0046064, 0.6283644, 1.0354309, -0.2020208, 0.3762419
1: -0.0400424, 0.0237515, -0.1539636, 0.1474711, -0.1875135, 0.1777151
2: -0.0094383, 0.0706261, -0.0852739, 0.2319089, -0.2413472, 0.1559000
3: -0.0229016, 0.0613562, -0.1004729, 0.1643557, -0.1872573, 0.1618291
4: -0.0521463, 0.0179974, -0.1776790, 0.1255062, -0.1776525, 0.1956764
5: -0.0215188, 0.1409431, -0.1747010, 0.2519401, -0.2734589, 0.3156441
6: -0.0360178, 0.0379951, -0.1164621, 0.1665962, -0.2026140, 0.1544572
7: -0.0619026, 0.0339731, -0.1945785, 0.1689508, -0.2308534, 0.2285516
8: -0.0384843, 0.0593510, -0.1696693, 0.1806009, -0.2190852, 0.2290203
9: -0.0406193, 0.0460488, -0.1455208, 0.2203633, -0.2609825, 0.1915696

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A2_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_A2_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4877074, upper bound: 0.4877185
time: 1.60 seconds

## Relational analysis of NS_A1_B2_A2_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_A2_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4764254, upper bound: 0.4797634
time: 1.57 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.8616754, 1.0044340, 0.6838112, 1.0256059, -0.1639305, 0.3206228
1: -0.0333027, 0.0083449, -0.1195573, 0.1164131, -0.1497158, 0.1279022
2: -0.0092107, 0.0512151, -0.0629248, 0.1881480, -0.1973588, 0.1141399
3: -0.0146899, 0.0484297, -0.0808155, 0.1370813, -0.1517712, 0.1292452
4: -0.0367162, 0.0109571, -0.1433007, 0.0968438, -0.1335599, 0.1542578
5: -0.0126253, 0.1204118, -0.1345268, 0.2227496, -0.2353749, 0.2549386
6: -0.0268300, 0.0261883, -0.0961836, 0.1273843, -0.1542143, 0.1223719
7: -0.0445550, 0.0220975, -0.1598517, 0.1296182, -0.1741732, 0.1819493
8: -0.0287317, 0.0439488, -0.1333932, 0.1464858, -0.1752175, 0.1773420
9: -0.0260282, 0.0339532, -0.1158700, 0.1770019, -0.2030301, 0.1498232

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B2_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

### Candidate
type: A, layer: 1, pos: 210

## Relational analysis of NS_A1_B2_A2_A2_B1_A1_A1

### Relational analysis result of NS_A1_B2_A2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5244307, upper bound: 0.5050311
time: 1.74 seconds

## Relational analysis of NS_A1_B2_A2_A2_B1_A1_A2

### Relational analysis result of NS_A1_B2_A2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5264119, upper bound: 0.5054369
time: 1.94 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.8590994, 1.0045886, 0.6507534, 1.0314637, -0.1723644, 0.3538352
1: -0.0338973, 0.0091483, -0.1400707, 0.1349303, -0.1688276, 0.1492190
2: -0.0094149, 0.0529563, -0.0762496, 0.2142388, -0.2236537, 0.1292059
3: -0.0151270, 0.0496077, -0.0925355, 0.1533426, -0.1684696, 0.1421432
4: -0.0378524, 0.0115987, -0.1637974, 0.1139327, -0.1517851, 0.1753961
5: -0.0128820, 0.1222830, -0.1584792, 0.2401533, -0.2530353, 0.2807622
6: -0.0275581, 0.0272377, -0.1082738, 0.1507629, -0.1783210, 0.1355116
7: -0.0458658, 0.0231798, -0.1805563, 0.1530689, -0.1989347, 0.2037361
8: -0.0295516, 0.0450520, -0.1550215, 0.1668256, -0.1963772, 0.2000735
9: -0.0273579, 0.0345514, -0.1335482, 0.2028544, -0.2302124, 0.1680996

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B2_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

### Candidate
type: A, layer: 1, pos: 210

## Relational analysis of NS_A1_B2_A2_A2_B2_A1_A1

### Relational analysis result of NS_A1_B2_A2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5245326, upper bound: 0.5047668
time: 1.93 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2_A1_A2

### Relational analysis result of NS_A1_B2_A2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5264645, upper bound: 0.5052084
time: 1.96 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 6.85 seconds
NS_A1_B1_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5624914, upper bound: 0.5534159
NS_A1_B1_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5621319, upper bound: 0.5535158
NS_A1_B1_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5630520, upper bound: 0.5626789
NS_A1_B1_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5626223, upper bound: 0.5627622
NS_A1_B1_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5463741, upper bound: 0.5318182
NS_A1_B1_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5461564, upper bound: 0.5318775
NS_A1_B1_B1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5461173, upper bound: 0.5289160
NS_A1_B1_B1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5461226, upper bound: 0.5284352
NS_A1_B1_B1_A2_A1_A1_B1, status: Status.VERIFIED, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5090392, upper bound: 0.5178166
NS_A1_B1_B1_A2_A1_A1_B2, status: Status.VERIFIED, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5082473, upper bound: 0.5042918
NS_A1_B1_B1_A2_A1_A2_B1, status: Status.VERIFIED, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5080748, upper bound: 0.5177051
NS_A1_B1_B1_A2_A1_A2_B2, status: Status.VERIFIED, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5074322, upper bound: 0.5043469
NS_A1_B1_B1_A2_A2_A1_B1, status: Status.VERIFIED, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5052731, upper bound: 0.5172062
NS_A1_B1_B1_A2_A2_A1_B2, status: Status.VERIFIED, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5051384, upper bound: 0.5042331
NS_A1_B1_B1_A2_A2_A2_A1, status: Status.VERIFIED, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5173235, upper bound: 0.5044173
NS_A1_B1_B1_A2_A2_A2_A2, status: Status.VERIFIED, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5039722, upper bound: 0.5042736
NS_A1_B1_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5624674, upper bound: 0.5530632
NS_A1_B1_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5621157, upper bound: 0.5531623
NS_A1_B1_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5630906, upper bound: 0.5625753
NS_A1_B1_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5626636, upper bound: 0.5626636
NS_A1_B1_B2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5458400, upper bound: 0.5320671
NS_A1_B1_B2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5458433, upper bound: 0.5308755
NS_A1_B1_B2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5458351, upper bound: 0.5288768
NS_A1_B1_B2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5458376, upper bound: 0.5283731
NS_A1_B1_B2_A2_A1_A1_B1, status: Status.VERIFIED, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5091478, upper bound: 0.5178166
NS_A1_B1_B2_A2_A1_A1_B2, status: Status.VERIFIED, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5082986, upper bound: 0.5040271
NS_A1_B1_B2_A2_A1_A2_B1, status: Status.VERIFIED, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5053137, upper bound: 0.5172026
NS_A1_B1_B2_A2_A1_A2_B2, status: Status.VERIFIED, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5051687, upper bound: 0.5039678
NS_A1_B1_B2_A2_A2_A1_B1, status: Status.VERIFIED, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5081830, upper bound: 0.5177046
NS_A1_B1_B2_A2_A2_A1_B2, status: Status.VERIFIED, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5074701, upper bound: 0.5040792
NS_A1_B1_B2_A2_A2_A2_A1, status: Status.VERIFIED, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5171246, upper bound: 0.5041586
NS_A1_B1_B2_A2_A2_A2_A2, status: Status.VERIFIED, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5040038, upper bound: 0.5040038
NS_A1_B2_A1_A1_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.4875081, upper bound: 0.4880029
NS_A1_B2_A1_A1_B1_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.4757259, upper bound: 0.4803642
NS_A1_B2_A1_A1_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.4882955, upper bound: 0.4880040
NS_A1_B2_A1_A1_B2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.4769896, upper bound: 0.4804103
NS_A1_B2_A1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5266434, upper bound: 0.5024419
NS_A1_B2_A1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5266546, upper bound: 0.5025889
NS_A1_B2_A1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5198610, upper bound: 0.5024076
NS_A1_B2_A1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5199137, upper bound: 0.5025229
NS_A1_B2_A1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.4877903, upper bound: 0.4860400
NS_A1_B2_A1_A2_B2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.4648138, upper bound: 0.4655482
NS_A1_B2_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5201075, upper bound: 0.5024261
NS_A1_B2_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5201711, upper bound: 0.5025229
NS_A1_B2_A2_A1_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.4869676, upper bound: 0.4877185
NS_A1_B2_A2_A1_B1_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.4751341, upper bound: 0.4796515
NS_A1_B2_A2_A1_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.4877074, upper bound: 0.4877185
NS_A1_B2_A2_A1_B2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.4764254, upper bound: 0.4797634
NS_A1_B2_A2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5244307, upper bound: 0.5050311
NS_A1_B2_A2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5264119, upper bound: 0.5054369
NS_A1_B2_A2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5245326, upper bound: 0.5047668
NS_A1_B2_A2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.85
Output dim: 0, lower bound: -0.5264645, upper bound: 0.5052084

## BFS NS instance: NS_A1_B1_B1_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: 0.8520788, 1.0038782, 0.8716040, 1.0037847, -0.1517059, 0.1322742
1: -0.0355179, 0.0113381, -0.0310108, 0.0052481, -0.0407661, 0.0423488
2: -0.0084768, 0.0577017, -0.0083532, 0.0445041, -0.0529809, 0.0660549
3: -0.0163183, 0.0528185, -0.0130051, 0.0438890, -0.0602073, 0.0658236
4: -0.0409491, 0.0133474, -0.0323368, 0.0084842, -0.0494332, 0.0456842
5: -0.0117022, 0.1273826, -0.0115467, 0.1131999, -0.1249022, 0.1389293
6: -0.0295428, 0.0300978, -0.0240233, 0.0221435, -0.0516863, 0.0541211
7: -0.0494383, 0.0261295, -0.0395027, 0.0179260, -0.0673644, 0.0656323
8: -0.0317864, 0.0480587, -0.0255712, 0.0396967, -0.0714830, 0.0736300
9: -0.0309821, 0.0361819, -0.0209028, 0.0316474, -0.0626295, 0.0570847

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A1_B1_B1_A1_B1_B1_A1_A1

### Relational analysis result of NS_A1_B1_B1_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5421258, upper bound: 0.5395136
time: 3.59 seconds

## Relational analysis of NS_A1_B1_B1_A1_B1_B1_A1_A2

### Relational analysis result of NS_A1_B1_B1_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5484777, upper bound: 0.5395845
time: 2.31 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: 0.8333762, 1.0042772, 0.8691394, 1.0039308, -0.1705546, 0.1351379
1: -0.0400523, 0.0237755, -0.0315798, 0.0060168, -0.0460691, 0.0553553
2: -0.0090035, 0.0706522, -0.0085462, 0.0461701, -0.0551736, 0.0791984
3: -0.0229170, 0.0613717, -0.0134233, 0.0450162, -0.0679333, 0.0747950
4: -0.0521728, 0.0180058, -0.0334239, 0.0090980, -0.0612708, 0.0514297
5: -0.0215396, 0.1409677, -0.0117895, 0.1149903, -0.1365299, 0.1527572
6: -0.0360298, 0.0380118, -0.0247200, 0.0231476, -0.0591774, 0.0627318
7: -0.0619292, 0.0339875, -0.0407570, 0.0189616, -0.0808908, 0.0747444
8: -0.0385027, 0.0593722, -0.0263558, 0.0407522, -0.0792549, 0.0857280
9: -0.0406368, 0.0460795, -0.0221751, 0.0322198, -0.0728566, 0.0682547

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A1_B1_B1_A1_B1_B1_A2_A1

### Relational analysis result of NS_A1_B1_B1_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5417619, upper bound: 0.5395983
time: 2.04 seconds

## Relational analysis of NS_A1_B1_B1_A1_B1_B1_A2_A2

### Relational analysis result of NS_A1_B1_B1_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5484013, upper bound: 0.5396925
time: 1.99 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: 0.8515794, 1.0038812, 0.8596557, 1.0038632, -0.1522838, 0.1442255
1: -0.0356332, 0.0114938, -0.0337689, 0.0089749, -0.0446081, 0.0452627
2: -0.0084807, 0.0580393, -0.0084571, 0.0525803, -0.0610611, 0.0664964
3: -0.0164030, 0.0530469, -0.0150326, 0.0493533, -0.0657564, 0.0680795
4: -0.0411693, 0.0134718, -0.0376070, 0.0114602, -0.0526295, 0.0510788
5: -0.0117071, 0.1277453, -0.0116773, 0.1218789, -0.1335860, 0.1394226
6: -0.0296840, 0.0303013, -0.0274009, 0.0270111, -0.0566951, 0.0577022
7: -0.0496925, 0.0263394, -0.0455828, 0.0229461, -0.0726386, 0.0719221
8: -0.0319453, 0.0482726, -0.0293746, 0.0448138, -0.0767591, 0.0776472
9: -0.0312399, 0.0362978, -0.0270708, 0.0344222, -0.0656621, 0.0633687

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A1_B1_B1_A1_B1_B2_A1_A1

### Relational analysis result of NS_A1_B1_B1_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5429514, upper bound: 0.5486487
time: 3.06 seconds

## Relational analysis of NS_A1_B1_B1_A1_B1_B2_A1_A2

### Relational analysis result of NS_A1_B1_B1_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5488637, upper bound: 0.5486884
time: 2.08 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: 0.8328726, 1.0042794, 0.8572305, 1.0040063, -0.1711337, 0.1470489
1: -0.0402069, 0.0241264, -0.0343287, 0.0097313, -0.0499381, 0.0584551
2: -0.0090066, 0.0710402, -0.0086460, 0.0542196, -0.0632262, 0.0796862
3: -0.0231480, 0.0615986, -0.0154441, 0.0504625, -0.0736105, 0.0770427
4: -0.0525599, 0.0181482, -0.0386767, 0.0120642, -0.0646241, 0.0568250
5: -0.0218623, 0.1413281, -0.0119149, 0.1236405, -0.1455028, 0.1532430
6: -0.0362137, 0.0382570, -0.0280865, 0.0279991, -0.0642128, 0.0663435
7: -0.0623193, 0.0341959, -0.0468169, 0.0239651, -0.0862844, 0.0810128
8: -0.0387720, 0.0596910, -0.0301465, 0.0458525, -0.0846245, 0.0898375
9: -0.0408930, 0.0465464, -0.0283227, 0.0349855, -0.0758784, 0.0748692

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A1_B1_B1_A1_B1_B2_A2_A1

### Relational analysis result of NS_A1_B1_B1_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5425479, upper bound: 0.5487261
time: 2.07 seconds

## Relational analysis of NS_A1_B1_B1_A1_B1_B2_A2_A2

### Relational analysis result of NS_A1_B1_B1_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5487496, upper bound: 0.5487801
time: 2.07 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: 0.8475784, 1.0039035, 0.7482457, 1.0177448, -0.1701664, 0.2556579
1: -0.0365567, 0.0137334, -0.0721417, 0.0778705, -0.1144272, 0.0858751
2: -0.0085102, 0.0607436, -0.0284348, 0.1380601, -0.1465703, 0.0891784
3: -0.0175447, 0.0548766, -0.0636280, 0.0963595, -0.1139042, 0.1185045
4: -0.0432775, 0.0144684, -0.1118577, 0.0586873, -0.1019648, 0.1263261
5: -0.0134083, 0.1306515, -0.0879042, 0.1965388, -0.2099472, 0.2185557
6: -0.0309951, 0.0319312, -0.0723862, 0.0758153, -0.1068104, 0.1043174
7: -0.0521120, 0.0280203, -0.1225931, 0.0661309, -0.1182428, 0.1506135
8: -0.0332189, 0.0504822, -0.0800320, 0.1167032, -0.1499221, 0.1305142
9: -0.0333053, 0.0378408, -0.0801301, 0.1341336, -0.1674389, 0.1179709

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_B1_A1_B2_B1_A1_A1

### Relational analysis result of NS_A1_B1_B1_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5296526, upper bound: 0.5094137
time: 1.91 seconds

## Relational analysis of NS_A1_B1_B1_A1_B2_B1_A1_A2

### Relational analysis result of NS_A1_B1_B1_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5235425, upper bound: 0.5093117
time: 1.90 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: 0.8282191, 1.0043017, 0.7456528, 1.0183709, -0.1901518, 0.2586489
1: -0.0419629, 0.0270816, -0.0731202, 0.0795171, -0.1214801, 0.1002017
2: -0.0090360, 0.0747254, -0.0290733, 0.1401136, -0.1491496, 0.1037987
3: -0.0253739, 0.0635101, -0.0648682, 0.0974245, -0.1227984, 0.1283783
4: -0.0558205, 0.0203774, -0.1136745, 0.0599294, -0.1157499, 0.1340519
5: -0.0254938, 0.1443641, -0.0899277, 0.1982304, -0.2237242, 0.2342917
6: -0.0382027, 0.0403223, -0.0734944, 0.0769660, -0.1151688, 0.1138167
7: -0.0656336, 0.0359519, -0.1244399, 0.0671093, -0.1327429, 0.1603918
8: -0.0410408, 0.0628260, -0.0812962, 0.1184500, -0.1594908, 0.1441222
9: -0.0430505, 0.0513626, -0.0813323, 0.1368171, -0.1798676, 0.1326949

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_B1_A1_B2_B1_A2_A1

### Relational analysis result of NS_A1_B1_B1_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5294882, upper bound: 0.5094831
time: 1.64 seconds

## Relational analysis of NS_A1_B1_B1_A1_B2_B1_A2_A2

### Relational analysis result of NS_A1_B1_B1_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5234199, upper bound: 0.5093601
time: 2.65 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: 0.8411281, 1.0044446, 0.7731626, 1.0117297, -0.1706017, 0.2312820
1: -0.0380457, 0.0182943, -0.0627391, 0.0620466, -0.1000923, 0.0810335
2: -0.0092246, 0.0651037, -0.0222991, 0.1183274, -0.1275520, 0.0874027
3: -0.0198287, 0.0578266, -0.0517094, 0.0861248, -0.1059534, 0.1095360
4: -0.0470054, 0.0160750, -0.0943986, 0.0467513, -0.0937567, 0.1104736
5: -0.0170121, 0.1353370, -0.0684594, 0.1802831, -0.1972951, 0.2037964
6: -0.0332818, 0.0345591, -0.0617359, 0.0647569, -0.0980387, 0.0962949
7: -0.0563805, 0.0307305, -0.1048467, 0.0567282, -0.1131087, 0.1355772
8: -0.0352721, 0.0545198, -0.0678838, 0.0999170, -0.1351892, 0.1224037
9: -0.0366351, 0.0409165, -0.0685774, 0.1083453, -0.1449804, 0.1094939

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_B1_A1_B2_B2_B1_A1

### Relational analysis result of NS_A1_B1_B1_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5288378, upper bound: 0.5058712
time: 1.65 seconds

## Relational analysis of NS_A1_B1_B1_A1_B2_B2_B1_A2

### Relational analysis result of NS_A1_B1_B1_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5227326, upper bound: 0.5057898
time: 2.65 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: 0.8384817, 1.0045962, 0.7595805, 1.0150084, -0.1765268, 0.2450157
1: -0.0386566, 0.0201657, -0.0678644, 0.0706721, -0.1093287, 0.0880301
2: -0.0094249, 0.0668925, -0.0256436, 0.1290836, -0.1385084, 0.0925361
3: -0.0207658, 0.0590369, -0.0582062, 0.0917036, -0.1124694, 0.1172430
4: -0.0485350, 0.0167342, -0.1039154, 0.0532576, -0.1017926, 0.1206497
5: -0.0184907, 0.1372593, -0.0790586, 0.1891440, -0.2076347, 0.2163179
6: -0.0342200, 0.0356372, -0.0675413, 0.0707848, -0.1050048, 0.1031785
7: -0.0581318, 0.0318424, -0.1145203, 0.0618535, -0.1199854, 0.1463627
8: -0.0361145, 0.0561765, -0.0745057, 0.1090671, -0.1451817, 0.1306822
9: -0.0380013, 0.0421785, -0.0748748, 0.1224023, -0.1604036, 0.1170532

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_B1_A1_B2_B2_B2_A1

### Relational analysis result of NS_A1_B1_B1_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5287735, upper bound: 0.5045434
time: 1.64 seconds

## Relational analysis of NS_A1_B1_B1_A1_B2_B2_B2_A2

### Relational analysis result of NS_A1_B1_B1_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5227195, upper bound: 0.5044693
time: 2.99 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: 0.8535470, 1.0038627, 0.8342614, 1.0039622, -0.1504152, 0.1696014
1: -0.0351790, 0.0108801, -0.0397914, 0.0231497, -0.0583287, 0.0506715
2: -0.0084565, 0.0567094, -0.0085875, 0.0699734, -0.0784299, 0.0652969
3: -0.0160692, 0.0521470, -0.0225142, 0.0609668, -0.0770360, 0.0746612
4: -0.0403014, 0.0129818, -0.0514823, 0.0177854, -0.0580868, 0.0644640
5: -0.0116765, 0.1263161, -0.0209939, 0.1403246, -0.1520012, 0.1473099
6: -0.0291277, 0.0294998, -0.0357160, 0.0375745, -0.0667022, 0.0652158
7: -0.0486912, 0.0255127, -0.0612344, 0.0336156, -0.0823068, 0.0867471
8: -0.0313190, 0.0474300, -0.0380222, 0.0588181, -0.0901371, 0.0854522
9: -0.0302242, 0.0358409, -0.0401799, 0.0452755, -0.0754998, 0.0760208

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A1_B1_B2_A1_B1_B1_A1_A1

### Relational analysis result of NS_A1_B1_B2_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5421269, upper bound: 0.5395136
time: 1.80 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_B1_A1_A2

### Relational analysis result of NS_A1_B1_B2_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5484666, upper bound: 0.5395834
time: 1.88 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: 0.8351812, 1.0042753, 0.8316084, 1.0041152, -0.1689340, 0.1726670
1: -0.0395203, 0.0224993, -0.0406840, 0.0249293, -0.0644496, 0.0631833
2: -0.0090013, 0.0692681, -0.0087898, 0.0720416, -0.0810428, 0.0780579
3: -0.0220954, 0.0605462, -0.0237528, 0.0621180, -0.0842134, 0.0842990
4: -0.0507646, 0.0175562, -0.0534459, 0.0187540, -0.0695186, 0.0710021
5: -0.0204267, 0.1396565, -0.0228490, 0.1421529, -0.1625797, 0.1625056
6: -0.0353899, 0.0371199, -0.0367542, 0.0388182, -0.0742081, 0.0738741
7: -0.0605122, 0.0332291, -0.0632199, 0.0346730, -0.0951853, 0.0964490
8: -0.0375229, 0.0582423, -0.0393885, 0.0605428, -0.0980657, 0.0976308
9: -0.0397050, 0.0444399, -0.0414792, 0.0478552, -0.0875602, 0.0859190

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A1_B1_B2_A1_B1_B1_A2_A1

### Relational analysis result of NS_A1_B1_B2_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5417663, upper bound: 0.5395994
time: 2.09 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_B1_A2_A2

### Relational analysis result of NS_A1_B1_B2_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5483925, upper bound: 0.5396914
time: 1.76 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: 0.8531200, 1.0038657, 0.8212079, 1.0040420, -0.1509220, 0.1826578
1: -0.0352776, 0.0110133, -0.0446087, 0.0315343, -0.0668118, 0.0556220
2: -0.0084603, 0.0569979, -0.0104680, 0.0802779, -0.0887383, 0.0674659
3: -0.0161416, 0.0523423, -0.0287276, 0.0663899, -0.0825315, 0.0810699
4: -0.0404897, 0.0130881, -0.0607333, 0.0237361, -0.0642258, 0.0738214
5: -0.0116814, 0.1266263, -0.0309653, 0.1489381, -0.1606195, 0.1575916
6: -0.0292484, 0.0296737, -0.0411996, 0.0434339, -0.0726824, 0.0708733
7: -0.0489085, 0.0256921, -0.0706272, 0.0385976, -0.0875061, 0.0963192
8: -0.0314549, 0.0476128, -0.0444592, 0.0675493, -0.0990043, 0.0920720
9: -0.0304447, 0.0359400, -0.0463012, 0.0586192, -0.0890639, 0.0822413

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A1_B1_B2_A1_B1_B2_A1_B1

### Relational analysis result of NS_A1_B1_B2_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5488426, upper bound: 0.5425318
time: 1.91 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_B2_A1_B2

### Relational analysis result of NS_A1_B1_B2_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5488910, upper bound: 0.5486872
time: 1.84 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: 0.8346947, 1.0042777, 0.8184041, 1.0041929, -0.1694982, 0.1858736
1: -0.0396637, 0.0228432, -0.0456667, 0.0333148, -0.0729785, 0.0685099
2: -0.0090044, 0.0696411, -0.0111584, 0.0824985, -0.0915029, 0.0807995
3: -0.0223168, 0.0607687, -0.0300688, 0.0675416, -0.0898584, 0.0908374
4: -0.0511441, 0.0176774, -0.0626979, 0.0250792, -0.0762233, 0.0803753
5: -0.0207267, 0.1400099, -0.0331534, 0.1507673, -0.1714940, 0.1731632
6: -0.0355624, 0.0373603, -0.0423980, 0.0446782, -0.0802406, 0.0797584
7: -0.0608942, 0.0334334, -0.0726241, 0.0396558, -0.1005499, 0.1060575
8: -0.0377870, 0.0585468, -0.0458261, 0.0694382, -0.1072252, 0.1043730
9: -0.0399561, 0.0448818, -0.0476012, 0.0615211, -0.1014772, 0.0924830

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A1_B1_B2_A1_B1_B2_A2_B1

### Relational analysis result of NS_A1_B1_B2_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5487272, upper bound: 0.5426080
time: 2.21 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_B2_A2_B2

### Relational analysis result of NS_A1_B1_B2_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5487801, upper bound: 0.5487789
time: 2.00 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: 0.8145250, 1.0045193, 0.6882526, 1.0218822, -0.2073572, 0.3162668
1: -0.0438425, 0.0312940, -0.1069111, 0.1213959, -0.1652384, 0.1382051
2: -0.0100873, 0.0795252, -0.0598213, 0.1889108, -0.1989981, 0.1393465
3: -0.0287216, 0.0660576, -0.0798449, 0.1410998, -0.1698214, 0.1459025
4: -0.0597177, 0.0229636, -0.1332126, 0.1014422, -0.1611599, 0.1561763
5: -0.0307217, 0.1484794, -0.1362083, 0.2203353, -0.2510570, 0.2846876
6: -0.0404361, 0.0444607, -0.0979205, 0.1076142, -0.1480503, 0.1423812
7: -0.0691237, 0.0414135, -0.1594229, 0.1326847, -0.2018084, 0.2008364
8: -0.0435542, 0.0673134, -0.1342014, 0.1299036, -0.1734578, 0.2015148
9: -0.0456899, 0.0569149, -0.1149814, 0.1787084, -0.2243983, 0.1718963

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_B2_A1_B2_B1_B1_A1

### Relational analysis result of NS_A1_B1_B2_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5293146, upper bound: 0.5096467
time: 1.46 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2_B1_B1_A2

### Relational analysis result of NS_A1_B1_B2_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5230343, upper bound: 0.5095323
time: 5.81 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: 0.8078992, 1.0046948, 0.6653655, 1.0251987, -0.2172995, 0.3393294
1: -0.0449542, 0.0335884, -0.1182197, 0.1365639, -0.1815180, 0.1518081
2: -0.0108622, 0.0822025, -0.0688093, 0.2083386, -0.2192007, 0.1510118
3: -0.0305303, 0.0674718, -0.0881623, 0.1543170, -0.1848472, 0.1556341
4: -0.0619460, 0.0244529, -0.1448052, 0.1154402, -0.1773862, 0.1692581
5: -0.0335666, 0.1507553, -0.1544885, 0.2317422, -0.2653087, 0.3052437
6: -0.0417370, 0.0466151, -0.1074000, 0.1189061, -0.1606431, 0.1540151
7: -0.0711985, 0.0441066, -0.1741630, 0.1509868, -0.2221853, 0.2182697
8: -0.0450244, 0.0697606, -0.1505165, 0.1395870, -0.1846114, 0.2202770
9: -0.0471887, 0.0601256, -0.1273372, 0.1984169, -0.2456057, 0.1874628

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_B2_A1_B2_B1_B2_A1

### Relational analysis result of NS_A1_B1_B2_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5292257, upper bound: 0.5086044
time: 1.84 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2_B1_B2_A2

### Relational analysis result of NS_A1_B1_B2_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5230221, upper bound: 0.5085023
time: 1.64 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: 0.8427994, 1.0044447, 0.7039602, 1.0196058, -0.1768063, 0.3004845
1: -0.0376599, 0.0171126, -0.0991498, 0.1109860, -0.1486460, 0.1162624
2: -0.0092248, 0.0639740, -0.0536528, 0.1755773, -0.1848022, 0.1176268
3: -0.0192369, 0.0570622, -0.0741367, 0.1320287, -0.1512655, 0.1311989
4: -0.0460395, 0.0156587, -0.1252564, 0.0918352, -0.1378748, 0.1409151
5: -0.0160782, 0.1341228, -0.1236624, 0.2125066, -0.2285849, 0.2577852
6: -0.0326893, 0.0338782, -0.0914146, 0.0998645, -0.1325538, 0.1252928
7: -0.0552745, 0.0300283, -0.1493065, 0.1201238, -0.1753983, 0.1793348
8: -0.0347401, 0.0534736, -0.1230042, 0.1232578, -0.1579979, 0.1764778
9: -0.0357724, 0.0401195, -0.1065015, 0.1651822, -0.2009546, 0.1466210

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_B2_A1_B2_B2_B1_A1

### Relational analysis result of NS_A1_B1_B2_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5286197, upper bound: 0.5054422
time: 1.58 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2_B2_B1_A2

### Relational analysis result of NS_A1_B1_B2_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5225080, upper bound: 0.5053666
time: 1.49 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: 0.8401427, 1.0045960, 0.6809356, 1.0229422, -0.1827995, 0.3236604
1: -0.0382732, 0.0189911, -0.1105264, 0.1262450, -0.1645182, 0.1295175
2: -0.0094244, 0.0657697, -0.0626948, 0.1951218, -0.2045463, 0.1284645
3: -0.0201776, 0.0582772, -0.0825040, 0.1453253, -0.1655029, 0.1407811
4: -0.0475749, 0.0163205, -0.1369187, 0.1059173, -0.1534922, 0.1532392
5: -0.0175627, 0.1360527, -0.1420524, 0.2239820, -0.2415446, 0.2781051
6: -0.0336311, 0.0349605, -0.1009510, 0.1112242, -0.1448553, 0.1359116
7: -0.0570326, 0.0311445, -0.1641353, 0.1385359, -0.1955684, 0.1952798
8: -0.0355858, 0.0551367, -0.1394172, 0.1329994, -0.1685852, 0.1945539
9: -0.0371438, 0.0413864, -0.1189315, 0.1850092, -0.2221530, 0.1603179

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_B2_A1_B2_B2_B2_A1

### Relational analysis result of NS_A1_B1_B2_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5285885, upper bound: 0.5042842
time: 1.73 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2_B2_B2_A2

### Relational analysis result of NS_A1_B1_B2_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5225046, upper bound: 0.5042034
time: 1.71 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: 0.9111160, 1.0035716, 0.6947390, 1.0236694, -0.1125534, 0.3088326
1: -0.0197777, -0.0003740, -0.1127762, 0.1102922, -0.1300699, 0.1124022
2: -0.0080721, 0.0258818, -0.0585202, 0.1795233, -0.1875954, 0.0844020
3: -0.0095315, 0.0216706, -0.0769414, 0.1317060, -0.1412374, 0.0986120
4: -0.0135931, 0.0065991, -0.1365252, 0.0911949, -0.1047880, 0.1431243
5: -0.0111932, 0.0788858, -0.1266090, 0.2169966, -0.2281897, 0.2054949
6: -0.0136158, 0.0121312, -0.0921870, 0.1196562, -0.1332719, 0.1043183
7: -0.0263265, 0.0081993, -0.1530077, 0.1218665, -0.1481930, 0.1612070
8: -0.0140886, 0.0245669, -0.1262437, 0.1397621, -0.1538508, 0.1508107
9: -0.0073692, 0.0205788, -0.1100264, 0.1684562, -0.1758253, 0.1306052

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 242

### Candidate
type: B, layer: 1, pos: 242

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of NS_A1_B2_A1_A2_B1_A1_A1_B1

### Relational analysis result of NS_A1_B2_A1_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5266434, upper bound: 0.5024419
time: 1.75 seconds

## Relational analysis of NS_A1_B2_A1_A2_B1_A1_A1_B2

### Relational analysis result of NS_A1_B2_A1_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5266434, upper bound: 0.5024419
time: 1.72 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: 0.8994395, 1.0039955, 0.6879048, 1.0248806, -0.1254411, 0.3160908
1: -0.0233873, -0.0002684, -0.1170172, 0.1141203, -0.1375076, 0.1167489
2: -0.0086319, 0.0282476, -0.0612749, 0.1849173, -0.1935492, 0.0895225
3: -0.0097514, 0.0288080, -0.0793643, 0.1350677, -0.1448192, 0.1081723
4: -0.0178744, 0.0069569, -0.1407627, 0.0947279, -0.1126023, 0.1477196
5: -0.0118972, 0.0892505, -0.1315609, 0.2205946, -0.2324918, 0.2208115
6: -0.0159549, 0.0151823, -0.0946866, 0.1244894, -0.1404443, 0.1098689
7: -0.0297483, 0.0086617, -0.1572881, 0.1267146, -0.1564629, 0.1659498
8: -0.0167734, 0.0293319, -0.1307151, 0.1439673, -0.1607407, 0.1600471
9: -0.0087631, 0.0240834, -0.1136812, 0.1738009, -0.1825640, 0.1377646

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 242

### Candidate
type: B, layer: 1, pos: 242

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of NS_A1_B2_A1_A2_B1_A1_A2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5266546, upper bound: 0.5025889
time: 1.75 seconds

## Relational analysis of NS_A1_B2_A1_A2_B1_A1_A2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5266546, upper bound: 0.5025889
time: 1.81 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: 0.8504339, 1.0032890, 0.7080527, 1.0213101, -0.1708761, 0.2952363
1: -0.0369974, 0.0093346, -0.1045149, 0.1028345, -0.1398319, 0.1138495
2: -0.0076988, 0.0590287, -0.0531538, 0.1690157, -0.1767145, 0.1121825
3: -0.0153813, 0.0557196, -0.0722213, 0.1251570, -0.1405383, 0.1279409
4: -0.0437761, 0.0088516, -0.1282706, 0.0843126, -0.1280887, 0.1371222
5: -0.0107237, 0.1320382, -0.1169626, 0.2099875, -0.2207112, 0.2490007
6: -0.0299941, 0.0266866, -0.0873179, 0.1102409, -0.1402349, 0.1140045
7: -0.0466836, 0.0288225, -0.1446691, 0.1124221, -0.1591057, 0.1734916
8: -0.0321943, 0.0472985, -0.1175333, 0.1315707, -0.1637650, 0.1648318
9: -0.0292908, 0.0376704, -0.1029068, 0.1580445, -0.1873353, 0.1405771

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 242

### Candidate
type: B, layer: 1, pos: 242

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of NS_A1_B2_A1_A2_B1_A2_A1_B1

### Relational analysis result of NS_A1_B2_A1_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5198610, upper bound: 0.5024076
time: 2.68 seconds

## Relational analysis of NS_A1_B2_A1_A2_B1_A2_A1_B2

### Relational analysis result of NS_A1_B2_A1_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5198610, upper bound: 0.5024076
time: 1.80 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: 0.8395494, 1.0036439, 0.7012402, 1.0225174, -0.1829680, 0.3024037
1: -0.0400170, 0.0118499, -0.1087423, 0.1066505, -0.1466676, 0.1205922
2: -0.0081675, 0.0664850, -0.0558998, 0.1743923, -0.1825598, 0.1223848
3: -0.0166676, 0.0616904, -0.0746365, 0.1285081, -0.1451756, 0.1363269
4: -0.0495460, 0.0093063, -0.1324945, 0.0878342, -0.1373802, 0.1418008
5: -0.0128902, 0.1415401, -0.1218986, 0.2135741, -0.2264642, 0.2634387
6: -0.0331637, 0.0292390, -0.0898094, 0.1150587, -0.1482224, 0.1190484
7: -0.0506948, 0.0343185, -0.1489359, 0.1172548, -0.1679496, 0.1832544
8: -0.0357246, 0.0512846, -0.1219904, 0.1357622, -0.1714868, 0.1732750
9: -0.0338689, 0.0407083, -0.1065499, 0.1633722, -0.1972412, 0.1472582

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 242

### Candidate
type: B, layer: 1, pos: 242

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of NS_A1_B2_A1_A2_B1_A2_A2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5199137, upper bound: 0.5025229
time: 1.50 seconds

## Relational analysis of NS_A1_B2_A1_A2_B1_A2_A2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5199137, upper bound: 0.5025229
time: 1.73 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: 0.8913522, 1.0043845, 0.5613515, 1.0473058, -0.1559536, 0.4430330
1: -0.0256457, -0.0000920, -0.1955470, 0.1850078, -0.2106535, 0.1954551
2: -0.0091453, 0.0316687, -0.1122849, 0.2847985, -0.2939437, 0.1439536
3: -0.0105759, 0.0332736, -0.1242308, 0.1973195, -0.2078954, 0.1575044
4: -0.0220851, 0.0071807, -0.2192287, 0.1601476, -0.1822327, 0.2264095
5: -0.0125430, 0.0963175, -0.2232556, 0.2872197, -0.2997627, 0.3195731
6: -0.0181277, 0.0170913, -0.1409705, 0.2139880, -0.2321157, 0.1580618
7: -0.0318891, 0.0090857, -0.2365492, 0.2164882, -0.2483773, 0.2456350
8: -0.0190352, 0.0323132, -0.2135125, 0.2218326, -0.2408678, 0.2458257
9: -0.0120799, 0.0262760, -0.1813567, 0.2727695, -0.2848494, 0.2076328

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 242

### Candidate
type: A, layer: 1, pos: 242

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B1_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5201075, upper bound: 0.5024261
time: 1.81 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B1_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5201075, upper bound: 0.5024261
time: 1.74 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: 0.8883783, 1.0045167, 0.5439715, 1.0503855, -0.1620072, 0.4605452
1: -0.0264708, 0.0005657, -0.2063318, 0.1947433, -0.2212140, 0.2068976
2: -0.0093199, 0.0330352, -0.1192904, 0.2985155, -0.3078355, 0.1523256
3: -0.0108973, 0.0349050, -0.1303927, 0.2058687, -0.2167660, 0.1652976
4: -0.0236616, 0.0072664, -0.2300048, 0.1691320, -0.1927936, 0.2372712
5: -0.0127626, 0.0989136, -0.2358486, 0.2963696, -0.3091322, 0.3347622
6: -0.0189442, 0.0177887, -0.1473269, 0.2262791, -0.2452233, 0.1651156
7: -0.0327002, 0.0096626, -0.2474346, 0.2288173, -0.2615175, 0.2570972
8: -0.0198875, 0.0334024, -0.2248835, 0.2325262, -0.2524136, 0.2582859
9: -0.0133308, 0.0270797, -0.1906509, 0.2863615, -0.2996923, 0.2177306

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 242

### Candidate
type: A, layer: 1, pos: 242

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5201711, upper bound: 0.5025240
time: 1.77 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5201711, upper bound: 0.5025229
time: 1.73 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: 0.8851993, 1.0036922, 0.6975579, 1.0231699, -0.1379706, 0.3061342
1: -0.0273527, 0.0013004, -0.1110272, 0.1087131, -0.1360658, 0.1123276
2: -0.0082311, 0.0352130, -0.0573840, 0.1772986, -0.1855297, 0.0925970
3: -0.0112730, 0.0366488, -0.0759420, 0.1303194, -0.1415924, 0.1125908
4: -0.0253469, 0.0073992, -0.1347775, 0.0897378, -0.1150846, 0.1421767
5: -0.0113931, 0.1016889, -0.1245667, 0.2155126, -0.2269057, 0.2262555
6: -0.0198700, 0.0185342, -0.0911561, 0.1176628, -0.1375328, 0.1096903
7: -0.0338718, 0.0112678, -0.1512422, 0.1198669, -0.1537387, 0.1625100
8: -0.0209185, 0.0345666, -0.1243996, 0.1380279, -0.1589464, 0.1589662
9: -0.0146679, 0.0279670, -0.1085190, 0.1662519, -0.1809198, 0.1364861

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 120

### Candidate
type: B, layer: 1, pos: 120

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A2_A2_B1_A1_A1_B1

### Relational analysis result of NS_A1_B2_A2_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5244307, upper bound: 0.5050311
time: 1.78 seconds

## Relational analysis of NS_A1_B2_A2_A2_B1_A1_A1_B2

### Relational analysis result of NS_A1_B2_A2_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5244307, upper bound: 0.5050311
time: 1.68 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: 0.8745394, 1.0040741, 0.6907549, 1.0243753, -0.1498359, 0.3133192
1: -0.0303100, 0.0043044, -0.1152486, 0.1125237, -0.1428337, 0.1195530
2: -0.0087355, 0.0425155, -0.0601261, 0.1826678, -0.1914033, 0.1026416
3: -0.0125327, 0.0425006, -0.0783538, 0.1336658, -0.1461985, 0.1208544
4: -0.0309977, 0.0078550, -0.1389955, 0.0932544, -0.1242521, 0.1468506
5: -0.0120276, 0.1109947, -0.1294958, 0.2190941, -0.2311217, 0.2404904
6: -0.0231651, 0.0210339, -0.0936441, 0.1224738, -0.1456388, 0.1146780
7: -0.0380477, 0.0166505, -0.1555029, 0.1246928, -0.1627405, 0.1721534
8: -0.0246049, 0.0384705, -0.1288504, 0.1422135, -0.1668184, 0.1673209
9: -0.0194033, 0.0309423, -0.1121570, 0.1715720, -0.1909753, 0.1430992

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 120

### Candidate
type: B, layer: 1, pos: 120

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A2_A2_B1_A1_A2_B1

### Relational analysis result of NS_A1_B2_A2_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5264119, upper bound: 0.5054369
time: 1.78 seconds

## Relational analysis of NS_A1_B2_A2_A2_B1_A1_A2_B2

### Relational analysis result of NS_A1_B2_A2_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5264119, upper bound: 0.5054369
time: 1.99 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: 0.8830469, 1.0038542, 0.6648651, 1.0289631, -0.1459162, 0.3389890
1: -0.0279498, 0.0017978, -0.1313140, 0.1270258, -0.1549756, 0.1331118
2: -0.0084449, 0.0366874, -0.0705616, 0.2031011, -0.2115460, 0.1072489
3: -0.0115274, 0.0378295, -0.0875325, 0.1464010, -0.1579284, 0.1253620
4: -0.0264878, 0.0074891, -0.1550479, 0.1066378, -0.1331256, 0.1625371
5: -0.0116621, 0.1035678, -0.1482544, 0.2327240, -0.2443861, 0.2518222
6: -0.0204967, 0.0190389, -0.1031128, 0.1407832, -0.1612799, 0.1221517
7: -0.0346649, 0.0123546, -0.1717180, 0.1430582, -0.1777232, 0.1840726
8: -0.0216166, 0.0353548, -0.1457888, 0.1581430, -0.1797597, 0.1811437
9: -0.0155733, 0.0285677, -0.1260018, 0.1918187, -0.2073919, 0.1545695

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 120

### Candidate
type: A, layer: 1, pos: 120

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A2_A2_B2_A1_A1_B1

### Relational analysis result of NS_A1_B2_A2_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5245326, upper bound: 0.5047668
time: 1.63 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2_A1_A1_B2

### Relational analysis result of NS_A1_B2_A2_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5245326, upper bound: 0.5047668
time: 1.79 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: 0.8720938, 1.0042260, 0.6579654, 1.0301858, -0.1580920, 0.3462606
1: -0.0308978, 0.0050954, -0.1355954, 0.1308905, -0.1617883, 0.1406908
2: -0.0089359, 0.0441730, -0.0733426, 0.2085467, -0.2174826, 0.1175156
3: -0.0129220, 0.0436651, -0.0899787, 0.1497950, -0.1627170, 0.1336437
4: -0.0321208, 0.0083622, -0.1593258, 0.1102046, -0.1423254, 0.1676880
5: -0.0122796, 0.1128442, -0.1532537, 0.2363565, -0.2486361, 0.2660978
6: -0.0238848, 0.0219440, -0.1056362, 0.1456625, -0.1695474, 0.1275802
7: -0.0392535, 0.0177203, -0.1760392, 0.1479529, -0.1872064, 0.1937595
8: -0.0254153, 0.0394869, -0.1503030, 0.1623881, -0.1878035, 0.1897899
9: -0.0206500, 0.0315336, -0.1296915, 0.1972145, -0.2178645, 0.1612251

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 120

### Candidate
type: B, layer: 1, pos: 120

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A2_A2_B2_A1_A2_B1

### Relational analysis result of NS_A1_B2_A2_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5264645, upper bound: 0.5052095
time: 1.78 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2_A1_A2_B2

### Relational analysis result of NS_A1_B2_A2_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5264645, upper bound: 0.5052095
time: 1.74 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 4.38 seconds
NS_A1_B1_B1_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5421258, upper bound: 0.5395136
NS_A1_B1_B1_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5484777, upper bound: 0.5395845
NS_A1_B1_B1_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5417619, upper bound: 0.5395983
NS_A1_B1_B1_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5484013, upper bound: 0.5396925
NS_A1_B1_B1_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5429514, upper bound: 0.5486487
NS_A1_B1_B1_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5488637, upper bound: 0.5486884
NS_A1_B1_B1_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5425479, upper bound: 0.5487261
NS_A1_B1_B1_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5487496, upper bound: 0.5487801
NS_A1_B1_B1_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5296526, upper bound: 0.5094137
NS_A1_B1_B1_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5235425, upper bound: 0.5093117
NS_A1_B1_B1_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5294882, upper bound: 0.5094831
NS_A1_B1_B1_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5234199, upper bound: 0.5093601
NS_A1_B1_B1_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5288378, upper bound: 0.5058712
NS_A1_B1_B1_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5227326, upper bound: 0.5057898
NS_A1_B1_B1_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5287735, upper bound: 0.5045434
NS_A1_B1_B1_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5227195, upper bound: 0.5044693
NS_A1_B1_B2_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5421269, upper bound: 0.5395136
NS_A1_B1_B2_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5484666, upper bound: 0.5395834
NS_A1_B1_B2_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5417663, upper bound: 0.5395994
NS_A1_B1_B2_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5483925, upper bound: 0.5396914
NS_A1_B1_B2_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5488426, upper bound: 0.5425318
NS_A1_B1_B2_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5488910, upper bound: 0.5486872
NS_A1_B1_B2_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5487272, upper bound: 0.5426080
NS_A1_B1_B2_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5487801, upper bound: 0.5487789
NS_A1_B1_B2_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5293146, upper bound: 0.5096467
NS_A1_B1_B2_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5230343, upper bound: 0.5095323
NS_A1_B1_B2_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5292257, upper bound: 0.5086044
NS_A1_B1_B2_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5230221, upper bound: 0.5085023
NS_A1_B1_B2_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5286197, upper bound: 0.5054422
NS_A1_B1_B2_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5225080, upper bound: 0.5053666
NS_A1_B1_B2_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5285885, upper bound: 0.5042842
NS_A1_B1_B2_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5225046, upper bound: 0.5042034
NS_A1_B2_A1_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5266434, upper bound: 0.5024419
NS_A1_B2_A1_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5266434, upper bound: 0.5024419
NS_A1_B2_A1_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5266546, upper bound: 0.5025889
NS_A1_B2_A1_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5266546, upper bound: 0.5025889
NS_A1_B2_A1_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5198610, upper bound: 0.5024076
NS_A1_B2_A1_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5198610, upper bound: 0.5024076
NS_A1_B2_A1_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5199137, upper bound: 0.5025229
NS_A1_B2_A1_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5199137, upper bound: 0.5025229
NS_A1_B2_A1_A2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5201075, upper bound: 0.5024261
NS_A1_B2_A1_A2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5201075, upper bound: 0.5024261
NS_A1_B2_A1_A2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5201711, upper bound: 0.5025240
NS_A1_B2_A1_A2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5201711, upper bound: 0.5025229
NS_A1_B2_A2_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5244307, upper bound: 0.5050311
NS_A1_B2_A2_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5244307, upper bound: 0.5050311
NS_A1_B2_A2_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5264119, upper bound: 0.5054369
NS_A1_B2_A2_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5264119, upper bound: 0.5054369
NS_A1_B2_A2_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5245326, upper bound: 0.5047668
NS_A1_B2_A2_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5245326, upper bound: 0.5047668
NS_A1_B2_A2_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5264645, upper bound: 0.5052095
NS_A1_B2_A2_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -0.5264645, upper bound: 0.5052095

## BFS NS instance: NS_A1_B1_B1_A1_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: 0.8980203, 1.0038106, 0.8757658, 1.0037787, -0.1057584, 0.1280448
1: -0.0237959, -0.0003144, -0.0299697, 0.0038523, -0.0276481, 0.0296553
2: -0.0083877, 0.0287579, -0.0083454, 0.0416752, -0.0500629, 0.0371033
3: -0.0098621, 0.0296158, -0.0123878, 0.0418264, -0.0516885, 0.0420036
4: -0.0185504, 0.0069974, -0.0303475, 0.0078005, -0.0263509, 0.0373448
5: -0.0115901, 0.0904964, -0.0115369, 0.1099239, -0.1215140, 0.1020333
6: -0.0163083, 0.0155276, -0.0227483, 0.0207463, -0.0370545, 0.0382759
7: -0.0301356, 0.0084600, -0.0375184, 0.0160311, -0.0461667, 0.0459784
8: -0.0171500, 0.0298713, -0.0241356, 0.0380213, -0.0551713, 0.0540069
9: -0.0092752, 0.0244801, -0.0188089, 0.0305999, -0.0398751, 0.0432889

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_B1_A1_B1_B1_A1_A1_A1

### Relational analysis result of NS_A1_B1_B1_A1_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5329381, upper bound: 0.5235796
time: 1.96 seconds

## Relational analysis of NS_A1_B1_B1_A1_B1_B1_A1_A1_A2

### Relational analysis result of NS_A1_B1_B1_A1_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5210414, upper bound: 0.5235063
time: 2.04 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: 0.9011835, 1.0042750, 0.8910927, 1.0037546, -0.1025711, 0.1131823
1: -0.0228481, -0.0001988, -0.0257177, -0.0000348, -0.0228133, 0.0255190
2: -0.0090008, 0.0278942, -0.0083137, 0.0317820, -0.0407828, 0.0362079
3: -0.0097186, 0.0277419, -0.0106037, 0.0334160, -0.0431346, 0.0383456
4: -0.0172349, 0.0069034, -0.0222228, 0.0071878, -0.0244227, 0.0291262
5: -0.0123612, 0.0877024, -0.0114970, 0.0965441, -0.1089053, 0.0991995
6: -0.0156055, 0.0147266, -0.0181985, 0.0171522, -0.0327577, 0.0329251
7: -0.0292371, 0.0089664, -0.0319574, 0.0083989, -0.0376360, 0.0409237
8: -0.0163724, 0.0286201, -0.0191086, 0.0324083, -0.0487807, 0.0477288
9: -0.0083461, 0.0235599, -0.0121891, 0.0263459, -0.0346920, 0.0357490

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_B1_A1_B1_B1_A1_A2_A1

### Relational analysis result of NS_A1_B1_B1_A1_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5405819, upper bound: 0.5236064
time: 2.03 seconds

## Relational analysis of NS_A1_B1_B1_A1_B1_B1_A1_A2_A2

### Relational analysis result of NS_A1_B1_B1_A1_B1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5271170, upper bound: 0.5235426
time: 1.73 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: 0.8819906, 1.0042106, 0.8736597, 1.0039250, -0.1219344, 0.1305509
1: -0.0282429, 0.0020419, -0.0305363, 0.0046069, -0.0328498, 0.0325782
2: -0.0089157, 0.0374111, -0.0085385, 0.0431145, -0.0520302, 0.0459496
3: -0.0116522, 0.0384090, -0.0126563, 0.0429489, -0.0546011, 0.0510653
4: -0.0270478, 0.0075333, -0.0314301, 0.0079721, -0.0350199, 0.0389633
5: -0.0122542, 0.1044900, -0.0117798, 0.1117067, -0.1239609, 0.1162698
6: -0.0208044, 0.0192866, -0.0234421, 0.0213060, -0.0421105, 0.0427287
7: -0.0350543, 0.0128880, -0.0384566, 0.0170623, -0.0521166, 0.0513447
8: -0.0219592, 0.0357417, -0.0249169, 0.0388162, -0.0607755, 0.0606586
9: -0.0160175, 0.0288626, -0.0198416, 0.0311699, -0.0471875, 0.0487042

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_B1_A1_B1_B1_A2_A1_A1

### Relational analysis result of NS_A1_B1_B1_A1_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5328711, upper bound: 0.5236712
time: 2.88 seconds

## Relational analysis of NS_A1_B1_B1_A1_B1_B1_A2_A1_A2

### Relational analysis result of NS_A1_B1_B1_A1_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5209941, upper bound: 0.5235869
time: 1.86 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: 0.8879350, 1.0046759, 0.8890359, 1.0039010, -0.1159660, 0.1156399
1: -0.0265937, 0.0006682, -0.0262883, 0.0004180, -0.0270117, 0.0269565
2: -0.0095300, 0.0333389, -0.0085069, 0.0326798, -0.0422098, 0.0418458
3: -0.0109497, 0.0351481, -0.0108239, 0.0345441, -0.0454939, 0.0459720
4: -0.0238966, 0.0072849, -0.0233130, 0.0072444, -0.0311410, 0.0305979
5: -0.0130268, 0.0993006, -0.0117400, 0.0983395, -0.1113662, 0.1110406
6: -0.0190733, 0.0178927, -0.0187597, 0.0176344, -0.0367078, 0.0366523
7: -0.0328636, 0.0098864, -0.0324982, 0.0093305, -0.0421941, 0.0423847
8: -0.0200312, 0.0335647, -0.0196901, 0.0331615, -0.0531927, 0.0532548
9: -0.0135172, 0.0272034, -0.0130541, 0.0268999, -0.0404171, 0.0402576

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_B1_A1_B1_B1_A2_A2_A1

### Relational analysis result of NS_A1_B1_B1_A1_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5405819, upper bound: 0.5237064
time: 1.92 seconds

## Relational analysis of NS_A1_B1_B1_A1_B1_B1_A2_A2_A2

### Relational analysis result of NS_A1_B1_B1_A1_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5271170, upper bound: 0.5236311
time: 2.03 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: 0.8976994, 1.0038136, 0.8641896, 1.0038574, -0.1061580, 0.1396240
1: -0.0238849, -0.0003137, -0.0327223, 0.0075607, -0.0314456, 0.0324086
2: -0.0083916, 0.0288980, -0.0084491, 0.0495156, -0.0579072, 0.0373472
3: -0.0098965, 0.0297918, -0.0142632, 0.0472798, -0.0571763, 0.0440551
4: -0.0187205, 0.0070062, -0.0356071, 0.0103309, -0.0290514, 0.0426133
5: -0.0115950, 0.0907765, -0.0116674, 0.1185854, -0.1301804, 0.1024439
6: -0.0163958, 0.0156029, -0.0261192, 0.0251640, -0.0415598, 0.0417221
7: -0.0302199, 0.0084632, -0.0432756, 0.0210411, -0.0512611, 0.0517388
8: -0.0172407, 0.0299888, -0.0279313, 0.0428720, -0.0601127, 0.0579201
9: -0.0094102, 0.0245664, -0.0247302, 0.0333692, -0.0427794, 0.0492967

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_B1_A1_B1_B2_A1_A1_A1

### Relational analysis result of NS_A1_B1_B1_A1_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5340457, upper bound: 0.5271642
time: 1.78 seconds

## Relational analysis of NS_A1_B1_B1_A1_B1_B2_A1_A1_A2

### Relational analysis result of NS_A1_B1_B1_A1_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5211257, upper bound: 0.5270947
time: 1.88 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: 0.9007396, 1.0042781, 0.8813615, 1.0038342, -0.1030946, 0.1229166
1: -0.0229854, -0.0001980, -0.0284174, 0.0021873, -0.0251727, 0.0282194
2: -0.0090049, 0.0279842, -0.0084187, 0.0378420, -0.0468469, 0.0364030
3: -0.0097269, 0.0280134, -0.0117265, 0.0387541, -0.0484810, 0.0397399
4: -0.0173978, 0.0069170, -0.0273813, 0.0075595, -0.0249573, 0.0342983
5: -0.0123664, 0.0880966, -0.0116291, 0.1050391, -0.1174055, 0.0997257
6: -0.0156945, 0.0148426, -0.0209876, 0.0194341, -0.0351285, 0.0358302
7: -0.0293673, 0.0089698, -0.0352861, 0.0132057, -0.0425730, 0.0442559
8: -0.0164745, 0.0288014, -0.0221633, 0.0359721, -0.0524466, 0.0509647
9: -0.0084523, 0.0236932, -0.0162821, 0.0290382, -0.0374905, 0.0399753

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_B1_A1_B1_B2_A1_A2_A1

### Relational analysis result of NS_A1_B1_B1_A1_B1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5414870, upper bound: 0.5271837
time: 1.84 seconds

## Relational analysis of NS_A1_B1_B1_A1_B1_B2_A1_A2_A2

### Relational analysis result of NS_A1_B1_B1_A1_B1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5271750, upper bound: 0.5271187
time: 1.76 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: 0.8817098, 1.0042131, 0.8617783, 1.0040003, -0.1222906, 0.1424348
1: -0.0283207, 0.0021068, -0.0332789, 0.0083128, -0.0366335, 0.0353858
2: -0.0089190, 0.0376034, -0.0086380, 0.0511456, -0.0600645, 0.0462414
3: -0.0116854, 0.0385630, -0.0146724, 0.0483826, -0.0600680, 0.0532354
4: -0.0271966, 0.0075450, -0.0366708, 0.0109315, -0.0381281, 0.0442158
5: -0.0122583, 0.1047351, -0.0119050, 0.1203371, -0.1325954, 0.1166401
6: -0.0208861, 0.0193524, -0.0268009, 0.0261464, -0.0470325, 0.0461533
7: -0.0351577, 0.0130298, -0.0445026, 0.0220544, -0.0572120, 0.0575325
8: -0.0220503, 0.0358445, -0.0286989, 0.0439047, -0.0659551, 0.0645434
9: -0.0161357, 0.0289409, -0.0259751, 0.0339293, -0.0500650, 0.0549160

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_B1_A1_B1_B2_A2_A1_A1

### Relational analysis result of NS_A1_B1_B1_A1_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5338835, upper bound: 0.5272606
time: 1.90 seconds

## Relational analysis of NS_A1_B1_B1_A1_B1_B2_A2_A1_A2

### Relational analysis result of NS_A1_B1_B1_A1_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5210890, upper bound: 0.5271783
time: 1.84 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: 0.8875599, 1.0046785, 0.8793467, 1.0039773, -0.1164174, 0.1253318
1: -0.0266978, 0.0007549, -0.0289763, 0.0026529, -0.0293507, 0.0297312
2: -0.0095334, 0.0335959, -0.0086076, 0.0392222, -0.0487556, 0.0422034
3: -0.0109941, 0.0353539, -0.0119646, 0.0398592, -0.0508533, 0.0473185
4: -0.0240955, 0.0073006, -0.0284493, 0.0076437, -0.0317392, 0.0357499
5: -0.0130311, 0.0996281, -0.0118666, 0.1067979, -0.1198290, 0.1114947
6: -0.0191826, 0.0179806, -0.0215743, 0.0199066, -0.0390891, 0.0395549
7: -0.0330019, 0.0100758, -0.0360285, 0.0142230, -0.0472249, 0.0461044
8: -0.0201529, 0.0337021, -0.0228167, 0.0367099, -0.0568628, 0.0565188
9: -0.0136750, 0.0273082, -0.0171296, 0.0296005, -0.0432755, 0.0444377

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_B1_A1_B1_B2_A2_A2_A1

### Relational analysis result of NS_A1_B1_B1_A1_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5414757, upper bound: 0.5272835
time: 1.83 seconds

## Relational analysis of NS_A1_B1_B1_A1_B1_B2_A2_A2_A2

### Relational analysis result of NS_A1_B1_B1_A1_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5271750, upper bound: 0.5272087
time: 1.81 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: 0.8548794, 1.0034195, 0.7482457, 1.0177448, -0.1628654, 0.2551739
1: -0.0348714, 0.0104646, -0.0721417, 0.0778705, -0.1127419, 0.0826063
2: -0.0078712, 0.0558088, -0.0284348, 0.1380601, -0.1459312, 0.0842435
3: -0.0158431, 0.0515377, -0.0636280, 0.0963595, -0.1122025, 0.1151656
4: -0.0397137, 0.0126498, -0.1118577, 0.0586873, -0.0984011, 0.1245075
5: -0.0109404, 0.1253483, -0.0879042, 0.1965388, -0.2074792, 0.2132525
6: -0.0287511, 0.0289569, -0.0723862, 0.0758153, -0.1045664, 0.1013430
7: -0.0480132, 0.0249529, -0.1225931, 0.0661309, -0.1141440, 0.1475461
8: -0.0308949, 0.0468593, -0.0800320, 0.1167032, -0.1475981, 0.1268913
9: -0.0295364, 0.0355315, -0.0801301, 0.1341336, -0.1636700, 0.1156616

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of NS_A1_B1_B1_A1_B2_B1_A1_A1_A1

### Relational analysis result of NS_A1_B1_B1_A1_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5296526, upper bound: 0.5094084
time: 1.87 seconds

## Relational analysis of NS_A1_B1_B1_A1_B2_B1_A1_A1_A2

### Relational analysis result of NS_A1_B1_B1_A1_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5296526, upper bound: 0.5094137
time: 1.92 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: 0.7778479, 1.0066890, 0.7547399, 1.0161773, -0.2383294, 0.2519491
1: -0.0526530, 0.0630386, -0.0696911, 0.0737464, -0.1263994, 0.1327296
2: -0.0226838, 0.1078765, -0.0268357, 0.1329171, -0.1556009, 0.1347122
3: -0.0422352, 0.0867664, -0.0605217, 0.0936920, -0.1359272, 0.1472881
4: -0.0835778, 0.0318367, -0.1073073, 0.0555764, -0.1391542, 0.1391440
5: -0.0523660, 0.1813021, -0.0828363, 0.1923021, -0.2446681, 0.2641384
6: -0.0557142, 0.0603386, -0.0696104, 0.0729332, -0.1286474, 0.1299490
7: -0.0982558, 0.0573176, -0.1179680, 0.0636802, -0.1619360, 0.1752856
8: -0.0554150, 0.0941307, -0.0768658, 0.1123283, -0.1677433, 0.1709965
9: -0.0693017, 0.0710903, -0.0771191, 0.1274124, -0.1967141, 0.1482094

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 251

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A1_B1_B1_A1_B2_B1_A1_A2_A1

### Relational analysis result of NS_A1_B1_B1_A1_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5213535, upper bound: 0.5093117
time: 2.04 seconds

## Relational analysis of NS_A1_B1_B1_A1_B2_B1_A1_A2_A2

### Relational analysis result of NS_A1_B1_B1_A1_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5213535, upper bound: 0.5093106
time: 1.72 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: 0.8359944, 1.0038214, 0.7456528, 1.0183709, -0.1823764, 0.2581686
1: -0.0392806, 0.0219243, -0.0731202, 0.0795171, -0.1187977, 0.0950444
2: -0.0084018, 0.0686445, -0.0290733, 0.1401136, -0.1485154, 0.0977178
3: -0.0217253, 0.0601743, -0.0648682, 0.0974245, -0.1191498, 0.1250425
4: -0.0501302, 0.0173537, -0.1136745, 0.0599294, -0.1100596, 0.1310282
5: -0.0199253, 0.1390658, -0.0899277, 0.1982304, -0.2181557, 0.2289935
6: -0.0351016, 0.0367181, -0.0734944, 0.0769660, -0.1120676, 0.1102125
7: -0.0598738, 0.0328874, -0.1244399, 0.0671093, -0.1269832, 0.1573273
8: -0.0370814, 0.0577333, -0.0812962, 0.1184500, -0.1555314, 0.1390294
9: -0.0392852, 0.0437012, -0.0813323, 0.1368171, -0.1761024, 0.1250334

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of NS_A1_B1_B1_A1_B2_B1_A2_A1_A1

### Relational analysis result of NS_A1_B1_B1_A1_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5294882, upper bound: 0.5094757
time: 1.54 seconds

## Relational analysis of NS_A1_B1_B1_A1_B2_B1_A2_A1_A2

### Relational analysis result of NS_A1_B1_B1_A1_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5294882, upper bound: 0.5094831
time: 1.65 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: 0.7510265, 1.0170735, 0.7521383, 1.0168052, -0.2657787, 0.2649353
1: -0.0710923, 0.0761046, -0.0706728, 0.0753984, -0.1464908, 0.1467773
2: -0.0277500, 0.1358579, -0.0274763, 0.1349774, -0.1627274, 0.1633341
3: -0.0622978, 0.0952172, -0.0617660, 0.0947606, -0.1570584, 0.1569833
4: -0.1099092, 0.0573552, -0.1091301, 0.0568227, -0.1667318, 0.1664853
5: -0.0857340, 0.1947246, -0.0848664, 0.1939994, -0.2797334, 0.2795911
6: -0.0711976, 0.0745812, -0.0707223, 0.0740878, -0.1452853, 0.1453035
7: -0.1206128, 0.0650815, -0.1198208, 0.0646619, -0.1852747, 0.1849023
8: -0.0786762, 0.1148299, -0.0781342, 0.1140808, -0.1927570, 0.1929640
9: -0.0788408, 0.1312554, -0.0783253, 0.1301048, -0.2089456, 0.2095807

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 251

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A1_B1_B1_A1_B2_B1_A2_A2_A1

### Relational analysis result of NS_A1_B1_B1_A1_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5211326, upper bound: 0.5093601
time: 1.76 seconds

## Relational analysis of NS_A1_B1_B1_A1_B2_B1_A2_A2_A2

### Relational analysis result of NS_A1_B1_B1_A1_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5211326, upper bound: 0.5093601
time: 1.55 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B2_B2_B1_A1

### Backsubstitution after applying NS history:
0: 0.8484727, 1.0039656, 0.7731626, 1.0117297, -0.1632570, 0.2308030
1: -0.0363503, 0.0131010, -0.0627391, 0.0620466, -0.0983969, 0.0758401
2: -0.0085921, 0.0601392, -0.0222991, 0.1183274, -0.1269194, 0.0824383
3: -0.0172280, 0.0544676, -0.0517094, 0.0861248, -0.1033528, 0.1061771
4: -0.0427606, 0.0142456, -0.0943986, 0.0467513, -0.0895119, 0.1086442
5: -0.0129087, 0.1300019, -0.0684594, 0.1802831, -0.1931918, 0.1984613
6: -0.0306781, 0.0315669, -0.0617359, 0.0647569, -0.0954351, 0.0933028
7: -0.0515202, 0.0276446, -0.1048467, 0.0567282, -0.1082484, 0.1324914
8: -0.0329342, 0.0499224, -0.0678838, 0.0999170, -0.1328513, 0.1178062
9: -0.0328437, 0.0374143, -0.0685774, 0.1083453, -0.1411889, 0.1059917

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of NS_A1_B1_B1_A1_B2_B2_B1_A1_A1

### Relational analysis result of NS_A1_B1_B1_A1_B2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5288378, upper bound: 0.5058612
time: 1.59 seconds

## Relational analysis of NS_A1_B1_B1_A1_B2_B2_B1_A1_A2

### Relational analysis result of NS_A1_B1_B1_A1_B2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5288378, upper bound: 0.5058712
time: 1.65 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B2_B2_B1_A2

### Backsubstitution after applying NS history:
0: 0.7724995, 1.0076349, 0.7800274, 1.0100725, -0.2375730, 0.2276075
1: -0.0538876, 0.0668202, -0.0601485, 0.0576868, -0.1115744, 0.1269687
2: -0.0241501, 0.1114917, -0.0206086, 0.1128906, -0.1370406, 0.1321003
3: -0.0441290, 0.0892124, -0.0484256, 0.0833049, -0.1274339, 0.1376380
4: -0.0866688, 0.0331688, -0.0895882, 0.0434628, -0.1301315, 0.1227570
5: -0.0553540, 0.1851871, -0.0631020, 0.1758043, -0.2311583, 0.2482891
6: -0.0576102, 0.0625174, -0.0588015, 0.0617102, -0.1193205, 0.1213189
7: -0.1017951, 0.0595647, -0.0999572, 0.0541375, -0.1559326, 0.1595219
8: -0.0571175, 0.0974784, -0.0645367, 0.0952922, -0.1524097, 0.1620151
9: -0.0720626, 0.0736405, -0.0653945, 0.1012400, -0.1733027, 0.1390350

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 245

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of NS_A1_B1_B1_A1_B2_B2_B1_A2_A1

### Relational analysis result of NS_A1_B1_B1_A1_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5227326, upper bound: 0.5057864
time: 1.62 seconds

## Relational analysis of NS_A1_B1_B1_A1_B2_B2_B1_A2_A2

### Relational analysis result of NS_A1_B1_B1_A1_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5227326, upper bound: 0.5057898
time: 2.63 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B2_B2_B2_A1

### Backsubstitution after applying NS history:
0: 0.8458276, 1.0041152, 0.7595805, 1.0150084, -0.1691809, 0.2445347
1: -0.0369609, 0.0149714, -0.0678644, 0.0706721, -0.1076330, 0.0828358
2: -0.0087898, 0.0619272, -0.0256436, 0.1290836, -0.1378734, 0.0875708
3: -0.0181646, 0.0556774, -0.0582062, 0.0917036, -0.1098683, 0.1138835
4: -0.0442894, 0.0149045, -0.1039154, 0.0532576, -0.0975470, 0.1188200
5: -0.0143865, 0.1319233, -0.0790586, 0.1891440, -0.2035306, 0.2109819
6: -0.0316159, 0.0326446, -0.0675413, 0.0707848, -0.1024007, 0.1001858
7: -0.0532707, 0.0287560, -0.1145203, 0.0618535, -0.1151242, 0.1432763
8: -0.0337762, 0.0515782, -0.0745057, 0.1090671, -0.1428434, 0.1260839
9: -0.0342092, 0.0386757, -0.0748748, 0.1224023, -0.1566115, 0.1135504

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of NS_A1_B1_B1_A1_B2_B2_B2_A1_A1

### Relational analysis result of NS_A1_B1_B1_A1_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5287735, upper bound: 0.5045156
time: 1.58 seconds

## Relational analysis of NS_A1_B1_B1_A1_B2_B2_B2_A1_A2

### Relational analysis result of NS_A1_B1_B1_A1_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5287735, upper bound: 0.5045434
time: 1.59 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B2_B2_B2_A2

### Backsubstitution after applying NS history:
0: 0.7698260, 1.0081077, 0.7663992, 1.0133624, -0.2435364, 0.2417085
1: -0.0545048, 0.0687106, -0.0652913, 0.0663417, -0.1208465, 0.1340019
2: -0.0248831, 0.1132988, -0.0239645, 0.1236834, -0.1485665, 0.1372634
3: -0.0450756, 0.0904351, -0.0549445, 0.0889028, -0.1339784, 0.1453796
4: -0.0882140, 0.0338347, -0.0991375, 0.0499912, -0.1382051, 0.1329723
5: -0.0568477, 0.1871291, -0.0737374, 0.1846955, -0.2415432, 0.2608665
6: -0.0585580, 0.0636066, -0.0646267, 0.0677586, -0.1263166, 0.1282333
7: -0.1035642, 0.0606880, -0.1096637, 0.0592803, -0.1628446, 0.1703517
8: -0.0579685, 0.0991520, -0.0711812, 0.1044734, -0.1624419, 0.1703332
9: -0.0734428, 0.0749153, -0.0717132, 0.1153451, -0.1887878, 0.1466285

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 251

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of NS_A1_B1_B1_A1_B2_B2_B2_A2_A1

### Relational analysis result of NS_A1_B1_B1_A1_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5227195, upper bound: 0.5044477
time: 1.49 seconds

## Relational analysis of NS_A1_B1_B1_A1_B2_B2_B2_A2_A2

### Relational analysis result of NS_A1_B1_B1_A1_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5227195, upper bound: 0.5044693
time: 2.93 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: 0.8993907, 1.0037953, 0.8385299, 1.0039562, -0.1045655, 0.1652653
1: -0.0234024, -0.0003183, -0.0386454, 0.0201313, -0.0435338, 0.0383272
2: -0.0083673, 0.0282575, -0.0085798, 0.0668597, -0.0752270, 0.0368374
3: -0.0097523, 0.0288379, -0.0207485, 0.0590147, -0.0687670, 0.0495864
4: -0.0178923, 0.0069584, -0.0485069, 0.0167221, -0.0346144, 0.0554653
5: -0.0115645, 0.0892940, -0.0184635, 0.1372240, -0.1487885, 0.1077575
6: -0.0159647, 0.0151951, -0.0342028, 0.0356174, -0.0515821, 0.0493979
7: -0.0297626, 0.0084432, -0.0580997, 0.0318220, -0.0615846, 0.0665429
8: -0.0167847, 0.0293519, -0.0360991, 0.0561461, -0.0729307, 0.0654510
9: -0.0087747, 0.0240980, -0.0379763, 0.0421553, -0.0509300, 0.0620743

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_B2_A1_B1_B1_A1_A1_A1

### Relational analysis result of NS_A1_B1_B2_A1_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5329281, upper bound: 0.5235797
time: 2.04 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_B1_A1_A1_A2

### Relational analysis result of NS_A1_B1_B2_A1_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5210846, upper bound: 0.5235063
time: 1.99 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: 0.9020573, 1.0042387, 0.8582391, 1.0039322, -0.1018749, 0.1459997
1: -0.0225780, -0.0002078, -0.0340959, 0.0094167, -0.0319948, 0.0338882
2: -0.0089529, 0.0277172, -0.0085480, 0.0535379, -0.0624908, 0.0362652
3: -0.0097021, 0.0272078, -0.0152730, 0.0500012, -0.0597033, 0.0424808
4: -0.0169145, 0.0068767, -0.0382319, 0.0118131, -0.0287276, 0.0451086
5: -0.0123010, 0.0869267, -0.0117917, 0.1229080, -0.1352090, 0.0987185
6: -0.0154305, 0.0144982, -0.0278014, 0.0275883, -0.0430187, 0.0422996
7: -0.0289811, 0.0089268, -0.0463036, 0.0235414, -0.0525225, 0.0552305
8: -0.0161715, 0.0282636, -0.0298255, 0.0454205, -0.0615920, 0.0580892
9: -0.0081372, 0.0232977, -0.0278022, 0.0347512, -0.0428884, 0.0510998

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_B2_A1_B1_B1_A1_A2_A1

### Relational analysis result of NS_A1_B1_B2_A1_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5404746, upper bound: 0.5236054
time: 1.96 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_B1_A1_A2_A2

### Relational analysis result of NS_A1_B1_B2_A1_B1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5271396, upper bound: 0.5235426
time: 2.01 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: 0.8836805, 1.0042089, 0.8360225, 1.0041095, -0.1204290, 0.1681864
1: -0.0277740, 0.0016513, -0.0392724, 0.0219044, -0.0496784, 0.0409237
2: -0.0089136, 0.0362533, -0.0087821, 0.0686231, -0.0775366, 0.0450354
3: -0.0114525, 0.0374819, -0.0217125, 0.0601616, -0.0716140, 0.0591944
4: -0.0261519, 0.0074627, -0.0501083, 0.0173467, -0.0434986, 0.0575710
5: -0.0122515, 0.1030147, -0.0199081, 0.1390456, -0.1512970, 0.1229227
6: -0.0203122, 0.0188903, -0.0350917, 0.0367043, -0.0570165, 0.0539820
7: -0.0344315, 0.0120346, -0.0598519, 0.0328756, -0.0673071, 0.0718865
8: -0.0214111, 0.0351228, -0.0370662, 0.0577158, -0.0791269, 0.0721890
9: -0.0153067, 0.0283909, -0.0392708, 0.0436757, -0.0589824, 0.0676617

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_B2_A1_B1_B1_A2_A1_A1

### Relational analysis result of NS_A1_B1_B2_A1_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5328511, upper bound: 0.5236712
time: 1.67 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_B1_A2_A1_A2

### Relational analysis result of NS_A1_B1_B2_A1_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5210342, upper bound: 0.5235869
time: 1.74 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: 0.8891956, 1.0046471, 0.8557150, 1.0040852, -0.1148896, 0.1489322
1: -0.0262440, 0.0003828, -0.0346786, 0.0102040, -0.0364480, 0.0350613
2: -0.0094921, 0.0326101, -0.0087501, 0.0552440, -0.0647361, 0.0413603
3: -0.0108068, 0.0344566, -0.0157013, 0.0511556, -0.0619624, 0.0501579
4: -0.0232284, 0.0072400, -0.0393452, 0.0124417, -0.0356701, 0.0465852
5: -0.0129792, 0.0982002, -0.0120460, 0.1247414, -0.1377206, 0.1102461
6: -0.0187161, 0.0175970, -0.0285149, 0.0286165, -0.0473327, 0.0461119
7: -0.0324563, 0.0093722, -0.0475880, 0.0246018, -0.0570581, 0.0569602
8: -0.0196450, 0.0331030, -0.0306289, 0.0465015, -0.0661464, 0.0637320
9: -0.0129870, 0.0268569, -0.0291051, 0.0353374, -0.0483245, 0.0559620

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_B2_A1_B1_B1_A2_A2_A1

### Relational analysis result of NS_A1_B1_B2_A1_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5404746, upper bound: 0.5237053
time: 2.11 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_B1_A2_A2_A2

### Relational analysis result of NS_A1_B1_B2_A1_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5271396, upper bound: 0.5236311
time: 1.82 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.8578250, 1.0038600, 0.8695760, 1.0039747, -0.1461497, 0.1342840
1: -0.0341915, 0.0095459, -0.0314790, 0.0058806, -0.0400721, 0.0410248
2: -0.0084525, 0.0538178, -0.0086042, 0.0458748, -0.0543273, 0.0624220
3: -0.0153433, 0.0501906, -0.0133492, 0.0448165, -0.0601598, 0.0635398
4: -0.0384145, 0.0119162, -0.0332313, 0.0089892, -0.0474037, 0.0451475
5: -0.0116716, 0.1232087, -0.0118624, 0.1146730, -0.1263446, 0.1350712
6: -0.0279184, 0.0277569, -0.0245965, 0.0229697, -0.0508881, 0.0523535
7: -0.0465144, 0.0237153, -0.0405347, 0.0187780, -0.0652924, 0.0642500
8: -0.0299573, 0.0455978, -0.0262168, 0.0405652, -0.0705225, 0.0718146
9: -0.0280159, 0.0348474, -0.0219497, 0.0321184, -0.0601342, 0.0567970

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_B2_A1_B1_B2_A1_B1_B1

### Relational analysis result of NS_A1_B1_B2_A1_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5272606, upper bound: 0.5338656
time: 2.03 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_B2_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2_A1_B1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5271781, upper bound: 0.5210706
time: 2.06 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.8749458, 1.0038371, 0.8817182, 1.0043988, -0.1294531, 0.1221189
1: -0.0301973, 0.0041546, -0.0283184, 0.0021048, -0.0323021, 0.0324731
2: -0.0084225, 0.0422372, -0.0091643, 0.0375976, -0.0460201, 0.0514015
3: -0.0124847, 0.0422773, -0.0116844, 0.0385584, -0.0510431, 0.0539617
4: -0.0307824, 0.0078370, -0.0271922, 0.0075446, -0.0383270, 0.0350291
5: -0.0116338, 0.1106400, -0.0125669, 0.1047278, -0.1163616, 0.1232069
6: -0.0230271, 0.0209386, -0.0208837, 0.0193505, -0.0423775, 0.0418224
7: -0.0378724, 0.0164453, -0.0351546, 0.0130255, -0.0508979, 0.0516000
8: -0.0244495, 0.0383217, -0.0220476, 0.0358415, -0.0602909, 0.0603693
9: -0.0192064, 0.0308289, -0.0161321, 0.0289386, -0.0481451, 0.0469610

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_B2_A1_B1_B2_A1_B2_B1

### Relational analysis result of NS_A1_B1_B2_A1_B1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5272835, upper bound: 0.5413906
time: 2.34 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_B2_A1_B2_B2

### Relational analysis result of NS_A1_B1_B2_A1_B1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5272098, upper bound: 0.5271175
time: 1.84 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.8393070, 1.0042719, 0.8671592, 1.0041252, -0.1648182, 0.1371127
1: -0.0384661, 0.0195820, -0.0320368, 0.0066345, -0.0451006, 0.0516188
2: -0.0089967, 0.0663346, -0.0088029, 0.0475085, -0.0565053, 0.0751375
3: -0.0204735, 0.0586594, -0.0137594, 0.0459218, -0.0663953, 0.0724188
4: -0.0480579, 0.0165286, -0.0342974, 0.0095912, -0.0576492, 0.0508260
5: -0.0180295, 0.1366597, -0.0121124, 0.1164286, -0.1344581, 0.1487721
6: -0.0339274, 0.0353010, -0.0252798, 0.0239543, -0.0578816, 0.0605808
7: -0.0575856, 0.0314956, -0.0417645, 0.0197936, -0.0773791, 0.0732601
8: -0.0358518, 0.0556598, -0.0269861, 0.0416003, -0.0774521, 0.0826459
9: -0.0375752, 0.0417849, -0.0231973, 0.0326796, -0.0702549, 0.0649822

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_B2_A1_B1_B2_A2_B1_B1

### Relational analysis result of NS_A1_B1_B2_A1_B1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5272606, upper bound: 0.5338795
time: 1.79 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_B2_A2_B1_B2

### Relational analysis result of NS_A1_B1_B2_A1_B1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5271781, upper bound: 0.5211465
time: 2.17 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.8577688, 1.0042491, 0.8796499, 1.0045490, -0.1467803, 0.1245992
1: -0.0342045, 0.0095634, -0.0288922, 0.0025828, -0.0367873, 0.0384556
2: -0.0089665, 0.0538557, -0.0093626, 0.0390145, -0.0479810, 0.0632183
3: -0.0153528, 0.0502163, -0.0119288, 0.0396930, -0.0550458, 0.0621451
4: -0.0384393, 0.0119301, -0.0282886, 0.0076311, -0.0460703, 0.0402187
5: -0.0123181, 0.1232495, -0.0128163, 0.1065333, -0.1188515, 0.1360658
6: -0.0279343, 0.0277798, -0.0214860, 0.0198355, -0.0477698, 0.0492658
7: -0.0465429, 0.0237389, -0.0359168, 0.0140699, -0.0606128, 0.0596557
8: -0.0299752, 0.0456219, -0.0227185, 0.0365989, -0.0665741, 0.0683403
9: -0.0280448, 0.0348604, -0.0170021, 0.0295159, -0.0575607, 0.0518625

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_B2_A1_B1_B2_A2_B2_B1

### Relational analysis result of NS_A1_B1_B2_A1_B1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5272835, upper bound: 0.5414050
time: 1.93 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_B2_A2_B2_B2

### Relational analysis result of NS_A1_B1_B2_A1_B1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5272098, upper bound: 0.5272087
time: 1.71 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B2_B1_B1_A1

### Backsubstitution after applying NS history:
0: 0.8310639, 1.0039811, 0.6882526, 1.0218822, -0.1908183, 0.3157285
1: -0.0407419, 0.0250747, -0.1069111, 0.1213959, -0.1621378, 0.1319858
2: -0.0086119, 0.0722028, -0.0598213, 0.1889108, -0.1975226, 0.1320241
3: -0.0238697, 0.0622039, -0.0798449, 0.1410998, -0.1649695, 0.1420489
4: -0.0535721, 0.0188369, -0.1332126, 0.1014422, -0.1550143, 0.1520496
5: -0.0230301, 0.1422932, -0.1362083, 0.2203353, -0.2433654, 0.2785015
6: -0.0368246, 0.0389713, -0.0979205, 0.1076142, -0.1444387, 0.1368918
7: -0.0633264, 0.0348857, -0.1594229, 0.1326847, -0.1960112, 0.1943086
8: -0.0394672, 0.0606982, -0.1342014, 0.1299036, -0.1693708, 0.1948996
9: -0.0415656, 0.0480320, -0.1149814, 0.1787084, -0.2202740, 0.1630134

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 210

## Relational analysis of NS_A1_B1_B2_A1_B2_B1_B1_A1_A1

### Relational analysis result of NS_A1_B1_B2_A1_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5266606, upper bound: 0.5089650
time: 1.94 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2_B1_B1_A1_A2

### Relational analysis result of NS_A1_B1_B2_A1_B2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5291702, upper bound: 0.5095127
time: 1.83 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B2_B1_B1_A2

### Backsubstitution after applying NS history:
0: 0.5841720, 1.0184758, 0.6990843, 1.0203123, -0.4361403, 0.3193915
1: -0.0735269, 0.0957775, -0.1015591, 0.1142173, -0.1877442, 0.1973366
2: -0.0311222, 0.1527507, -0.0555676, 0.1797162, -0.2108384, 0.2083183
3: -0.0812600, 0.1053303, -0.0759086, 0.1348446, -0.2161045, 0.1812389
4: -0.1204913, 0.0621521, -0.1277261, 0.0948174, -0.2153086, 0.1898782
5: -0.1107231, 0.2115418, -0.1275568, 0.2149368, -0.3256598, 0.3390986
6: -0.0755107, 0.1118883, -0.0934341, 0.1022701, -0.1777808, 0.2053225
7: -0.1243536, 0.1334529, -0.1524467, 0.1240230, -0.2483766, 0.2858996
8: -0.0830847, 0.1364374, -0.1264800, 0.1253207, -0.2084054, 0.2629174
9: -0.0867477, 0.1438442, -0.1091337, 0.1693810, -0.2561287, 0.2529779

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 245

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A1_B1_B2_A1_B2_B1_B1_A2_A1

### Relational analysis result of NS_A1_B1_B2_A1_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5209231, upper bound: 0.5095323
time: 1.59 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2_B1_B1_A2_A2

### Relational analysis result of NS_A1_B1_B2_A1_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5209231, upper bound: 0.5095323
time: 1.61 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B2_B1_B2_A1

### Backsubstitution after applying NS history:
0: 0.8246547, 1.0041546, 0.6653655, 1.0251987, -0.2005440, 0.3387891
1: -0.0418495, 0.0273619, -0.1182197, 0.1365639, -0.1784133, 0.1455816
2: -0.0088338, 0.0748717, -0.0688093, 0.2083386, -0.2171723, 0.1436810
3: -0.0256615, 0.0636090, -0.0881623, 0.1543170, -0.1799785, 0.1517713
4: -0.0557921, 0.0203250, -0.1448052, 0.1154402, -0.1712323, 0.1651302
5: -0.0258538, 0.1445566, -0.1544885, 0.2317422, -0.2575960, 0.2990451
6: -0.0381213, 0.0410377, -0.1074000, 0.1189061, -0.1570274, 0.1484376
7: -0.0653940, 0.0373790, -0.1741630, 0.1509868, -0.2163808, 0.2115421
8: -0.0409326, 0.0631273, -0.1505165, 0.1395870, -0.1805196, 0.2136438
9: -0.0430594, 0.0512276, -0.1273372, 0.1984169, -0.2414763, 0.1785649

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of NS_A1_B1_B2_A1_B2_B1_B2_A1_B1

### Relational analysis result of NS_A1_B1_B2_A1_B2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5236860, upper bound: 0.5039629
time: 1.52 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2_B1_B2_A1_B2

### Relational analysis result of NS_A1_B1_B2_A1_B2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5290796, upper bound: 0.5084701
time: 1.67 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B2_B1_B2_A2

### Backsubstitution after applying NS history:
0: 0.5758343, 1.0192022, 0.6763388, 1.0236084, -0.4477741, 0.3428634
1: -0.0746702, 0.0981404, -0.1127978, 0.1292915, -0.2039617, 0.2109382
2: -0.0319240, 0.1554761, -0.0645000, 0.1990238, -0.2309478, 0.2199761
3: -0.0831817, 0.1067985, -0.0841745, 0.1479799, -0.2311616, 0.1909730
4: -0.1227941, 0.0636171, -0.1392471, 0.1087288, -0.2315229, 0.2028642
5: -0.1136854, 0.2139085, -0.1457240, 0.2262730, -0.3399584, 0.3596325
6: -0.0768515, 0.1143400, -0.1028550, 0.1134921, -0.1903436, 0.2171950
7: -0.1264856, 0.1368062, -0.1670958, 0.1422119, -0.2686976, 0.3039020
8: -0.0845990, 0.1390018, -0.1426940, 0.1349443, -0.2195432, 0.2816958
9: -0.0882989, 0.1471569, -0.1214132, 0.1889675, -0.2772664, 0.2685701

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 245

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A1_B1_B2_A1_B2_B1_B2_A2_A1

### Relational analysis result of NS_A1_B1_B2_A1_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5208986, upper bound: 0.5085023
time: 1.84 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2_B1_B2_A2_A2

### Relational analysis result of NS_A1_B1_B2_A1_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5208986, upper bound: 0.5085012
time: 1.64 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B2_B2_B1_A1

### Backsubstitution after applying NS history:
0: 0.8501607, 1.0039561, 0.7039602, 1.0196058, -0.1694451, 0.2999959
1: -0.0359607, 0.0119364, -0.0991498, 0.1109860, -0.1469467, 0.1110863
2: -0.0085797, 0.0589984, -0.0536528, 0.1755773, -0.1841570, 0.1126512
3: -0.0166438, 0.0536958, -0.0741367, 0.1320287, -0.1486725, 0.1278325
4: -0.0417952, 0.0138252, -0.1252564, 0.0918352, -0.1336304, 0.1390816
5: -0.0119792, 0.1287759, -0.1236624, 0.2125066, -0.2244858, 0.2524383
6: -0.0300851, 0.0308793, -0.0914146, 0.0998645, -0.1299496, 0.1222940
7: -0.0504145, 0.0269355, -0.1493065, 0.1201238, -0.1705383, 0.1762420
8: -0.0323970, 0.0488803, -0.1230042, 0.1232578, -0.1556548, 0.1718845
9: -0.0319724, 0.0366274, -0.1065015, 0.1651822, -0.1971546, 0.1431288

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of NS_A1_B1_B2_A1_B2_B2_B1_A1_B1

### Relational analysis result of NS_A1_B1_B2_A1_B2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5215245, upper bound: 0.4955477
time: 1.85 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2_B2_B1_A1_B2

### Relational analysis result of NS_A1_B1_B2_A1_B2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5284794, upper bound: 0.5053130
time: 1.52 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.81 + 598.58 = 602.39 seconds
