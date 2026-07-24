## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00390744


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0057170, 0.0075433, 0.0057170, 0.0075433, -0.0018263, 0.0018263)
1: (-0.0017535, 0.0026928, -0.0017535, 0.0026928, -0.0044463, 0.0044463)
2: (-0.0047920, 0.0248813, -0.0047920, 0.0248813, -0.0296734, 0.0296734)
3: (-0.0046330, -0.0020846, -0.0046330, -0.0020846, -0.0025484, 0.0025484)
4: (-0.0013036, 0.0112222, -0.0013036, 0.0112222, -0.0125258, 0.0125258)
5: (-0.0021950, 0.0009417, -0.0021950, 0.0009417, -0.0031367, 0.0031367)
6: (0.9886318, 0.9942788, 0.9886318, 0.9942788, -0.0056471, 0.0056471)
7: (-0.0158263, 0.0069313, -0.0158263, 0.0069313, -0.0227576, 0.0227576)
8: (-0.0094076, 0.0031599, -0.0094076, 0.0031599, -0.0125675, 0.0125675)
9: (-0.0136358, 0.0010965, -0.0136358, 0.0010965, -0.0147323, 0.0147323)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.33 + 3.77 = 5.11 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0043416, upper bound: 0.0043415

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 213

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0042269, upper bound: 0.0041713
time: 1.87 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0041964, upper bound: 0.0041964
time: 2.45 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 4.45 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 4.45
Output dim: 6, lower bound: -0.0042269, upper bound: 0.0041713
IS_A2, status: Status.UNKNOWN, split count: 1, time: 4.45
Output dim: 6, lower bound: -0.0041964, upper bound: 0.0041964

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0057288, 0.0074924, 0.0057185, 0.0075366, -0.0018077, 0.0017739
1: -0.0016488, 0.0025942, -0.0017400, 0.0026797, -0.0043285, 0.0043343
2: -0.0039970, 0.0245938, -0.0046861, 0.0248444, -0.0288415, 0.0292798
3: -0.0046164, -0.0021556, -0.0046308, -0.0020940, -0.0025224, 0.0024753
4: -0.0010732, 0.0108777, -0.0012740, 0.0111763, -0.0122495, 0.0121517
5: -0.0021435, 0.0003544, -0.0021881, 0.0008664, -0.0030099, 0.0025425
6: 0.9891213, 0.9941845, 0.9886946, 0.9942662, -0.0051449, 0.0054899
7: -0.0156474, 0.0063077, -0.0158033, 0.0068481, -0.0224955, 0.0221110
8: -0.0088625, 0.0029645, -0.0093377, 0.0031338, -0.0119963, 0.0123021
9: -0.0132459, 0.0009388, -0.0135838, 0.0010762, -0.0143221, 0.0145227

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 213

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0041714, upper bound: 0.0041714
time: 2.41 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0041714, upper bound: 0.0041714
time: 2.41 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0056132, 0.0082665, 0.0057227, 0.0075279, -0.0019146, 0.0025438
1: -0.0026705, 0.0026099, -0.0017026, 0.0026629, -0.0053334, 0.0043125
2: -0.0041230, 0.0274010, -0.0045504, 0.0247417, -0.0288647, 0.0319514
3: -0.0047777, -0.0021443, -0.0046249, -0.0021062, -0.0026716, 0.0024806
4: -0.0033217, 0.0109323, -0.0011917, 0.0111175, -0.0144392, 0.0121240
5: -0.0021517, 0.0060874, -0.0021793, 0.0006566, -0.0028083, 0.0082667
6: 0.9843435, 0.9941995, 0.9888694, 0.9942501, -0.0099066, 0.0053300
7: -0.0173940, 0.0064064, -0.0157394, 0.0067418, -0.0241357, 0.0221458
8: -0.0141835, 0.0029954, -0.0091429, 0.0031005, -0.0172840, 0.0121384
9: -0.0133076, 0.0024777, -0.0135173, 0.0010199, -0.0143276, 0.0159950

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=58, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 213

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039521, upper bound: 0.0040220
time: 1.99 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039568, upper bound: 0.0039566
time: 2.54 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 5.84 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.84
Output dim: 6, lower bound: -0.0041714, upper bound: 0.0041714
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.84
Output dim: 6, lower bound: -0.0041714, upper bound: 0.0041714
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.84
Output dim: 6, lower bound: -0.0039521, upper bound: 0.0040220
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 5.84
Output dim: 6, lower bound: -0.0039568, upper bound: 0.0039566

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.0057288, 0.0074924, 0.0057288, 0.0074924, -0.0017636, 0.0017636
1: -0.0016488, 0.0025942, -0.0016488, 0.0025942, -0.0042430, 0.0042430
2: -0.0039970, 0.0245938, -0.0039970, 0.0245938, -0.0285908, 0.0285908
3: -0.0046164, -0.0021556, -0.0046164, -0.0021556, -0.0024609, 0.0024609
4: -0.0010732, 0.0108777, -0.0010732, 0.0108777, -0.0119509, 0.0119509
5: -0.0021435, 0.0003544, -0.0021435, 0.0003544, -0.0024980, 0.0024980
6: 0.9891213, 0.9941845, 0.9891213, 0.9941845, -0.0050632, 0.0050632
7: -0.0156474, 0.0063077, -0.0156474, 0.0063077, -0.0219550, 0.0219550
8: -0.0088625, 0.0029645, -0.0088625, 0.0029645, -0.0118270, 0.0118270
9: -0.0132459, 0.0009388, -0.0132459, 0.0009388, -0.0141847, 0.0141847

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 213

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040539, upper bound: 0.0039249
time: 2.33 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039990, upper bound: 0.0039348
time: 1.97 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0057288, 0.0074924, 0.0056132, 0.0082665, -0.0025377, 0.0018792
1: -0.0016488, 0.0025942, -0.0026705, 0.0026099, -0.0042586, 0.0052648
2: -0.0039970, 0.0245938, -0.0041230, 0.0274010, -0.0313980, 0.0287167
3: -0.0046164, -0.0021556, -0.0047777, -0.0021443, -0.0024721, 0.0026221
4: -0.0010732, 0.0108777, -0.0033217, 0.0109323, -0.0120055, 0.0141994
5: -0.0021435, 0.0003544, -0.0021517, 0.0060874, -0.0082309, 0.0025061
6: 0.9891213, 0.9941845, 0.9843435, 0.9941995, -0.0050781, 0.0098410
7: -0.0156474, 0.0063077, -0.0173940, 0.0064064, -0.0220538, 0.0237016
8: -0.0088625, 0.0029645, -0.0141835, 0.0029954, -0.0118579, 0.0171480
9: -0.0132459, 0.0009388, -0.0133076, 0.0024777, -0.0157235, 0.0142465

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=58, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 213

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040539, upper bound: 0.0039249
time: 2.28 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039990, upper bound: 0.0039348
time: 1.93 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.0056132, 0.0082665, 0.0057374, 0.0074937, -0.0018804, 0.0025291
1: -0.0026705, 0.0026099, -0.0015725, 0.0025966, -0.0052671, 0.0041824
2: -0.0041230, 0.0274010, -0.0040162, 0.0243842, -0.0285072, 0.0314172
3: -0.0047777, -0.0021443, -0.0046044, -0.0021539, -0.0026239, 0.0024601
4: -0.0033217, 0.0109323, -0.0010035, 0.0108860, -0.0142077, 0.0119358
5: -0.0021517, 0.0060874, -0.0021448, 0.0002587, -0.0024104, 0.0082322
6: 0.9843435, 0.9941995, 0.9892869, 0.9941868, -0.0098433, 0.0049126
7: -0.0173940, 0.0064064, -0.0155170, 0.0063227, -0.0237167, 0.0219234
8: -0.0141835, 0.0029954, -0.0084653, 0.0029692, -0.0171527, 0.0114608
9: -0.0133076, 0.0024777, -0.0132553, 0.0008244, -0.0141320, 0.0157330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=58, inp2_unstable=56, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039519, upper bound: 0.0039519
time: 2.04 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039519, upper bound: 0.0039568
time: 2.00 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.0056224, 0.0080988, 0.0056567, 0.0074750, -0.0018527, 0.0024422
1: -0.0025898, 0.0025736, -0.0022866, 0.0025606, -0.0051504, 0.0048602
2: -0.0038307, 0.0271792, -0.0037257, 0.0263461, -0.0301768, 0.0309049
3: -0.0047650, -0.0021704, -0.0047171, -0.0021798, -0.0025852, 0.0025467
4: -0.0031440, 0.0108056, -0.0024768, 0.0107601, -0.0139041, 0.0132824
5: -0.0021328, 0.0056344, -0.0021260, 0.0039331, -0.0060659, 0.0077604
6: 0.9847211, 0.9941648, 0.9861389, 0.9941523, -0.0094312, 0.0080258
7: -0.0172560, 0.0061771, -0.0167377, 0.0060948, -0.0233508, 0.0229148
8: -0.0137631, 0.0029236, -0.0121840, 0.0028978, -0.0166608, 0.0151076
9: -0.0131643, 0.0023561, -0.0131128, 0.0018994, -0.0150637, 0.0154689

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=58, inp2_unstable=57, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038655, upper bound: 0.0037970
time: 2.48 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038098, upper bound: 0.0038097
time: 2.19 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 5.99 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.99
Output dim: 6, lower bound: -0.0040539, upper bound: 0.0039249
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.99
Output dim: 6, lower bound: -0.0039990, upper bound: 0.0039348
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.99
Output dim: 6, lower bound: -0.0040539, upper bound: 0.0039249
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.99
Output dim: 6, lower bound: -0.0039990, upper bound: 0.0039348
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.99
Output dim: 6, lower bound: -0.0039519, upper bound: 0.0039519
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.99
Output dim: 6, lower bound: -0.0039519, upper bound: 0.0039568
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 5.99
Output dim: 6, lower bound: -0.0038655, upper bound: 0.0037970
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 5.99
Output dim: 6, lower bound: -0.0038098, upper bound: 0.0038097

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0057433, 0.0074577, 0.0057288, 0.0074924, -0.0017491, 0.0017289
1: -0.0015207, 0.0025269, -0.0016488, 0.0025942, -0.0041149, 0.0041757
2: -0.0034542, 0.0242418, -0.0039970, 0.0245938, -0.0280479, 0.0282388
3: -0.0045962, -0.0022041, -0.0046164, -0.0021556, -0.0024406, 0.0024124
4: -0.0009638, 0.0106424, -0.0010732, 0.0108777, -0.0118415, 0.0117157
5: -0.0021084, 0.0002196, -0.0021435, 0.0003544, -0.0024629, 0.0023632
6: 0.9893845, 0.9941201, 0.9891213, 0.9941845, -0.0048000, 0.0049987
7: -0.0154284, 0.0058818, -0.0156474, 0.0063077, -0.0217360, 0.0215292
8: -0.0081953, 0.0028311, -0.0088625, 0.0029645, -0.0111598, 0.0116936
9: -0.0129796, 0.0007466, -0.0132459, 0.0009388, -0.0139184, 0.0139925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=57, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 213

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040487, upper bound: 0.0040487
time: 2.23 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040487, upper bound: 0.0040485
time: 2.50 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0056626, 0.0074408, 0.0057381, 0.0074746, -0.0018119, 0.0017027
1: -0.0022339, 0.0024943, -0.0015669, 0.0025597, -0.0047935, 0.0040611
2: -0.0031906, 0.0262013, -0.0037180, 0.0243688, -0.0275593, 0.0299193
3: -0.0047088, -0.0022276, -0.0046035, -0.0021805, -0.0025283, 0.0023759
4: -0.0023608, 0.0105282, -0.0009992, 0.0107568, -0.0131176, 0.0115275
5: -0.0020914, 0.0036374, -0.0021255, 0.0002545, -0.0023459, 0.0057629
6: 0.9863853, 0.9940888, 0.9892975, 0.9941515, -0.0077661, 0.0047913
7: -0.0166476, 0.0056750, -0.0155074, 0.0060888, -0.0227363, 0.0211824
8: -0.0119096, 0.0027663, -0.0084360, 0.0028959, -0.0148055, 0.0112023
9: -0.0128503, 0.0018201, -0.0131090, 0.0008160, -0.0136662, 0.0149290

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=58, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038919, upper bound: 0.0039655
time: 2.30 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038919, upper bound: 0.0039072
time: 2.24 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0057433, 0.0074577, 0.0056132, 0.0082665, -0.0025232, 0.0018444
1: -0.0015207, 0.0025269, -0.0026705, 0.0026099, -0.0041305, 0.0051975
2: -0.0034542, 0.0242418, -0.0041230, 0.0274010, -0.0308552, 0.0283647
3: -0.0045962, -0.0022041, -0.0047777, -0.0021443, -0.0024519, 0.0025737
4: -0.0009638, 0.0106424, -0.0033217, 0.0109323, -0.0118961, 0.0139641
5: -0.0021084, 0.0002196, -0.0021517, 0.0060874, -0.0081958, 0.0023713
6: 0.9893845, 0.9941201, 0.9843435, 0.9941995, -0.0048150, 0.0097766
7: -0.0154284, 0.0058818, -0.0173940, 0.0064064, -0.0218348, 0.0232758
8: -0.0081953, 0.0028311, -0.0141835, 0.0029954, -0.0111908, 0.0170145
9: -0.0129796, 0.0007466, -0.0133076, 0.0024777, -0.0154573, 0.0140543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=57, inp2_unstable=58, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039927, upper bound: 0.0039250
time: 1.97 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039927, upper bound: 0.0039250
time: 2.18 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0056626, 0.0074408, 0.0056224, 0.0080988, -0.0024362, 0.0018184
1: -0.0022339, 0.0024943, -0.0025898, 0.0025736, -0.0048075, 0.0050841
2: -0.0031906, 0.0262013, -0.0038307, 0.0271792, -0.0303698, 0.0300320
3: -0.0047088, -0.0022276, -0.0047650, -0.0021704, -0.0025384, 0.0025374
4: -0.0023608, 0.0105282, -0.0031440, 0.0108056, -0.0131664, 0.0136723
5: -0.0020914, 0.0036374, -0.0021328, 0.0056344, -0.0077258, 0.0057702
6: 0.9863853, 0.9940888, 0.9847211, 0.9941648, -0.0077794, 0.0093677
7: -0.0166476, 0.0056750, -0.0172560, 0.0061771, -0.0228247, 0.0229310
8: -0.0119096, 0.0027663, -0.0137631, 0.0029236, -0.0148332, 0.0165294
9: -0.0128503, 0.0018201, -0.0131643, 0.0023561, -0.0152064, 0.0149843

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=58, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038384, upper bound: 0.0038451
time: 1.97 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038521, upper bound: 0.0037920
time: 2.71 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0056275, 0.0080050, 0.0057374, 0.0074937, -0.0018662, 0.0022676
1: -0.0025446, 0.0025452, -0.0015725, 0.0025966, -0.0051413, 0.0041177
2: -0.0036016, 0.0270551, -0.0040162, 0.0243842, -0.0279858, 0.0310714
3: -0.0047579, -0.0021909, -0.0046044, -0.0021539, -0.0026040, 0.0024135
4: -0.0030447, 0.0107063, -0.0010035, 0.0108860, -0.0139307, 0.0117099
5: -0.0021180, 0.0053811, -0.0021448, 0.0002587, -0.0023767, 0.0075259
6: 0.9849321, 0.9941376, 0.9892869, 0.9941868, -0.0092548, 0.0048507
7: -0.0171788, 0.0059974, -0.0155170, 0.0063227, -0.0235015, 0.0215144
8: -0.0135279, 0.0028673, -0.0084653, 0.0029692, -0.0164971, 0.0113326
9: -0.0130519, 0.0022881, -0.0132553, 0.0008244, -0.0138763, 0.0155434

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=58, inp2_unstable=56, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039250, upper bound: 0.0040221
time: 2.46 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039250, upper bound: 0.0040220
time: 2.64 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0055515, 0.0094001, 0.0057374, 0.0074937, -0.0019422, 0.0036626
1: -0.0032161, 0.0024979, -0.0015725, 0.0025966, -0.0058128, 0.0040704
2: -0.0032198, 0.0289001, -0.0040162, 0.0243842, -0.0276040, 0.0329163
3: -0.0048639, -0.0021618, -0.0046044, -0.0021539, -0.0027100, 0.0024426
4: -0.0045224, 0.0105409, -0.0010035, 0.0108860, -0.0154084, 0.0115444
5: -0.0020933, 0.0091489, -0.0021448, 0.0002587, -0.0023520, 0.0112936
6: 0.9817921, 0.9940923, 0.9892869, 0.9941868, -0.0123947, 0.0048054
7: -0.0183267, 0.0056980, -0.0155170, 0.0063227, -0.0246494, 0.0212149
8: -0.0170249, 0.0027735, -0.0084653, 0.0029692, -0.0199941, 0.0112388
9: -0.0128646, 0.0032994, -0.0132553, 0.0008244, -0.0136890, 0.0165547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=56, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039250, upper bound: 0.0040220
time: 2.27 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039250, upper bound: 0.0040221
time: 2.69 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 6.28 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 6, lower bound: -0.0040487, upper bound: 0.0040487
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 6, lower bound: -0.0040487, upper bound: 0.0040485
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 6, lower bound: -0.0038919, upper bound: 0.0039655
IS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 6.28
Output dim: 6, lower bound: -0.0038919, upper bound: 0.0039072
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 6, lower bound: -0.0039927, upper bound: 0.0039250
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 6, lower bound: -0.0039927, upper bound: 0.0039250
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 6.28
Output dim: 6, lower bound: -0.0038384, upper bound: 0.0038451
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 6.28
Output dim: 6, lower bound: -0.0038521, upper bound: 0.0037920
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 6, lower bound: -0.0039250, upper bound: 0.0040221
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 6, lower bound: -0.0039250, upper bound: 0.0040220
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 6, lower bound: -0.0039250, upper bound: 0.0040220
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 6, lower bound: -0.0039250, upper bound: 0.0040221

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0057433, 0.0074577, 0.0057433, 0.0074577, -0.0017144, 0.0017144
1: -0.0015207, 0.0025269, -0.0015207, 0.0025269, -0.0040476, 0.0040476
2: -0.0034542, 0.0242418, -0.0034542, 0.0242418, -0.0276959, 0.0276959
3: -0.0045962, -0.0022041, -0.0045962, -0.0022041, -0.0023921, 0.0023921
4: -0.0009638, 0.0106424, -0.0009638, 0.0106424, -0.0116063, 0.0116063
5: -0.0021084, 0.0002196, -0.0021084, 0.0002196, -0.0023281, 0.0023281
6: 0.9893845, 0.9941201, 0.9893845, 0.9941201, -0.0047356, 0.0047356
7: -0.0154284, 0.0058818, -0.0154284, 0.0058818, -0.0213102, 0.0213102
8: -0.0081953, 0.0028311, -0.0081953, 0.0028311, -0.0110264, 0.0110264
9: -0.0129796, 0.0007466, -0.0129796, 0.0007466, -0.0137262, 0.0137262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=57, inp2_unstable=57, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040186, upper bound: 0.0038817
time: 2.62 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039746, upper bound: 0.0038990
time: 2.64 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0057433, 0.0074577, 0.0056626, 0.0074408, -0.0016975, 0.0017950
1: -0.0015207, 0.0025269, -0.0022339, 0.0024943, -0.0040149, 0.0047608
2: -0.0034542, 0.0242418, -0.0031906, 0.0262013, -0.0296555, 0.0274323
3: -0.0045962, -0.0022041, -0.0047088, -0.0022276, -0.0023686, 0.0025047
4: -0.0009638, 0.0106424, -0.0023608, 0.0105282, -0.0114921, 0.0130033
5: -0.0021084, 0.0002196, -0.0020914, 0.0036374, -0.0057458, 0.0023110
6: 0.9893845, 0.9941201, 0.9863853, 0.9940888, -0.0047043, 0.0077347
7: -0.0154284, 0.0058818, -0.0166476, 0.0056750, -0.0211034, 0.0225294
8: -0.0081953, 0.0028311, -0.0119096, 0.0027663, -0.0109616, 0.0147407
9: -0.0129796, 0.0007466, -0.0128503, 0.0018201, -0.0147996, 0.0135969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=57, inp2_unstable=56, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040186, upper bound: 0.0038817
time: 2.58 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039746, upper bound: 0.0038990
time: 2.34 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0056657, 0.0074359, 0.0057608, 0.0074416, -0.0017759, 0.0016752
1: -0.0022068, 0.0024848, -0.0013664, 0.0024957, -0.0047025, 0.0038512
2: -0.0031143, 0.0261268, -0.0032022, 0.0238180, -0.0269323, 0.0293290
3: -0.0047045, -0.0022344, -0.0045719, -0.0022266, -0.0024779, 0.0023375
4: -0.0023011, 0.0104952, -0.0008457, 0.0105333, -0.0128344, 0.0113409
5: -0.0020864, 0.0034852, -0.0020921, 0.0001033, -0.0021898, 0.0055774
6: 0.9865121, 0.9940798, 0.9896749, 0.9940902, -0.0075781, 0.0044048
7: -0.0166012, 0.0056152, -0.0151647, 0.0056842, -0.0222854, 0.0207799
8: -0.0117683, 0.0027476, -0.0073921, 0.0027692, -0.0145375, 0.0101396
9: -0.0128129, 0.0017792, -0.0128560, 0.0005153, -0.0133282, 0.0146352

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=56, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 213

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038657, upper bound: 0.0039005
time: 2.45 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038666, upper bound: 0.0039387
time: 2.28 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0057433, 0.0074577, 0.0056275, 0.0080050, -0.0022617, 0.0018302
1: -0.0015207, 0.0025269, -0.0025446, 0.0025452, -0.0040659, 0.0050716
2: -0.0034542, 0.0242418, -0.0036016, 0.0270551, -0.0305093, 0.0278433
3: -0.0045962, -0.0022041, -0.0047579, -0.0021909, -0.0024053, 0.0025538
4: -0.0009638, 0.0106424, -0.0030447, 0.0107063, -0.0116702, 0.0136871
5: -0.0021084, 0.0002196, -0.0021180, 0.0053811, -0.0074895, 0.0023376
6: 0.9893845, 0.9941201, 0.9849321, 0.9941376, -0.0047531, 0.0091880
7: -0.0154284, 0.0058818, -0.0171788, 0.0059974, -0.0214258, 0.0230606
8: -0.0081953, 0.0028311, -0.0135279, 0.0028673, -0.0110626, 0.0163590
9: -0.0129796, 0.0007466, -0.0130519, 0.0022881, -0.0152677, 0.0137985

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=57, inp2_unstable=58, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039627, upper bound: 0.0037725
time: 2.18 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039189, upper bound: 0.0037860
time: 2.48 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0057433, 0.0074577, 0.0055515, 0.0094001, -0.0036568, 0.0019062
1: -0.0015207, 0.0025269, -0.0032161, 0.0024979, -0.0040186, 0.0057431
2: -0.0034542, 0.0242418, -0.0032198, 0.0289001, -0.0323543, 0.0274615
3: -0.0045962, -0.0022041, -0.0048639, -0.0021618, -0.0024344, 0.0026598
4: -0.0009638, 0.0106424, -0.0045224, 0.0105409, -0.0115047, 0.0151648
5: -0.0021084, 0.0002196, -0.0020933, 0.0091489, -0.0112573, 0.0023129
6: 0.9893845, 0.9941201, 0.9817921, 0.9940923, -0.0047078, 0.0123280
7: -0.0154284, 0.0058818, -0.0183267, 0.0056980, -0.0211263, 0.0242085
8: -0.0081953, 0.0028311, -0.0170249, 0.0027735, -0.0109688, 0.0198560
9: -0.0129796, 0.0007466, -0.0128646, 0.0032994, -0.0162790, 0.0136112

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=57, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039627, upper bound: 0.0037725
time: 2.32 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039189, upper bound: 0.0037860
time: 2.49 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0056275, 0.0080050, 0.0057434, 0.0074577, -0.0018302, 0.0022616
1: -0.0025446, 0.0025452, -0.0015203, 0.0025269, -0.0050716, 0.0040655
2: -0.0036016, 0.0270551, -0.0034542, 0.0242407, -0.0278422, 0.0305093
3: -0.0047579, -0.0021909, -0.0045962, -0.0022041, -0.0025538, 0.0024052
4: -0.0030447, 0.0107063, -0.0009635, 0.0106424, -0.0136871, 0.0116699
5: -0.0021180, 0.0053811, -0.0021084, 0.0002193, -0.0023373, 0.0074895
6: 0.9849321, 0.9941376, 0.9893852, 0.9941201, -0.0091880, 0.0047524
7: -0.0171788, 0.0059974, -0.0154277, 0.0058818, -0.0230606, 0.0214251
8: -0.0135279, 0.0028673, -0.0081932, 0.0028311, -0.0163590, 0.0110605
9: -0.0130519, 0.0022881, -0.0129796, 0.0007460, -0.0137979, 0.0152677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=58, inp2_unstable=57, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040433, upper bound: 0.0040130
time: 1.99 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040063, upper bound: 0.0040312
time: 2.45 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0056275, 0.0080050, 0.0056275, 0.0074671, -0.0018397, 0.0023775
1: -0.0025446, 0.0025452, -0.0025444, 0.0025452, -0.0050898, 0.0050896
2: -0.0036016, 0.0270551, -0.0036016, 0.0270545, -0.0306560, 0.0306567
3: -0.0047579, -0.0021909, -0.0047578, -0.0021909, -0.0025670, 0.0025669
4: -0.0030447, 0.0107063, -0.0017479, 0.0107063, -0.0137510, 0.0124542
5: -0.0021180, 0.0053811, -0.0021180, 0.0009914, -0.0031094, 0.0074990
6: 0.9849321, 0.9941376, 0.9874569, 0.9941376, -0.0092055, 0.0066807
7: -0.0171788, 0.0059974, -0.0171784, 0.0059974, -0.0231762, 0.0231758
8: -0.0135279, 0.0028673, -0.0135266, 0.0028673, -0.0163952, 0.0163939
9: -0.0130519, 0.0022881, -0.0130519, 0.0022820, -0.0153339, 0.0153400

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=58, inp2_unstable=58, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040433, upper bound: 0.0040130
time: 2.51 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040063, upper bound: 0.0040312
time: 2.08 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0055515, 0.0094001, 0.0057434, 0.0074577, -0.0019062, 0.0036567
1: -0.0032161, 0.0024979, -0.0015203, 0.0025269, -0.0057431, 0.0040181
2: -0.0032198, 0.0289001, -0.0034542, 0.0242407, -0.0274604, 0.0323543
3: -0.0048639, -0.0021618, -0.0045962, -0.0022041, -0.0026598, 0.0024344
4: -0.0045224, 0.0105409, -0.0009635, 0.0106424, -0.0151648, 0.0115044
5: -0.0020933, 0.0091489, -0.0021084, 0.0002193, -0.0023126, 0.0112573
6: 0.9817921, 0.9940923, 0.9893852, 0.9941201, -0.0123280, 0.0047071
7: -0.0183267, 0.0056980, -0.0154277, 0.0058818, -0.0242085, 0.0211256
8: -0.0170249, 0.0027735, -0.0081932, 0.0028311, -0.0198560, 0.0109667
9: -0.0128646, 0.0032994, -0.0129796, 0.0007460, -0.0136106, 0.0162790

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=57, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 213

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038366, upper bound: 0.0038747
time: 2.43 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037860, upper bound: 0.0038857
time: 2.42 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0055515, 0.0094001, 0.0056275, 0.0074671, -0.0019156, 0.0037726
1: -0.0032161, 0.0024979, -0.0025444, 0.0025452, -0.0057614, 0.0050423
2: -0.0032198, 0.0289001, -0.0036016, 0.0270545, -0.0302742, 0.0325017
3: -0.0048639, -0.0021618, -0.0047578, -0.0021909, -0.0026730, 0.0025960
4: -0.0045224, 0.0105409, -0.0017479, 0.0107063, -0.0152287, 0.0122888
5: -0.0020933, 0.0091489, -0.0021180, 0.0009914, -0.0030847, 0.0112668
6: 0.9817921, 0.9940923, 0.9874569, 0.9941376, -0.0123455, 0.0066354
7: -0.0183267, 0.0056980, -0.0171784, 0.0059974, -0.0243241, 0.0228763
8: -0.0170249, 0.0027735, -0.0135266, 0.0028673, -0.0198922, 0.0163001
9: -0.0128646, 0.0032994, -0.0130519, 0.0022820, -0.0151466, 0.0163513

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=58, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 213

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038366, upper bound: 0.0038747
time: 2.40 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037860, upper bound: 0.0038857
time: 2.40 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 6.13 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.13
Output dim: 6, lower bound: -0.0040186, upper bound: 0.0038817
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.13
Output dim: 6, lower bound: -0.0039746, upper bound: 0.0038990
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.13
Output dim: 6, lower bound: -0.0040186, upper bound: 0.0038817
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.13
Output dim: 6, lower bound: -0.0039746, upper bound: 0.0038990
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.13
Output dim: 6, lower bound: -0.0038657, upper bound: 0.0039005
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.13
Output dim: 6, lower bound: -0.0038666, upper bound: 0.0039387
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.13
Output dim: 6, lower bound: -0.0039627, upper bound: 0.0037725
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.13
Output dim: 6, lower bound: -0.0039189, upper bound: 0.0037860
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.13
Output dim: 6, lower bound: -0.0039627, upper bound: 0.0037725
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.13
Output dim: 6, lower bound: -0.0039189, upper bound: 0.0037860
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.13
Output dim: 6, lower bound: -0.0040433, upper bound: 0.0040130
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.13
Output dim: 6, lower bound: -0.0040063, upper bound: 0.0040312
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.13
Output dim: 6, lower bound: -0.0040433, upper bound: 0.0040130
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.13
Output dim: 6, lower bound: -0.0040063, upper bound: 0.0040312
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.13
Output dim: 6, lower bound: -0.0038366, upper bound: 0.0038747
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 6.13
Output dim: 6, lower bound: -0.0037860, upper bound: 0.0038857
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 6.13
Output dim: 6, lower bound: -0.0038366, upper bound: 0.0038747
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 6.13
Output dim: 6, lower bound: -0.0037860, upper bound: 0.0038857

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0057653, 0.0074242, 0.0057467, 0.0074528, -0.0016875, 0.0016775
1: -0.0013262, 0.0024621, -0.0014912, 0.0025174, -0.0038436, 0.0039532
2: -0.0029308, 0.0237074, -0.0033771, 0.0241607, -0.0270915, 0.0270845
3: -0.0045655, -0.0022508, -0.0045916, -0.0022109, -0.0023546, 0.0023408
4: -0.0008149, 0.0104156, -0.0009412, 0.0106091, -0.0114239, 0.0113569
5: -0.0020746, 0.0000730, -0.0021034, 0.0001974, -0.0022720, 0.0021764
6: 0.9897507, 0.9940580, 0.9894400, 0.9941109, -0.0043602, 0.0046180
7: -0.0150959, 0.0054713, -0.0153779, 0.0058214, -0.0209172, 0.0208492
8: -0.0071824, 0.0027024, -0.0080417, 0.0028121, -0.0099946, 0.0107441
9: -0.0127229, 0.0004549, -0.0129418, 0.0007024, -0.0134253, 0.0133967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=57, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040727, upper bound: 0.0040727
time: 2.06 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040727, upper bound: 0.0040725
time: 2.22 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0056904, 0.0074302, 0.0057517, 0.0074473, -0.0017569, 0.0016785
1: -0.0019881, 0.0024738, -0.0014462, 0.0025069, -0.0044950, 0.0039199
2: -0.0030253, 0.0255260, -0.0032925, 0.0240371, -0.0270623, 0.0288185
3: -0.0046700, -0.0022424, -0.0045845, -0.0022185, -0.0024515, 0.0023421
4: -0.0013218, 0.0104566, -0.0009068, 0.0105724, -0.0118942, 0.0113634
5: -0.0020807, 0.0005720, -0.0020980, 0.0001635, -0.0022441, 0.0026700
6: 0.9885043, 0.9940692, 0.9895248, 0.9941009, -0.0055966, 0.0045444
7: -0.0162274, 0.0055454, -0.0153010, 0.0057550, -0.0219824, 0.0208464
8: -0.0106295, 0.0027257, -0.0078073, 0.0027914, -0.0134208, 0.0105330
9: -0.0127692, 0.0014476, -0.0129003, 0.0006349, -0.0134041, 0.0143479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=58, inp2_unstable=57, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040191, upper bound: 0.0040636
time: 2.44 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040657, upper bound: 0.0040657
time: 1.90 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0057653, 0.0074242, 0.0056657, 0.0074359, -0.0016706, 0.0017585
1: -0.0013262, 0.0024621, -0.0022068, 0.0024848, -0.0038110, 0.0046688
2: -0.0029308, 0.0237074, -0.0031143, 0.0261268, -0.0290576, 0.0268217
3: -0.0045655, -0.0022508, -0.0047045, -0.0022344, -0.0023311, 0.0024537
4: -0.0008149, 0.0104156, -0.0023011, 0.0104952, -0.0113101, 0.0127168
5: -0.0020746, 0.0000730, -0.0020864, 0.0034852, -0.0055598, 0.0021594
6: 0.9897507, 0.9940580, 0.9865121, 0.9940798, -0.0043290, 0.0075459
7: -0.0150959, 0.0054713, -0.0166012, 0.0056152, -0.0207111, 0.0220725
8: -0.0071824, 0.0027024, -0.0117683, 0.0027476, -0.0099300, 0.0144708
9: -0.0127229, 0.0004549, -0.0128129, 0.0017792, -0.0145021, 0.0132678

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 213

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039579, upper bound: 0.0038571
time: 2.37 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039579, upper bound: 0.0038576
time: 2.87 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0056904, 0.0074302, 0.0056705, 0.0074306, -0.0017401, 0.0017598
1: -0.0019881, 0.0024738, -0.0021647, 0.0024744, -0.0044625, 0.0046384
2: -0.0030253, 0.0255260, -0.0030307, 0.0260111, -0.0290364, 0.0285567
3: -0.0046700, -0.0022424, -0.0046979, -0.0022419, -0.0024281, 0.0024555
4: -0.0013218, 0.0104566, -0.0022085, 0.0104590, -0.0117808, 0.0126650
5: -0.0020807, 0.0005720, -0.0020810, 0.0032489, -0.0053296, 0.0026530
6: 0.9885043, 0.9940692, 0.9867090, 0.9940699, -0.0055656, 0.0073602
7: -0.0162274, 0.0055454, -0.0165292, 0.0055497, -0.0217770, 0.0220746
8: -0.0106295, 0.0027257, -0.0115490, 0.0027270, -0.0133565, 0.0142746
9: -0.0127692, 0.0014476, -0.0127719, 0.0017158, -0.0144850, 0.0142196

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=58, inp2_unstable=54, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 213

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039141, upper bound: 0.0038720
time: 2.38 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039141, upper bound: 0.0038733
time: 2.45 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0056758, 0.0074191, 0.0057608, 0.0074416, -0.0017657, 0.0016584
1: -0.0021172, 0.0024522, -0.0013664, 0.0024957, -0.0046129, 0.0038187
2: -0.0028515, 0.0258807, -0.0032022, 0.0238180, -0.0266695, 0.0290829
3: -0.0046904, -0.0022579, -0.0045719, -0.0022266, -0.0024638, 0.0023140
4: -0.0021040, 0.0103813, -0.0008457, 0.0105333, -0.0126373, 0.0112270
5: -0.0020694, 0.0029825, -0.0020921, 0.0001033, -0.0021728, 0.0050747
6: 0.9869310, 0.9940486, 0.9896749, 0.9940902, -0.0071592, 0.0043737
7: -0.0164481, 0.0054091, -0.0151647, 0.0056842, -0.0221323, 0.0205738
8: -0.0113018, 0.0026830, -0.0073921, 0.0027692, -0.0140709, 0.0100750
9: -0.0126840, 0.0016443, -0.0128560, 0.0005153, -0.0131993, 0.0145003

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=56, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038576, upper bound: 0.0039387
time: 2.56 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038576, upper bound: 0.0039387
time: 2.43 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0057653, 0.0074242, 0.0056308, 0.0079445, -0.0021792, 0.0017934
1: -0.0013262, 0.0024621, -0.0025155, 0.0025358, -0.0038619, 0.0049775
2: -0.0029308, 0.0237074, -0.0035254, 0.0269751, -0.0299059, 0.0272328
3: -0.0045655, -0.0022508, -0.0047533, -0.0021977, -0.0023678, 0.0025024
4: -0.0008149, 0.0104156, -0.0029805, 0.0106733, -0.0114882, 0.0133962
5: -0.0020746, 0.0000730, -0.0021130, 0.0052176, -0.0072922, 0.0021860
6: 0.9897507, 0.9940580, 0.9850683, 0.9941285, -0.0043778, 0.0089897
7: -0.0150959, 0.0054713, -0.0171290, 0.0059377, -0.0210336, 0.0226003
8: -0.0071824, 0.0027024, -0.0133762, 0.0028486, -0.0100310, 0.0160786
9: -0.0127229, 0.0004549, -0.0130145, 0.0022442, -0.0149671, 0.0134695

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=58, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040385, upper bound: 0.0039892
time: 2.43 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040385, upper bound: 0.0039892
time: 2.69 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0056904, 0.0074302, 0.0056361, 0.0078472, -0.0021568, 0.0017942
1: -0.0019881, 0.0024738, -0.0024687, 0.0025250, -0.0045131, 0.0049424
2: -0.0030253, 0.0255260, -0.0034386, 0.0268464, -0.0298717, 0.0289646
3: -0.0046700, -0.0022424, -0.0047459, -0.0022055, -0.0024645, 0.0025035
4: -0.0013218, 0.0104566, -0.0028775, 0.0106357, -0.0119575, 0.0133341
5: -0.0020807, 0.0005720, -0.0021074, 0.0049548, -0.0070355, 0.0026794
6: 0.9885043, 0.9940692, 0.9852874, 0.9941183, -0.0056140, 0.0087818
7: -0.0162274, 0.0055454, -0.0170489, 0.0058696, -0.0220970, 0.0225943
8: -0.0106295, 0.0027257, -0.0131323, 0.0028273, -0.0134567, 0.0158580
9: -0.0127692, 0.0014476, -0.0129720, 0.0021737, -0.0149429, 0.0144196

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=58, inp2_unstable=57, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039831, upper bound: 0.0039760
time: 2.49 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040303, upper bound: 0.0039803
time: 8.56 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0057653, 0.0074242, 0.0055548, 0.0093404, -0.0035751, 0.0018694
1: -0.0013262, 0.0024621, -0.0031874, 0.0024886, -0.0038148, 0.0056495
2: -0.0029308, 0.0237074, -0.0031448, 0.0288212, -0.0317519, 0.0268522
3: -0.0045655, -0.0022508, -0.0048593, -0.0021941, -0.0023714, 0.0026085
4: -0.0008149, 0.0104156, -0.0044592, 0.0105084, -0.0113233, 0.0148748
5: -0.0020746, 0.0000730, -0.0020884, 0.0089877, -0.0110622, 0.0021614
6: 0.9897507, 0.9940580, 0.9819264, 0.9940834, -0.0043327, 0.0121316
7: -0.0150959, 0.0054713, -0.0182776, 0.0056391, -0.0207350, 0.0237488
8: -0.0071824, 0.0027024, -0.0168753, 0.0027550, -0.0099375, 0.0195778
9: -0.0127229, 0.0004549, -0.0128278, 0.0032562, -0.0159790, 0.0132828

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 213

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039077, upper bound: 0.0037725
time: 2.70 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039077, upper bound: 0.0037725
time: 2.07 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0056904, 0.0074302, 0.0055600, 0.0092446, -0.0035541, 0.0018703
1: -0.0019881, 0.0024738, -0.0031413, 0.0024772, -0.0044653, 0.0056150
2: -0.0030253, 0.0255260, -0.0030531, 0.0286944, -0.0317197, 0.0285790
3: -0.0046700, -0.0022424, -0.0048520, -0.0022399, -0.0024301, 0.0026097
4: -0.0013218, 0.0104566, -0.0043576, 0.0104686, -0.0117904, 0.0148142
5: -0.0020807, 0.0005720, -0.0020825, 0.0087288, -0.0108095, 0.0026545
6: 0.9885043, 0.9940692, 0.9821422, 0.9940725, -0.0055682, 0.0119271
7: -0.0162274, 0.0055454, -0.0181987, 0.0055672, -0.0217946, 0.0237441
8: -0.0106295, 0.0027257, -0.0166351, 0.0027325, -0.0133620, 0.0193607
9: -0.0127692, 0.0014476, -0.0127829, 0.0031867, -0.0159559, 0.0142305

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=58, inp2_unstable=57, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 213

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038600, upper bound: 0.0037598
time: 2.21 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038913, upper bound: 0.0037608
time: 2.66 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0056499, 0.0075929, 0.0057467, 0.0074527, -0.0018028, 0.0018462
1: -0.0023463, 0.0024811, -0.0014908, 0.0025174, -0.0048637, 0.0039718
2: -0.0030840, 0.0265102, -0.0033771, 0.0241596, -0.0272436, 0.0298873
3: -0.0047265, -0.0022371, -0.0045915, -0.0022109, -0.0025156, 0.0023544
4: -0.0026082, 0.0104821, -0.0009409, 0.0106091, -0.0132172, 0.0114230
5: -0.0020845, 0.0042681, -0.0021034, 0.0001971, -0.0022816, 0.0063715
6: 0.9858598, 0.9940761, 0.9894408, 0.9941109, -0.0082512, 0.0046353
7: -0.0168397, 0.0055915, -0.0153772, 0.0058214, -0.0226611, 0.0209687
8: -0.0124949, 0.0027401, -0.0080396, 0.0028121, -0.0153071, 0.0107797
9: -0.0127980, 0.0019893, -0.0129418, 0.0007018, -0.0134998, 0.0149311

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=57, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039892, upper bound: 0.0040385
time: 2.24 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039892, upper bound: 0.0040385
time: 2.88 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0055761, 0.0089478, 0.0057518, 0.0074473, -0.0018712, 0.0031960
1: -0.0029984, 0.0024884, -0.0014457, 0.0025069, -0.0055053, 0.0039341
2: -0.0031433, 0.0283020, -0.0032925, 0.0240359, -0.0271792, 0.0315945
3: -0.0048295, -0.0022318, -0.0045844, -0.0022185, -0.0026110, 0.0023525
4: -0.0040433, 0.0105077, -0.0009065, 0.0105724, -0.0146157, 0.0114142
5: -0.0020883, 0.0079274, -0.0020980, 0.0001632, -0.0022515, 0.0100254
6: 0.9828101, 0.9940832, 0.9895255, 0.9941009, -0.0112908, 0.0045577
7: -0.0179546, 0.0056380, -0.0153003, 0.0057550, -0.0237096, 0.0209383
8: -0.0158912, 0.0027547, -0.0078052, 0.0027914, -0.0186826, 0.0105599
9: -0.0128271, 0.0029716, -0.0129003, 0.0006343, -0.0134614, 0.0158719

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=57, inp2_unstable=57, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039296, upper bound: 0.0040276
time: 2.24 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039802, upper bound: 0.0040303
time: 1.92 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0056499, 0.0075929, 0.0056308, 0.0074623, -0.0018123, 0.0019621
1: -0.0023463, 0.0024811, -0.0025153, 0.0025358, -0.0048821, 0.0049963
2: -0.0030840, 0.0265102, -0.0035254, 0.0269744, -0.0300585, 0.0300356
3: -0.0047265, -0.0022371, -0.0047532, -0.0021977, -0.0025289, 0.0025161
4: -0.0026082, 0.0104821, -0.0017256, 0.0106733, -0.0132815, 0.0122076
5: -0.0020845, 0.0042681, -0.0021130, 0.0009695, -0.0030539, 0.0063811
6: 0.9858598, 0.9940761, 0.9875118, 0.9941285, -0.0082688, 0.0065644
7: -0.0168397, 0.0055915, -0.0171286, 0.0059377, -0.0227774, 0.0227201
8: -0.0124949, 0.0027401, -0.0133749, 0.0028486, -0.0153435, 0.0161150
9: -0.0127980, 0.0019893, -0.0130145, 0.0022383, -0.0150364, 0.0150039

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=58, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039892, upper bound: 0.0040130
time: 2.54 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039892, upper bound: 0.0040130
time: 2.44 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0055761, 0.0089478, 0.0056361, 0.0074567, -0.0018805, 0.0033117
1: -0.0029984, 0.0024884, -0.0024684, 0.0025250, -0.0055235, 0.0049568
2: -0.0031433, 0.0283020, -0.0034386, 0.0268458, -0.0299890, 0.0317406
3: -0.0048295, -0.0022318, -0.0047458, -0.0022055, -0.0026240, 0.0025140
4: -0.0040433, 0.0105077, -0.0016897, 0.0106357, -0.0146790, 0.0121974
5: -0.0020883, 0.0079274, -0.0021074, 0.0009342, -0.0030225, 0.0100348
6: 0.9828101, 0.9940832, 0.9875998, 0.9941183, -0.0113082, 0.0064834
7: -0.0179546, 0.0056380, -0.0170485, 0.0058696, -0.0238242, 0.0226865
8: -0.0158912, 0.0027547, -0.0131310, 0.0028273, -0.0187185, 0.0158857
9: -0.0128271, 0.0029716, -0.0129720, 0.0021681, -0.0149952, 0.0159435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=57, inp2_unstable=57, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039296, upper bound: 0.0040013
time: 2.39 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039803, upper bound: 0.0040061
time: 2.51 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 6.24 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -0.0040727, upper bound: 0.0040727
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -0.0040727, upper bound: 0.0040725
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -0.0040191, upper bound: 0.0040636
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -0.0040657, upper bound: 0.0040657
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -0.0039579, upper bound: 0.0038571
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -0.0039579, upper bound: 0.0038576
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -0.0039141, upper bound: 0.0038720
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -0.0039141, upper bound: 0.0038733
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -0.0038576, upper bound: 0.0039387
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -0.0038576, upper bound: 0.0039387
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -0.0040385, upper bound: 0.0039892
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -0.0040385, upper bound: 0.0039892
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -0.0039831, upper bound: 0.0039760
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -0.0040303, upper bound: 0.0039803
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -0.0039077, upper bound: 0.0037725
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -0.0039077, upper bound: 0.0037725
IS_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 6.24
Output dim: 6, lower bound: -0.0038600, upper bound: 0.0037598
IS_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 6.24
Output dim: 6, lower bound: -0.0038913, upper bound: 0.0037608
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -0.0039892, upper bound: 0.0040385
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -0.0039892, upper bound: 0.0040385
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -0.0039296, upper bound: 0.0040276
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -0.0039802, upper bound: 0.0040303
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -0.0039892, upper bound: 0.0040130
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -0.0039892, upper bound: 0.0040130
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -0.0039296, upper bound: 0.0040013
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -0.0039803, upper bound: 0.0040061

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0057653, 0.0074242, 0.0057653, 0.0074242, -0.0016589, 0.0016589
1: -0.0013262, 0.0024621, -0.0013262, 0.0024621, -0.0037882, 0.0037882
2: -0.0029308, 0.0237074, -0.0029308, 0.0237074, -0.0266381, 0.0266381
3: -0.0045655, -0.0022508, -0.0045655, -0.0022508, -0.0023147, 0.0023147
4: -0.0008149, 0.0104156, -0.0008149, 0.0104156, -0.0112305, 0.0112305
5: -0.0020746, 0.0000730, -0.0020746, 0.0000730, -0.0021476, 0.0021476
6: 0.9897507, 0.9940580, 0.9897507, 0.9940580, -0.0043073, 0.0043073
7: -0.0150959, 0.0054713, -0.0150959, 0.0054713, -0.0205671, 0.0205671
8: -0.0071824, 0.0027024, -0.0071824, 0.0027024, -0.0098849, 0.0098849
9: -0.0127229, 0.0004549, -0.0127229, 0.0004549, -0.0131778, 0.0131778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0041034, upper bound: 0.0039999
time: 2.39 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0041060, upper bound: 0.0040482
time: 2.27 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0057653, 0.0074242, 0.0056904, 0.0074302, -0.0016649, 0.0017337
1: -0.0013262, 0.0024621, -0.0019881, 0.0024738, -0.0037999, 0.0044501
2: -0.0029308, 0.0237074, -0.0030253, 0.0255260, -0.0284567, 0.0267327
3: -0.0045655, -0.0022508, -0.0046700, -0.0022424, -0.0023231, 0.0024192
4: -0.0008149, 0.0104156, -0.0013218, 0.0104566, -0.0112715, 0.0117375
5: -0.0020746, 0.0000730, -0.0020807, 0.0005720, -0.0026466, 0.0021537
6: 0.9897507, 0.9940580, 0.9885043, 0.9940692, -0.0043185, 0.0055537
7: -0.0150959, 0.0054713, -0.0162274, 0.0055454, -0.0206413, 0.0216986
8: -0.0071824, 0.0027024, -0.0106295, 0.0027257, -0.0099081, 0.0133319
9: -0.0127229, 0.0004549, -0.0127692, 0.0014476, -0.0141705, 0.0132242

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=58, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0041034, upper bound: 0.0039999
time: 2.47 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0041060, upper bound: 0.0040482
time: 2.47 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0057046, 0.0074111, 0.0057654, 0.0073851, -0.0016805, 0.0016457
1: -0.0018633, 0.0024366, -0.0013257, 0.0023863, -0.0042496, 0.0037623
2: -0.0027258, 0.0251831, -0.0023194, 0.0237060, -0.0264318, 0.0275025
3: -0.0046503, -0.0022691, -0.0045654, -0.0023054, -0.0023449, 0.0022963
4: -0.0012262, 0.0103268, -0.0008145, 0.0101507, -0.0113770, 0.0111413
5: -0.0020613, 0.0004779, -0.0020350, 0.0000726, -0.0021339, 0.0025130
6: 0.9887394, 0.9940338, 0.9897516, 0.9939854, -0.0052460, 0.0042821
7: -0.0160141, 0.0053105, -0.0150950, 0.0049917, -0.0210058, 0.0202976
8: -0.0099796, 0.0026521, -0.0071799, 0.0025522, -0.0125318, 0.0098319
9: -0.0126223, 0.0012605, -0.0124230, 0.0004542, -0.0130765, 0.0136835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=54, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039378, upper bound: 0.0039708
time: 2.33 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039241, upper bound: 0.0039707
time: 2.42 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0056904, 0.0074302, 0.0057619, 0.0074311, -0.0017407, 0.0016684
1: -0.0019881, 0.0024738, -0.0013568, 0.0024754, -0.0044635, 0.0038305
2: -0.0030253, 0.0255260, -0.0030388, 0.0237914, -0.0268166, 0.0285648
3: -0.0046700, -0.0022424, -0.0045703, -0.0022412, -0.0024288, 0.0023280
4: -0.0013218, 0.0104566, -0.0008383, 0.0104625, -0.0117843, 0.0112949
5: -0.0020807, 0.0005720, -0.0020816, 0.0000960, -0.0021767, 0.0026536
6: 0.9885043, 0.9940692, 0.9896932, 0.9940708, -0.0055665, 0.0043761
7: -0.0162274, 0.0055454, -0.0151481, 0.0055560, -0.0217834, 0.0206935
8: -0.0106295, 0.0027257, -0.0073416, 0.0027290, -0.0133585, 0.0100673
9: -0.0127692, 0.0014476, -0.0127759, 0.0005008, -0.0132700, 0.0142235

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=58, inp2_unstable=56, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040637, upper bound: 0.0040190
time: 1.95 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040637, upper bound: 0.0040657
time: 2.47 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0057796, 0.0074047, 0.0056731, 0.0073741, -0.0015945, 0.0017316
1: -0.0012003, 0.0024244, -0.0021414, 0.0023650, -0.0035653, 0.0045658
2: -0.0026271, 0.0233616, -0.0021479, 0.0259471, -0.0285742, 0.0255095
3: -0.0045456, -0.0022779, -0.0046942, -0.0023207, -0.0022249, 0.0024162
4: -0.0007185, 0.0102840, -0.0021572, 0.0100764, -0.0107949, 0.0124412
5: -0.0020549, -0.0000219, -0.0020239, 0.0031182, -0.0051732, 0.0020021
6: 0.9899877, 0.9940220, 0.9868180, 0.9939651, -0.0039774, 0.0072040
7: -0.0148807, 0.0052330, -0.0164894, 0.0048571, -0.0197379, 0.0213469
8: -0.0065270, 0.0026278, -0.0114277, 0.0025100, -0.0090371, 0.0140555
9: -0.0125739, 0.0002662, -0.0123389, 0.0016807, -0.0142546, 0.0126050

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038806, upper bound: 0.0037592
time: 2.66 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038660, upper bound: 0.0037588
time: 2.49 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0057653, 0.0074242, 0.0056758, 0.0074191, -0.0016538, 0.0017483
1: -0.0013262, 0.0024621, -0.0021172, 0.0024522, -0.0037784, 0.0045792
2: -0.0029308, 0.0237074, -0.0028515, 0.0258807, -0.0288114, 0.0265589
3: -0.0045655, -0.0022508, -0.0046904, -0.0022579, -0.0023076, 0.0024396
4: -0.0008149, 0.0104156, -0.0021040, 0.0103813, -0.0111962, 0.0125196
5: -0.0020746, 0.0000730, -0.0020694, 0.0029825, -0.0050571, 0.0021424
6: 0.9897507, 0.9940580, 0.9869310, 0.9940486, -0.0042979, 0.0071270
7: -0.0150959, 0.0054713, -0.0164481, 0.0054091, -0.0205050, 0.0219193
8: -0.0071824, 0.0027024, -0.0113018, 0.0026830, -0.0098654, 0.0140042
9: -0.0127229, 0.0004549, -0.0126840, 0.0016443, -0.0143671, 0.0131389

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039892, upper bound: 0.0038092
time: 2.48 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039892, upper bound: 0.0038576
time: 2.48 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0057046, 0.0074111, 0.0056776, 0.0073694, -0.0016649, 0.0017334
1: -0.0018633, 0.0024366, -0.0021011, 0.0023560, -0.0042193, 0.0045378
2: -0.0027258, 0.0251831, -0.0020751, 0.0258366, -0.0285624, 0.0272582
3: -0.0046503, -0.0022691, -0.0046878, -0.0023272, -0.0023231, 0.0024187
4: -0.0012262, 0.0103268, -0.0020687, 0.0100448, -0.0112711, 0.0123955
5: -0.0020613, 0.0004779, -0.0020192, 0.0028926, -0.0049539, 0.0024972
6: 0.9887394, 0.9940338, 0.9870059, 0.9939564, -0.0052170, 0.0070279
7: -0.0160141, 0.0053105, -0.0164207, 0.0048000, -0.0208141, 0.0215534
8: -0.0099796, 0.0026521, -0.0112183, 0.0024922, -0.0124718, 0.0138704
9: -0.0126223, 0.0012605, -0.0123032, 0.0016201, -0.0142425, 0.0135637

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038313, upper bound: 0.0037713
time: 2.35 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038141, upper bound: 0.0037707
time: 2.29 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0056904, 0.0074302, 0.0056806, 0.0074136, -0.0017232, 0.0017496
1: -0.0019881, 0.0024738, -0.0020752, 0.0024415, -0.0044296, 0.0045489
2: -0.0030253, 0.0255260, -0.0027651, 0.0257653, -0.0287905, 0.0282911
3: -0.0046700, -0.0022424, -0.0046837, -0.0022656, -0.0024044, 0.0024414
4: -0.0013218, 0.0104566, -0.0020116, 0.0103439, -0.0116657, 0.0124681
5: -0.0020807, 0.0005720, -0.0020639, 0.0027469, -0.0048276, 0.0026359
6: 0.9885043, 0.9940692, 0.9871275, 0.9940383, -0.0055340, 0.0069417
7: -0.0162274, 0.0055454, -0.0163762, 0.0053413, -0.0215687, 0.0219216
8: -0.0106295, 0.0027257, -0.0110830, 0.0026617, -0.0132912, 0.0138087
9: -0.0127692, 0.0014476, -0.0126416, 0.0015810, -0.0143502, 0.0140893

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=58, inp2_unstable=54, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039460, upper bound: 0.0038281
time: 2.24 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039460, upper bound: 0.0038733
time: 2.32 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0056758, 0.0074191, 0.0057653, 0.0074242, -0.0017483, 0.0016538
1: -0.0021172, 0.0024522, -0.0013262, 0.0024621, -0.0045792, 0.0037784
2: -0.0028515, 0.0258807, -0.0029308, 0.0237074, -0.0265589, 0.0288114
3: -0.0046904, -0.0022579, -0.0045655, -0.0022508, -0.0024396, 0.0023076
4: -0.0021040, 0.0103813, -0.0008149, 0.0104156, -0.0125196, 0.0111962
5: -0.0020694, 0.0029825, -0.0020746, 0.0000730, -0.0021424, 0.0050571
6: 0.9869310, 0.9940486, 0.9897507, 0.9940580, -0.0071270, 0.0042979
7: -0.0164481, 0.0054091, -0.0150959, 0.0054713, -0.0219193, 0.0205050
8: -0.0113018, 0.0026830, -0.0071824, 0.0027024, -0.0140042, 0.0098654
9: -0.0126840, 0.0016443, -0.0127229, 0.0004549, -0.0131389, 0.0143671

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 213

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038576, upper bound: 0.0039387
time: 2.27 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038576, upper bound: 0.0039387
time: 2.44 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0056758, 0.0074191, 0.0056833, 0.0074078, -0.0017320, 0.0017358
1: -0.0021172, 0.0024522, -0.0020511, 0.0024304, -0.0045476, 0.0045034
2: -0.0028515, 0.0258807, -0.0026753, 0.0256993, -0.0285508, 0.0285560
3: -0.0046904, -0.0022579, -0.0046800, -0.0022736, -0.0024168, 0.0024221
4: -0.0021040, 0.0103813, -0.0013701, 0.0103050, -0.0124089, 0.0117514
5: -0.0020694, 0.0029825, -0.0020581, 0.0006196, -0.0026890, 0.0050406
6: 0.9869310, 0.9940486, 0.9883857, 0.9940277, -0.0070967, 0.0056629
7: -0.0164481, 0.0054091, -0.0163352, 0.0052709, -0.0217190, 0.0217443
8: -0.0113018, 0.0026830, -0.0109579, 0.0026397, -0.0139414, 0.0136409
9: -0.0126840, 0.0016443, -0.0125976, 0.0015422, -0.0142262, 0.0142419

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=54, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 213

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038576, upper bound: 0.0039387
time: 2.49 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038576, upper bound: 0.0039387
time: 2.49 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0057653, 0.0074242, 0.0056499, 0.0075929, -0.0018276, 0.0017743
1: -0.0013262, 0.0024621, -0.0023463, 0.0024811, -0.0038072, 0.0048083
2: -0.0029308, 0.0237074, -0.0030840, 0.0265102, -0.0294409, 0.0267914
3: -0.0045655, -0.0022508, -0.0047265, -0.0022371, -0.0023284, 0.0024757
4: -0.0008149, 0.0104156, -0.0026082, 0.0104821, -0.0112969, 0.0130238
5: -0.0020746, 0.0000730, -0.0020845, 0.0042681, -0.0063427, 0.0021575
6: 0.9897507, 0.9940580, 0.9858598, 0.9940761, -0.0043254, 0.0081983
7: -0.0150959, 0.0054713, -0.0168397, 0.0055915, -0.0206874, 0.0223110
8: -0.0071824, 0.0027024, -0.0124949, 0.0027401, -0.0099225, 0.0151974
9: -0.0127229, 0.0004549, -0.0127980, 0.0019893, -0.0147122, 0.0132530

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040644, upper bound: 0.0039106
time: 2.31 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040683, upper bound: 0.0039654
time: 2.57 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0057653, 0.0074242, 0.0055761, 0.0089478, -0.0031825, 0.0018480
1: -0.0013262, 0.0024621, -0.0029984, 0.0024884, -0.0038146, 0.0054605
2: -0.0029308, 0.0237074, -0.0031433, 0.0283020, -0.0312328, 0.0268506
3: -0.0045655, -0.0022508, -0.0048295, -0.0022318, -0.0023337, 0.0025787
4: -0.0008149, 0.0104156, -0.0040433, 0.0105077, -0.0113226, 0.0144590
5: -0.0020746, 0.0000730, -0.0020883, 0.0079274, -0.0100020, 0.0021613
6: 0.9897507, 0.9940580, 0.9828101, 0.9940832, -0.0043325, 0.0112479
7: -0.0150959, 0.0054713, -0.0179546, 0.0056380, -0.0207338, 0.0234258
8: -0.0071824, 0.0027024, -0.0158912, 0.0027547, -0.0099371, 0.0185937
9: -0.0127229, 0.0004549, -0.0128271, 0.0029716, -0.0156944, 0.0132820

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=57, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040644, upper bound: 0.0039106
time: 2.53 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040683, upper bound: 0.0039654
time: 2.61 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0057046, 0.0074111, 0.0056527, 0.0075422, -0.0018377, 0.0017584
1: -0.0018633, 0.0024366, -0.0023219, 0.0023997, -0.0042630, 0.0047585
2: -0.0027258, 0.0251831, -0.0024276, 0.0264431, -0.0291689, 0.0276107
3: -0.0046503, -0.0022691, -0.0047227, -0.0022958, -0.0023545, 0.0024536
4: -0.0012262, 0.0103268, -0.0025545, 0.0101976, -0.0114238, 0.0128813
5: -0.0020613, 0.0004779, -0.0020420, 0.0041312, -0.0061925, 0.0025200
6: 0.9887394, 0.9940338, 0.9859738, 0.9939983, -0.0052589, 0.0080599
7: -0.0160141, 0.0053105, -0.0167980, 0.0050765, -0.0210906, 0.0221085
8: -0.0099796, 0.0026521, -0.0123678, 0.0025788, -0.0125584, 0.0150199
9: -0.0126223, 0.0012605, -0.0124760, 0.0019526, -0.0145749, 0.0137365

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=54, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0039040, upper bound: 0.0038879
time: 2.55 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038913, upper bound: 0.0038879
time: 2.63 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0056904, 0.0074302, 0.0056466, 0.0076541, -0.0019637, 0.0017836
1: -0.0019881, 0.0024738, -0.0023757, 0.0024947, -0.0044828, 0.0048495
2: -0.0030253, 0.0255260, -0.0031943, 0.0265911, -0.0296164, 0.0287202
3: -0.0046700, -0.0022424, -0.0047312, -0.0022273, -0.0024427, 0.0024888
4: -0.0013218, 0.0104566, -0.0026730, 0.0105298, -0.0118516, 0.0131296
5: -0.0020807, 0.0005720, -0.0020916, 0.0044334, -0.0065141, 0.0026636
6: 0.9885043, 0.9940692, 0.9857219, 0.9940893, -0.0055850, 0.0083473
7: -0.0162274, 0.0055454, -0.0168901, 0.0056780, -0.0219053, 0.0224355
8: -0.0106295, 0.0027257, -0.0126484, 0.0027672, -0.0133967, 0.0153741
9: -0.0127692, 0.0014476, -0.0128521, 0.0020337, -0.0148029, 0.0142998

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=58, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040276, upper bound: 0.0039296
time: 2.35 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040276, upper bound: 0.0039803
time: 2.43 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0057653, 0.0074242, 0.0055728, 0.0090099, -0.0032446, 0.0018514
1: -0.0013262, 0.0024621, -0.0030283, 0.0024353, -0.0037615, 0.0054904
2: -0.0029308, 0.0237074, -0.0027152, 0.0283841, -0.0313149, 0.0264226
3: -0.0045655, -0.0022508, -0.0048342, -0.0022701, -0.0022954, 0.0025834
4: -0.0008149, 0.0104156, -0.0041091, 0.0103222, -0.0111371, 0.0145247
5: -0.0020746, 0.0000730, -0.0020606, 0.0080951, -0.0101697, 0.0021336
6: 0.9897507, 0.9940580, 0.9826703, 0.9940324, -0.0042817, 0.0113877
7: -0.0150959, 0.0054713, -0.0180057, 0.0053021, -0.0203980, 0.0234769
8: -0.0071824, 0.0027024, -0.0160469, 0.0026495, -0.0098319, 0.0187493
9: -0.0127229, 0.0004549, -0.0126171, 0.0030166, -0.0157394, 0.0130721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=56, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039346, upper bound: 0.0036945
time: 2.48 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039348, upper bound: 0.0037490
time: 2.47 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0057653, 0.0074242, 0.0055015, 0.0103181, -0.0045528, 0.0019227
1: -0.0013262, 0.0024621, -0.0036580, 0.0024405, -0.0037667, 0.0061201
2: -0.0029308, 0.0237074, -0.0027568, 0.0301142, -0.0330449, 0.0264642
3: -0.0045655, -0.0022508, -0.0049336, -0.0016650, -0.0029005, 0.0026828
4: -0.0008149, 0.0104156, -0.0054948, 0.0103402, -0.0111551, 0.0159104
5: -0.0020746, 0.0000730, -0.0020633, 0.0116283, -0.0137028, 0.0021363
6: 0.9897507, 0.9940580, 0.9797258, 0.9940374, -0.0042866, 0.0143322
7: -0.0150959, 0.0054713, -0.0190821, 0.0053348, -0.0204307, 0.0245533
8: -0.0071824, 0.0027024, -0.0193261, 0.0026597, -0.0098421, 0.0220286
9: -0.0127229, 0.0004549, -0.0126375, 0.0039650, -0.0166878, 0.0130925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=57, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039346, upper bound: 0.0036946
time: 2.47 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039348, upper bound: 0.0037490
time: 2.49 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0056499, 0.0075929, 0.0057654, 0.0074242, -0.0017743, 0.0018276
1: -0.0023463, 0.0024811, -0.0013258, 0.0024621, -0.0048083, 0.0038068
2: -0.0030840, 0.0265102, -0.0029308, 0.0237063, -0.0267903, 0.0294409
3: -0.0047265, -0.0022371, -0.0045655, -0.0022508, -0.0024757, 0.0023283
4: -0.0026082, 0.0104821, -0.0008146, 0.0104156, -0.0130238, 0.0112966
5: -0.0020845, 0.0042681, -0.0020746, 0.0000727, -0.0021572, 0.0063427
6: 0.9858598, 0.9940761, 0.9897515, 0.9940580, -0.0081983, 0.0043247
7: -0.0168397, 0.0055915, -0.0150952, 0.0054713, -0.0223110, 0.0206867
8: -0.0124949, 0.0027401, -0.0071803, 0.0027024, -0.0151974, 0.0099205
9: -0.0127980, 0.0019893, -0.0127229, 0.0004543, -0.0132524, 0.0147122

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040105, upper bound: 0.0039632
time: 2.28 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040162, upper bound: 0.0040139
time: 2.30 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0056499, 0.0075929, 0.0056908, 0.0074302, -0.0017803, 0.0019021
1: -0.0023463, 0.0024811, -0.0019845, 0.0024738, -0.0048200, 0.0044655
2: -0.0030840, 0.0265102, -0.0030253, 0.0255160, -0.0286001, 0.0295354
3: -0.0047265, -0.0022371, -0.0046694, -0.0022424, -0.0024842, 0.0024323
4: -0.0026082, 0.0104821, -0.0013190, 0.0104566, -0.0130648, 0.0118011
5: -0.0020845, 0.0042681, -0.0020807, 0.0005693, -0.0026538, 0.0063488
6: 0.9858598, 0.9940761, 0.9885111, 0.9940692, -0.0082095, 0.0055650
7: -0.0168397, 0.0055915, -0.0162212, 0.0055454, -0.0223851, 0.0218127
8: -0.0124949, 0.0027401, -0.0106106, 0.0027257, -0.0152206, 0.0133508
9: -0.0127980, 0.0019893, -0.0127692, 0.0014422, -0.0142403, 0.0147586

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=58, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040105, upper bound: 0.0039633
time: 2.27 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040162, upper bound: 0.0040140
time: 2.25 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0055918, 0.0086610, 0.0057655, 0.0073850, -0.0017933, 0.0028955
1: -0.0028604, 0.0024505, -0.0013245, 0.0023863, -0.0052466, 0.0037750
2: -0.0028380, 0.0279227, -0.0023194, 0.0237028, -0.0265407, 0.0302421
3: -0.0048077, -0.0022591, -0.0045652, -0.0023054, -0.0025023, 0.0023061
4: -0.0037395, 0.0103754, -0.0008136, 0.0101507, -0.0138902, 0.0111890
5: -0.0020686, 0.0071527, -0.0020350, 0.0000717, -0.0021403, 0.0091878
6: 0.9834556, 0.9940470, 0.9897540, 0.9939854, -0.0105298, 0.0042930
7: -0.0177186, 0.0053985, -0.0150930, 0.0049917, -0.0227103, 0.0204915
8: -0.0151723, 0.0026796, -0.0071737, 0.0025522, -0.0177245, 0.0098533
9: -0.0126773, 0.0027636, -0.0124230, 0.0004524, -0.0131298, 0.0151867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=54, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038541, upper bound: 0.0039371
time: 1.98 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038398, upper bound: 0.0039371
time: 2.30 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0055761, 0.0089478, 0.0057619, 0.0074311, -0.0018550, 0.0031859
1: -0.0029984, 0.0024884, -0.0013563, 0.0024754, -0.0054739, 0.0038447
2: -0.0031433, 0.0283020, -0.0030388, 0.0237903, -0.0269335, 0.0313408
3: -0.0048295, -0.0022318, -0.0045703, -0.0022412, -0.0025883, 0.0023384
4: -0.0040433, 0.0105077, -0.0008380, 0.0104625, -0.0145058, 0.0113457
5: -0.0020883, 0.0079274, -0.0020816, 0.0000957, -0.0021841, 0.0100090
6: 0.9828101, 0.9940832, 0.9896939, 0.9940708, -0.0112607, 0.0043893
7: -0.0179546, 0.0056380, -0.0151474, 0.0055560, -0.0235106, 0.0207854
8: -0.0158912, 0.0027547, -0.0073395, 0.0027290, -0.0186202, 0.0100942
9: -0.0128271, 0.0029716, -0.0127759, 0.0005002, -0.0133273, 0.0157474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=57, inp2_unstable=56, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039761, upper bound: 0.0039830
time: 2.38 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039761, upper bound: 0.0040303
time: 2.56 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0056499, 0.0075929, 0.0056499, 0.0074340, -0.0017841, 0.0019430
1: -0.0023463, 0.0024811, -0.0023460, 0.0024811, -0.0048273, 0.0048271
2: -0.0030840, 0.0265102, -0.0030840, 0.0265095, -0.0295935, 0.0295942
3: -0.0047265, -0.0022371, -0.0047265, -0.0022371, -0.0024894, 0.0024894
4: -0.0026082, 0.0104821, -0.0015960, 0.0104821, -0.0130902, 0.0120780
5: -0.0020845, 0.0042681, -0.0020845, 0.0008419, -0.0029264, 0.0063526
6: 0.9858598, 0.9940761, 0.9878303, 0.9940761, -0.0082164, 0.0062459
7: -0.0168397, 0.0055915, -0.0168393, 0.0055915, -0.0224312, 0.0224308
8: -0.0124949, 0.0027401, -0.0124937, 0.0027401, -0.0152350, 0.0152338
9: -0.0127980, 0.0019893, -0.0127980, 0.0019845, -0.0147826, 0.0147874

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040105, upper bound: 0.0039336
time: 2.50 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040162, upper bound: 0.0039889
time: 2.44 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0056499, 0.0075929, 0.0055776, 0.0074378, -0.0017879, 0.0020154
1: -0.0023463, 0.0024811, -0.0029859, 0.0024884, -0.0048347, 0.0054669
2: -0.0030840, 0.0265102, -0.0031433, 0.0282675, -0.0313516, 0.0296534
3: -0.0047265, -0.0022371, -0.0048275, -0.0022318, -0.0024947, 0.0025904
4: -0.0026082, 0.0104821, -0.0020860, 0.0105077, -0.0131159, 0.0125681
5: -0.0020845, 0.0042681, -0.0020883, 0.0013243, -0.0034088, 0.0063564
6: 0.9858598, 0.9940761, 0.9866256, 0.9940832, -0.0082235, 0.0074506
7: -0.0168397, 0.0055915, -0.0179331, 0.0056380, -0.0224777, 0.0235246
8: -0.0124949, 0.0027401, -0.0158259, 0.0027547, -0.0152496, 0.0185660
9: -0.0127980, 0.0019893, -0.0128271, 0.0029442, -0.0157422, 0.0148164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=57, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040105, upper bound: 0.0039334
time: 2.31 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040162, upper bound: 0.0039889
time: 2.35 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0055918, 0.0086610, 0.0056527, 0.0073920, -0.0018002, 0.0030083
1: -0.0028604, 0.0024505, -0.0023219, 0.0023997, -0.0052600, 0.0047724
2: -0.0028380, 0.0279227, -0.0024275, 0.0264431, -0.0292811, 0.0303502
3: -0.0048077, -0.0022591, -0.0047227, -0.0022958, -0.0025119, 0.0024636
4: -0.0037395, 0.0103754, -0.0015775, 0.0101976, -0.0139371, 0.0119529
5: -0.0020686, 0.0071527, -0.0020420, 0.0008237, -0.0028922, 0.0091948
6: 0.9834556, 0.9940470, 0.9878759, 0.9939983, -0.0105427, 0.0061711
7: -0.0177186, 0.0053985, -0.0167980, 0.0050765, -0.0227951, 0.0221965
8: -0.0151723, 0.0026796, -0.0123678, 0.0025788, -0.0177510, 0.0150474
9: -0.0126773, 0.0027636, -0.0124760, 0.0019483, -0.0146256, 0.0152397

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=54, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038541, upper bound: 0.0039113
time: 2.57 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038398, upper bound: 0.0039112
time: 2.40 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0055761, 0.0089478, 0.0056466, 0.0074410, -0.0018649, 0.0033012
1: -0.0029984, 0.0024884, -0.0023757, 0.0024947, -0.0054932, 0.0048641
2: -0.0031433, 0.0283020, -0.0031943, 0.0265911, -0.0297344, 0.0314963
3: -0.0048295, -0.0022318, -0.0047312, -0.0022273, -0.0026022, 0.0024994
4: -0.0040433, 0.0105077, -0.0016187, 0.0105298, -0.0145731, 0.0121264
5: -0.0020883, 0.0079274, -0.0020916, 0.0008643, -0.0029526, 0.0100190
6: 0.9828101, 0.9940832, 0.9877744, 0.9940893, -0.0112792, 0.0063088
7: -0.0179546, 0.0056380, -0.0168901, 0.0056780, -0.0236325, 0.0225280
8: -0.0158912, 0.0027547, -0.0126484, 0.0027672, -0.0186584, 0.0154031
9: -0.0128271, 0.0029716, -0.0128521, 0.0020291, -0.0148562, 0.0158237

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=57, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039761, upper bound: 0.0039523
time: 2.35 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039761, upper bound: 0.0040061
time: 2.22 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 5.94 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0041034, upper bound: 0.0039999
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0041060, upper bound: 0.0040482
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0041034, upper bound: 0.0039999
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0041060, upper bound: 0.0040482
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0039378, upper bound: 0.0039708
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0039241, upper bound: 0.0039707
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0040637, upper bound: 0.0040190
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0040637, upper bound: 0.0040657
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0038806, upper bound: 0.0037592
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0038660, upper bound: 0.0037588
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0039892, upper bound: 0.0038092
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0039892, upper bound: 0.0038576
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0038313, upper bound: 0.0037713
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0038141, upper bound: 0.0037707
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0039460, upper bound: 0.0038281
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0039460, upper bound: 0.0038733
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0038576, upper bound: 0.0039387
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0038576, upper bound: 0.0039387
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0038576, upper bound: 0.0039387
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0038576, upper bound: 0.0039387
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0040644, upper bound: 0.0039106
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0040683, upper bound: 0.0039654
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0040644, upper bound: 0.0039106
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0040683, upper bound: 0.0039654
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0039040, upper bound: 0.0038879
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0038913, upper bound: 0.0038879
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0040276, upper bound: 0.0039296
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0040276, upper bound: 0.0039803
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0039346, upper bound: 0.0036945
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0039348, upper bound: 0.0037490
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0039346, upper bound: 0.0036946
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0039348, upper bound: 0.0037490
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0040105, upper bound: 0.0039632
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0040162, upper bound: 0.0040139
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0040105, upper bound: 0.0039633
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0040162, upper bound: 0.0040140
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0038541, upper bound: 0.0039371
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0038398, upper bound: 0.0039371
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0039761, upper bound: 0.0039830
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0039761, upper bound: 0.0040303
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0040105, upper bound: 0.0039336
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0040162, upper bound: 0.0039889
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0040105, upper bound: 0.0039334
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0040162, upper bound: 0.0039889
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0038541, upper bound: 0.0039113
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0038398, upper bound: 0.0039112
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0039761, upper bound: 0.0039523
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.94
Output dim: 6, lower bound: -0.0039761, upper bound: 0.0040061

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0057781, 0.0073610, 0.0057796, 0.0074047, -0.0016266, 0.0015815
1: -0.0012130, 0.0023397, -0.0012003, 0.0024244, -0.0036374, 0.0035400
2: -0.0019438, 0.0233963, -0.0026271, 0.0233616, -0.0253054, 0.0260234
3: -0.0045476, -0.0023390, -0.0045456, -0.0022779, -0.0022697, 0.0022067
4: -0.0007282, 0.0099879, -0.0007185, 0.0102840, -0.0110122, 0.0107064
5: -0.0020107, -0.0000124, -0.0020549, -0.0000219, -0.0019889, 0.0020426
6: 0.9899639, 0.9939409, 0.9899877, 0.9940220, -0.0040580, 0.0039532
7: -0.0149023, 0.0046971, -0.0148807, 0.0052330, -0.0197895, 0.0195778
8: -0.0065928, 0.0024599, -0.0065270, 0.0026278, -0.0092206, 0.0089869
9: -0.0122388, 0.0002851, -0.0125739, 0.0002662, -0.0125049, 0.0128590

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=54, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040540, upper bound: 0.0040210
time: 1.94 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040540, upper bound: 0.0040078
time: 2.03 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0057753, 0.0074082, 0.0057653, 0.0074242, -0.0016489, 0.0016428
1: -0.0012381, 0.0024310, -0.0013262, 0.0024621, -0.0037001, 0.0037572
2: -0.0026806, 0.0234654, -0.0029308, 0.0237074, -0.0263880, 0.0263961
3: -0.0045516, -0.0022731, -0.0045655, -0.0022508, -0.0023008, 0.0022924
4: -0.0007474, 0.0103073, -0.0008149, 0.0104156, -0.0111631, 0.0111221
5: -0.0020584, 0.0000066, -0.0020746, 0.0000730, -0.0021314, 0.0020812
6: 0.9899166, 0.9940283, 0.9897507, 0.9940580, -0.0041414, 0.0042775
7: -0.0149453, 0.0052751, -0.0150959, 0.0054713, -0.0204166, 0.0203709
8: -0.0067237, 0.0026410, -0.0071824, 0.0027024, -0.0094262, 0.0098234
9: -0.0126002, 0.0003228, -0.0127229, 0.0004549, -0.0130551, 0.0130457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040984, upper bound: 0.0041433
time: 2.41 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040984, upper bound: 0.0041472
time: 2.37 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0057781, 0.0073610, 0.0057046, 0.0074111, -0.0016330, 0.0016565
1: -0.0012130, 0.0023397, -0.0018633, 0.0024366, -0.0036496, 0.0042030
2: -0.0019438, 0.0233963, -0.0027258, 0.0251831, -0.0271269, 0.0261221
3: -0.0045476, -0.0023390, -0.0046503, -0.0022691, -0.0022785, 0.0023113
4: -0.0007282, 0.0099879, -0.0012262, 0.0103268, -0.0110550, 0.0112142
5: -0.0020107, -0.0000124, -0.0020613, 0.0004779, -0.0024887, 0.0020490
6: 0.9899639, 0.9939409, 0.9887394, 0.9940338, -0.0040698, 0.0052015
7: -0.0149023, 0.0046971, -0.0160141, 0.0053105, -0.0199343, 0.0207111
8: -0.0065928, 0.0024599, -0.0099796, 0.0026521, -0.0092449, 0.0124395
9: -0.0122388, 0.0002851, -0.0126223, 0.0012605, -0.0134993, 0.0129075

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040130, upper bound: 0.0039186
time: 2.55 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040130, upper bound: 0.0039053
time: 2.56 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0057753, 0.0074082, 0.0056904, 0.0074302, -0.0016549, 0.0017177
1: -0.0012381, 0.0024310, -0.0019881, 0.0024738, -0.0037119, 0.0044191
2: -0.0026806, 0.0234654, -0.0030253, 0.0255260, -0.0282066, 0.0264906
3: -0.0045516, -0.0022731, -0.0046700, -0.0022424, -0.0023092, 0.0023969
4: -0.0007474, 0.0103073, -0.0013218, 0.0104566, -0.0112040, 0.0116291
5: -0.0020584, 0.0000066, -0.0020807, 0.0005720, -0.0026304, 0.0020873
6: 0.9899166, 0.9940283, 0.9885043, 0.9940692, -0.0041526, 0.0055240
7: -0.0149453, 0.0052751, -0.0162274, 0.0055454, -0.0204907, 0.0215024
8: -0.0067237, 0.0026410, -0.0106295, 0.0027257, -0.0094494, 0.0132704
9: -0.0126002, 0.0003228, -0.0127692, 0.0014476, -0.0140478, 0.0130920

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=58, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040634, upper bound: 0.0040467
time: 2.42 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040634, upper bound: 0.0040482
time: 2.60 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0057486, 0.0074079, 0.0057723, 0.0073846, -0.0016360, 0.0016356
1: -0.0014736, 0.0024305, -0.0012645, 0.0023854, -0.0038590, 0.0036950
2: -0.0026765, 0.0241125, -0.0023126, 0.0235378, -0.0262143, 0.0264251
3: -0.0045888, -0.0022735, -0.0045558, -0.0023060, -0.0022828, 0.0022823
4: -0.0009278, 0.0103055, -0.0007676, 0.0101478, -0.0110756, 0.0110731
5: -0.0020581, 0.0001842, -0.0020346, 0.0000265, -0.0020846, 0.0022188
6: 0.9894731, 0.9940279, 0.9898670, 0.9939846, -0.0045115, 0.0041609
7: -0.0153479, 0.0052718, -0.0149904, 0.0049863, -0.0203343, 0.0200876
8: -0.0079503, 0.0026400, -0.0068610, 0.0025505, -0.0105009, 0.0095010
9: -0.0125982, 0.0006761, -0.0124197, 0.0003624, -0.0129605, 0.0130957

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039186, upper bound: 0.0039707
time: 2.32 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039186, upper bound: 0.0039706
time: 2.46 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0057558, 0.0074237, 0.0057825, 0.0073841, -0.0016283, 0.0016413
1: -0.0014106, 0.0024612, -0.0011745, 0.0023843, -0.0037949, 0.0036357
2: -0.0029239, 0.0239393, -0.0023040, 0.0232907, -0.0262145, 0.0262433
3: -0.0045788, -0.0022514, -0.0045416, -0.0023068, -0.0022721, 0.0022902
4: -0.0008795, 0.0104127, -0.0006987, 0.0101440, -0.0110236, 0.0111114
5: -0.0020741, 0.0001366, -0.0020340, -0.0000413, -0.0020328, 0.0021707
6: 0.9895918, 0.9940572, 0.9900364, 0.9939836, -0.0043918, 0.0040208
7: -0.0152402, 0.0054659, -0.0148366, 0.0049796, -0.0202198, 0.0203025
8: -0.0076221, 0.0027008, -0.0063926, 0.0025484, -0.0101705, 0.0090933
9: -0.0127195, 0.0005815, -0.0124154, 0.0002275, -0.0129469, 0.0129970

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039053, upper bound: 0.0039706
time: 2.28 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039053, upper bound: 0.0039707
time: 2.76 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0057036, 0.0073688, 0.0057619, 0.0074311, -0.0017276, 0.0016069
1: -0.0018721, 0.0023547, -0.0013568, 0.0024754, -0.0043476, 0.0037115
2: -0.0020651, 0.0252075, -0.0030388, 0.0237914, -0.0258564, 0.0282463
3: -0.0046517, -0.0023281, -0.0045703, -0.0022412, -0.0024105, 0.0022422
4: -0.0012330, 0.0100405, -0.0008383, 0.0104625, -0.0116955, 0.0108788
5: -0.0020186, 0.0004846, -0.0020816, 0.0000960, -0.0021146, 0.0025662
6: 0.9887227, 0.9939553, 0.9896932, 0.9940708, -0.0053480, 0.0042621
7: -0.0160292, 0.0047922, -0.0151481, 0.0055560, -0.0213634, 0.0199403
8: -0.0100258, 0.0024897, -0.0073416, 0.0027290, -0.0127548, 0.0098313
9: -0.0122983, 0.0012738, -0.0127759, 0.0005008, -0.0127990, 0.0140497

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=56, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039999, upper bound: 0.0040190
time: 2.35 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039999, upper bound: 0.0040191
time: 2.39 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0057005, 0.0074138, 0.0057619, 0.0074311, -0.0017306, 0.0016520
1: -0.0018990, 0.0024420, -0.0013568, 0.0024754, -0.0043744, 0.0037988
2: -0.0027691, 0.0252812, -0.0030388, 0.0237914, -0.0265604, 0.0283200
3: -0.0046559, -0.0022653, -0.0045703, -0.0022412, -0.0024148, 0.0023051
4: -0.0012536, 0.0103456, -0.0008383, 0.0104625, -0.0117160, 0.0111839
5: -0.0020641, 0.0005049, -0.0020816, 0.0000960, -0.0021601, 0.0025864
6: 0.9886721, 0.9940388, 0.9896932, 0.9940708, -0.0053986, 0.0043457
7: -0.0160751, 0.0053444, -0.0151481, 0.0055560, -0.0216311, 0.0204926
8: -0.0101656, 0.0026627, -0.0073416, 0.0027290, -0.0128946, 0.0100043
9: -0.0126436, 0.0013141, -0.0127759, 0.0005008, -0.0131443, 0.0140899

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=56, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039999, upper bound: 0.0040657
time: 2.38 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039999, upper bound: 0.0040657
time: 2.67 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0057781, 0.0073610, 0.0056758, 0.0074191, -0.0016410, 0.0016852
1: -0.0012130, 0.0023397, -0.0021172, 0.0024522, -0.0036652, 0.0044569
2: -0.0019438, 0.0233963, -0.0028515, 0.0258807, -0.0278244, 0.0262478
3: -0.0045476, -0.0023390, -0.0046904, -0.0022579, -0.0022898, 0.0023514
4: -0.0007282, 0.0099879, -0.0021040, 0.0103813, -0.0111095, 0.0120919
5: -0.0020107, -0.0000124, -0.0020694, 0.0029825, -0.0049933, 0.0020571
6: 0.9899639, 0.9939409, 0.9869310, 0.9940486, -0.0040846, 0.0070099
7: -0.0149023, 0.0046971, -0.0164481, 0.0054091, -0.0203114, 0.0211451
8: -0.0065928, 0.0024599, -0.0113018, 0.0026830, -0.0092758, 0.0137617
9: -0.0122388, 0.0002851, -0.0126840, 0.0016443, -0.0138830, 0.0129691

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039579, upper bound: 0.0038092
time: 2.48 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039579, upper bound: 0.0038093
time: 2.68 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0057753, 0.0074082, 0.0056758, 0.0074191, -0.0016438, 0.0017323
1: -0.0012381, 0.0024310, -0.0021172, 0.0024522, -0.0036903, 0.0045482
2: -0.0026806, 0.0234654, -0.0028515, 0.0258807, -0.0285613, 0.0263169
3: -0.0045516, -0.0022731, -0.0046904, -0.0022579, -0.0022937, 0.0024172
4: -0.0007474, 0.0103073, -0.0021040, 0.0103813, -0.0111287, 0.0124112
5: -0.0020584, 0.0000066, -0.0020694, 0.0029825, -0.0050409, 0.0020760
6: 0.9899166, 0.9940283, 0.9869310, 0.9940486, -0.0041320, 0.0070973
7: -0.0149453, 0.0052751, -0.0164481, 0.0054091, -0.0203544, 0.0217231
8: -0.0067237, 0.0026410, -0.0113018, 0.0026830, -0.0094067, 0.0139427
9: -0.0126002, 0.0003228, -0.0126840, 0.0016443, -0.0142445, 0.0130068

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039579, upper bound: 0.0038576
time: 2.42 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039579, upper bound: 0.0038575
time: 2.36 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0057036, 0.0073688, 0.0056806, 0.0074136, -0.0017100, 0.0016882
1: -0.0018721, 0.0023547, -0.0020752, 0.0024415, -0.0043137, 0.0044299
2: -0.0020651, 0.0252075, -0.0027651, 0.0257653, -0.0278303, 0.0279726
3: -0.0046517, -0.0023281, -0.0046837, -0.0022656, -0.0023861, 0.0023556
4: -0.0012330, 0.0100405, -0.0020116, 0.0103439, -0.0115769, 0.0120521
5: -0.0020186, 0.0004846, -0.0020639, 0.0027469, -0.0047654, 0.0025485
6: 0.9887227, 0.9939553, 0.9871275, 0.9940383, -0.0053155, 0.0068277
7: -0.0160292, 0.0047922, -0.0163762, 0.0053413, -0.0213705, 0.0211684
8: -0.0100258, 0.0024897, -0.0110830, 0.0026617, -0.0126875, 0.0135727
9: -0.0122983, 0.0012738, -0.0126416, 0.0015810, -0.0138793, 0.0139154

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=54, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 213

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038155, upper bound: 0.0037384
time: 2.50 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038141, upper bound: 0.0037203
time: 2.33 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0057005, 0.0074138, 0.0056806, 0.0074136, -0.0017131, 0.0017333
1: -0.0018990, 0.0024420, -0.0020752, 0.0024415, -0.0043405, 0.0045172
2: -0.0027691, 0.0252812, -0.0027651, 0.0257653, -0.0285343, 0.0280464
3: -0.0046559, -0.0022653, -0.0046837, -0.0022656, -0.0023903, 0.0024185
4: -0.0012536, 0.0103456, -0.0020116, 0.0103439, -0.0115975, 0.0123571
5: -0.0020641, 0.0005049, -0.0020639, 0.0027469, -0.0048110, 0.0025687
6: 0.9886721, 0.9940388, 0.9871275, 0.9940383, -0.0053661, 0.0069113
7: -0.0160751, 0.0053444, -0.0163762, 0.0053413, -0.0214164, 0.0217207
8: -0.0101656, 0.0026627, -0.0110830, 0.0026617, -0.0128273, 0.0137457
9: -0.0126436, 0.0013141, -0.0126416, 0.0015810, -0.0142246, 0.0139557

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=54, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 213

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038155, upper bound: 0.0037919
time: 2.54 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038141, upper bound: 0.0037740
time: 2.31 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0056933, 0.0073912, 0.0057653, 0.0074242, -0.0017309, 0.0016259
1: -0.0019629, 0.0023982, -0.0013262, 0.0024621, -0.0044249, 0.0037244
2: -0.0024161, 0.0254567, -0.0029308, 0.0237074, -0.0261235, 0.0283874
3: -0.0046660, -0.0022968, -0.0045655, -0.0022508, -0.0024152, 0.0022687
4: -0.0017644, 0.0101926, -0.0008149, 0.0104156, -0.0121800, 0.0110075
5: -0.0020413, 0.0021167, -0.0020746, 0.0000730, -0.0021143, 0.0041913
6: 0.9876527, 0.9939969, 0.9897507, 0.9940580, -0.0064054, 0.0042462
7: -0.0161843, 0.0050675, -0.0150959, 0.0054713, -0.0216555, 0.0201634
8: -0.0104981, 0.0025760, -0.0071824, 0.0027024, -0.0132006, 0.0097584
9: -0.0124704, 0.0014119, -0.0127229, 0.0004549, -0.0129254, 0.0141347

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038093, upper bound: 0.0039891
time: 2.07 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038093, upper bound: 0.0039900
time: 2.62 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0056212, 0.0081196, 0.0057653, 0.0074242, -0.0018029, 0.0023543
1: -0.0025998, 0.0024077, -0.0013262, 0.0024621, -0.0050618, 0.0037338
2: -0.0024920, 0.0272067, -0.0029308, 0.0237074, -0.0261994, 0.0301375
3: -0.0047666, -0.0022900, -0.0045655, -0.0022508, -0.0025158, 0.0022755
4: -0.0031661, 0.0102255, -0.0008149, 0.0104156, -0.0135817, 0.0110404
5: -0.0020462, 0.0056906, -0.0020746, 0.0000730, -0.0021192, 0.0077652
6: 0.9846743, 0.9940059, 0.9897507, 0.9940580, -0.0093837, 0.0042551
7: -0.0172731, 0.0051271, -0.0150959, 0.0054713, -0.0227444, 0.0202229
8: -0.0138152, 0.0025946, -0.0071824, 0.0027024, -0.0165176, 0.0097770
9: -0.0125076, 0.0023712, -0.0127229, 0.0004549, -0.0129626, 0.0150940

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038093, upper bound: 0.0039892
time: 2.03 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038093, upper bound: 0.0039900
time: 2.38 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0056933, 0.0073912, 0.0056833, 0.0074078, -0.0017145, 0.0017079
1: -0.0019629, 0.0023982, -0.0020511, 0.0024304, -0.0043933, 0.0044494
2: -0.0024161, 0.0254567, -0.0026753, 0.0256993, -0.0281153, 0.0281320
3: -0.0046660, -0.0022968, -0.0046800, -0.0022736, -0.0023924, 0.0023832
4: -0.0017644, 0.0101926, -0.0013701, 0.0103050, -0.0120694, 0.0115627
5: -0.0020413, 0.0021167, -0.0020581, 0.0006196, -0.0026608, 0.0041748
6: 0.9876527, 0.9939969, 0.9883857, 0.9940277, -0.0063750, 0.0056112
7: -0.0161843, 0.0050675, -0.0163352, 0.0052709, -0.0214552, 0.0214027
8: -0.0104981, 0.0025760, -0.0109579, 0.0026397, -0.0131378, 0.0135339
9: -0.0124704, 0.0014119, -0.0125976, 0.0015422, -0.0140127, 0.0140094

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=54, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 213

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038093, upper bound: 0.0039382
time: 2.55 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038093, upper bound: 0.0039387
time: 2.47 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0056212, 0.0081196, 0.0056833, 0.0074078, -0.0017866, 0.0024363
1: -0.0025998, 0.0024077, -0.0020511, 0.0024304, -0.0050302, 0.0044588
2: -0.0024920, 0.0272067, -0.0026753, 0.0256993, -0.0281912, 0.0298821
3: -0.0047666, -0.0022900, -0.0046800, -0.0022736, -0.0024929, 0.0023899
4: -0.0031661, 0.0102255, -0.0013701, 0.0103050, -0.0134710, 0.0115956
5: -0.0020462, 0.0056906, -0.0020581, 0.0006196, -0.0026657, 0.0077486
6: 0.9846743, 0.9940059, 0.9883857, 0.9940277, -0.0093534, 0.0056202
7: -0.0172731, 0.0051271, -0.0163352, 0.0052709, -0.0225440, 0.0214623
8: -0.0138152, 0.0025946, -0.0109579, 0.0026397, -0.0164549, 0.0135525
9: -0.0125076, 0.0023712, -0.0125976, 0.0015422, -0.0140499, 0.0149688

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=54, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 213

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038093, upper bound: 0.0039383
time: 2.20 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038093, upper bound: 0.0039387
time: 2.45 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0057781, 0.0073610, 0.0056643, 0.0074140, -0.0016359, 0.0016967
1: -0.0012130, 0.0023397, -0.0022193, 0.0024424, -0.0036554, 0.0045590
2: -0.0019438, 0.0233963, -0.0027726, 0.0261612, -0.0281050, 0.0261689
3: -0.0045476, -0.0023390, -0.0047065, -0.0022649, -0.0022827, 0.0023675
4: -0.0007282, 0.0099879, -0.0023287, 0.0103471, -0.0110752, 0.0123166
5: -0.0020107, -0.0000124, -0.0020643, 0.0035555, -0.0055662, 0.0020520
6: 0.9899639, 0.9939409, 0.9864536, 0.9940392, -0.0040753, 0.0074873
7: -0.0149023, 0.0046971, -0.0166226, 0.0053471, -0.0202495, 0.0213197
8: -0.0065928, 0.0024599, -0.0118335, 0.0026636, -0.0092564, 0.0142934
9: -0.0122388, 0.0002851, -0.0126453, 0.0017981, -0.0140368, 0.0129304

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040102, upper bound: 0.0039118
time: 2.04 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040102, upper bound: 0.0038990
time: 2.48 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0057753, 0.0074082, 0.0056499, 0.0075929, -0.0018177, 0.0017582
1: -0.0012381, 0.0024310, -0.0023463, 0.0024811, -0.0037192, 0.0047773
2: -0.0026806, 0.0234654, -0.0030840, 0.0265102, -0.0291908, 0.0265494
3: -0.0045516, -0.0022731, -0.0047265, -0.0022371, -0.0023145, 0.0024534
4: -0.0007474, 0.0103073, -0.0026082, 0.0104821, -0.0112295, 0.0129154
5: -0.0020584, 0.0000066, -0.0020845, 0.0042681, -0.0063265, 0.0020911
6: 0.9899166, 0.9940283, 0.9858598, 0.9940761, -0.0041595, 0.0081685
7: -0.0149453, 0.0052751, -0.0168397, 0.0055915, -0.0205368, 0.0221148
8: -0.0067237, 0.0026410, -0.0124949, 0.0027401, -0.0094638, 0.0151359
9: -0.0126002, 0.0003228, -0.0127980, 0.0019893, -0.0145895, 0.0131209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040514, upper bound: 0.0040370
time: 2.48 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040514, upper bound: 0.0040464
time: 2.96 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0057781, 0.0073610, 0.0055918, 0.0086610, -0.0028829, 0.0017693
1: -0.0012130, 0.0023397, -0.0028604, 0.0024505, -0.0036635, 0.0052001
2: -0.0019438, 0.0233963, -0.0028380, 0.0279227, -0.0298665, 0.0262343
3: -0.0045476, -0.0023390, -0.0048077, -0.0022591, -0.0022885, 0.0024687
4: -0.0007282, 0.0099879, -0.0037395, 0.0103754, -0.0111036, 0.0137274
5: -0.0020107, -0.0000124, -0.0020686, 0.0071527, -0.0091635, 0.0020562
6: 0.9899639, 0.9939409, 0.9834556, 0.9940470, -0.0040830, 0.0104853
7: -0.0149023, 0.0046971, -0.0177186, 0.0053985, -0.0203008, 0.0224156
8: -0.0065928, 0.0024599, -0.0151723, 0.0026796, -0.0092725, 0.0176322
9: -0.0122388, 0.0002851, -0.0126773, 0.0027636, -0.0150024, 0.0129625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=56, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039737, upper bound: 0.0038330
time: 3.06 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039737, upper bound: 0.0038198
time: 2.70 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0057753, 0.0074082, 0.0055761, 0.0089478, -0.0031725, 0.0018320
1: -0.0012381, 0.0024310, -0.0029984, 0.0024884, -0.0037265, 0.0054295
2: -0.0026806, 0.0234654, -0.0031433, 0.0283020, -0.0309826, 0.0266086
3: -0.0045516, -0.0022731, -0.0048295, -0.0022318, -0.0023198, 0.0025563
4: -0.0007474, 0.0103073, -0.0040433, 0.0105077, -0.0112551, 0.0143506
5: -0.0020584, 0.0000066, -0.0020883, 0.0079274, -0.0099858, 0.0020949
6: 0.9899166, 0.9940283, 0.9828101, 0.9940832, -0.0041666, 0.0112182
7: -0.0149453, 0.0052751, -0.0179546, 0.0056380, -0.0205833, 0.0232296
8: -0.0067237, 0.0026410, -0.0158912, 0.0027547, -0.0094784, 0.0185322
9: -0.0126002, 0.0003228, -0.0128271, 0.0029716, -0.0155718, 0.0131499

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=57, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040223, upper bound: 0.0039622
time: 2.22 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040223, upper bound: 0.0039654
time: 2.59 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0057036, 0.0073688, 0.0056466, 0.0076541, -0.0019506, 0.0017222
1: -0.0018721, 0.0023547, -0.0023757, 0.0024947, -0.0043669, 0.0047305
2: -0.0020651, 0.0252075, -0.0031943, 0.0265911, -0.0286562, 0.0284017
3: -0.0046517, -0.0023281, -0.0047312, -0.0022273, -0.0024244, 0.0024031
4: -0.0012330, 0.0100405, -0.0026730, 0.0105298, -0.0117628, 0.0127135
5: -0.0020186, 0.0004846, -0.0020916, 0.0044334, -0.0064520, 0.0025762
6: 0.9887227, 0.9939553, 0.9857219, 0.9940893, -0.0053666, 0.0082333
7: -0.0160292, 0.0047922, -0.0168901, 0.0056780, -0.0217072, 0.0216823
8: -0.0100258, 0.0024897, -0.0126484, 0.0027672, -0.0127930, 0.0151381
9: -0.0122983, 0.0012738, -0.0128521, 0.0020337, -0.0143320, 0.0141259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039633, upper bound: 0.0039296
time: 2.36 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039633, upper bound: 0.0039295
time: 2.46 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0057005, 0.0074138, 0.0056466, 0.0076541, -0.0019536, 0.0017673
1: -0.0018990, 0.0024420, -0.0023757, 0.0024947, -0.0043937, 0.0048177
2: -0.0027691, 0.0252812, -0.0031943, 0.0265911, -0.0293602, 0.0284755
3: -0.0046559, -0.0022653, -0.0047312, -0.0022273, -0.0024287, 0.0024659
4: -0.0012536, 0.0103456, -0.0026730, 0.0105298, -0.0117834, 0.0130186
5: -0.0020641, 0.0005049, -0.0020916, 0.0044334, -0.0064976, 0.0025965
6: 0.9886721, 0.9940388, 0.9857219, 0.9940893, -0.0054172, 0.0083169
7: -0.0160751, 0.0053444, -0.0168901, 0.0056780, -0.0217531, 0.0222345
8: -0.0101656, 0.0026627, -0.0126484, 0.0027672, -0.0129328, 0.0153111
9: -0.0126436, 0.0013141, -0.0128521, 0.0020337, -0.0146773, 0.0141662

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039633, upper bound: 0.0039803
time: 2.57 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039633, upper bound: 0.0039803
time: 2.46 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0057781, 0.0073610, 0.0055866, 0.0087563, -0.0029782, 0.0017744
1: -0.0012130, 0.0023397, -0.0029063, 0.0023983, -0.0036113, 0.0052459
2: -0.0019438, 0.0233963, -0.0024166, 0.0280487, -0.0299925, 0.0258130
3: -0.0045476, -0.0023390, -0.0048149, -0.0022967, -0.0022509, 0.0024760
4: -0.0007282, 0.0099879, -0.0038405, 0.0101929, -0.0109210, 0.0138284
5: -0.0020107, -0.0000124, -0.0020413, 0.0074101, -0.0094209, 0.0020290
6: 0.9899639, 0.9939409, 0.9832411, 0.9939970, -0.0040330, 0.0106997
7: -0.0149023, 0.0046971, -0.0177970, 0.0050680, -0.0199703, 0.0224940
8: -0.0065928, 0.0024599, -0.0154112, 0.0025761, -0.0091689, 0.0178711
9: -0.0122388, 0.0002851, -0.0124707, 0.0028327, -0.0150715, 0.0127558

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 213

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038846, upper bound: 0.0037129
time: 2.45 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038845, upper bound: 0.0036939
time: 2.46 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0057753, 0.0074082, 0.0055728, 0.0090099, -0.0032346, 0.0018354
1: -0.0012381, 0.0024310, -0.0030283, 0.0024353, -0.0036734, 0.0054594
2: -0.0026806, 0.0234654, -0.0027152, 0.0283841, -0.0310648, 0.0261806
3: -0.0045516, -0.0022731, -0.0048342, -0.0022701, -0.0022815, 0.0025611
4: -0.0007474, 0.0103073, -0.0041091, 0.0103222, -0.0110696, 0.0144164
5: -0.0020584, 0.0000066, -0.0020606, 0.0080951, -0.0101535, 0.0020672
6: 0.9899166, 0.9940283, 0.9826703, 0.9940324, -0.0041158, 0.0113580
7: -0.0149453, 0.0052751, -0.0180057, 0.0053021, -0.0202475, 0.0232807
8: -0.0067237, 0.0026410, -0.0160469, 0.0026495, -0.0093732, 0.0186879
9: -0.0126002, 0.0003228, -0.0126171, 0.0030166, -0.0156168, 0.0129399

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=56, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 213

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039434, upper bound: 0.0038481
time: 2.39 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039434, upper bound: 0.0038514
time: 2.57 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0057781, 0.0073610, 0.0055154, 0.0100625, -0.0042844, 0.0018456
1: -0.0012130, 0.0023397, -0.0035350, 0.0024032, -0.0036162, 0.0058747
2: -0.0019438, 0.0233963, -0.0024564, 0.0297762, -0.0317199, 0.0258527
3: -0.0045476, -0.0023390, -0.0049142, -0.0018033, -0.0027443, 0.0025752
4: -0.0007282, 0.0099879, -0.0052240, 0.0102101, -0.0109383, 0.0152120
5: -0.0020107, -0.0000124, -0.0020439, 0.0109379, -0.0129486, 0.0020315
6: 0.9899639, 0.9939409, 0.9803011, 0.9940017, -0.0040377, 0.0136397
7: -0.0149023, 0.0046971, -0.0188718, 0.0050992, -0.0200015, 0.0235688
8: -0.0065928, 0.0024599, -0.0186854, 0.0025859, -0.0091787, 0.0211453
9: -0.0122388, 0.0002851, -0.0124902, 0.0037797, -0.0160184, 0.0127753

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 213

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038397, upper bound: 0.0036039
time: 2.47 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038393, upper bound: 0.0035852
time: 2.49 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0057753, 0.0074082, 0.0055015, 0.0103181, -0.0045428, 0.0019066
1: -0.0012381, 0.0024310, -0.0036580, 0.0024405, -0.0036786, 0.0060891
2: -0.0026806, 0.0234654, -0.0027568, 0.0301142, -0.0327948, 0.0262222
3: -0.0045516, -0.0022731, -0.0049336, -0.0016650, -0.0028866, 0.0026605
4: -0.0007474, 0.0103073, -0.0054948, 0.0103402, -0.0110877, 0.0158020
5: -0.0020584, 0.0000066, -0.0020633, 0.0116283, -0.0136866, 0.0020699
6: 0.9899166, 0.9940283, 0.9797258, 0.9940374, -0.0041208, 0.0143025
7: -0.0149453, 0.0052751, -0.0190821, 0.0053348, -0.0202801, 0.0243571
8: -0.0067237, 0.0026410, -0.0193261, 0.0026597, -0.0093834, 0.0219671
9: -0.0126002, 0.0003228, -0.0126375, 0.0039650, -0.0165652, 0.0129604

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=57, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0039041, upper bound: 0.0037484
time: 2.43 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0039041, upper bound: 0.0037490
time: 2.95 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0056660, 0.0073688, 0.0057796, 0.0074047, -0.0017388, 0.0015892
1: -0.0022043, 0.0023547, -0.0011999, 0.0024244, -0.0046287, 0.0035546
2: -0.0020649, 0.0261201, -0.0026271, 0.0233605, -0.0254254, 0.0287472
3: -0.0047041, -0.0023281, -0.0045456, -0.0022779, -0.0024262, 0.0022174
4: -0.0022958, 0.0100404, -0.0007182, 0.0102840, -0.0125798, 0.0107586
5: -0.0020186, 0.0034716, -0.0020549, -0.0000222, -0.0019964, 0.0055265
6: 0.9865234, 0.9939553, 0.9899884, 0.9940220, -0.0074985, 0.0039669
7: -0.0165971, 0.0047921, -0.0148800, 0.0052330, -0.0216540, 0.0196721
8: -0.0117557, 0.0024897, -0.0065250, 0.0026278, -0.0143835, 0.0090146
9: -0.0122982, 0.0017756, -0.0125739, 0.0002656, -0.0125638, 0.0143495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=54, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039491, upper bound: 0.0039729
time: 2.32 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039491, upper bound: 0.0039616
time: 2.31 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0056603, 0.0074185, 0.0057654, 0.0074242, -0.0017639, 0.0016532
1: -0.0022543, 0.0024511, -0.0013258, 0.0024621, -0.0047163, 0.0037769
2: -0.0028426, 0.0262574, -0.0029308, 0.0237063, -0.0265489, 0.0291882
3: -0.0047120, -0.0022587, -0.0045655, -0.0022508, -0.0024612, 0.0023068
4: -0.0024057, 0.0103775, -0.0008146, 0.0104156, -0.0128214, 0.0111920
5: -0.0020689, 0.0037519, -0.0020746, 0.0000727, -0.0021416, 0.0058265
6: 0.9862899, 0.9940475, 0.9897515, 0.9940580, -0.0077682, 0.0042961
7: -0.0166825, 0.0054021, -0.0150952, 0.0054713, -0.0221537, 0.0204973
8: -0.0120159, 0.0026808, -0.0071803, 0.0027024, -0.0147183, 0.0098611
9: -0.0126796, 0.0018508, -0.0127229, 0.0004543, -0.0131340, 0.0145737

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039896, upper bound: 0.0040979
time: 2.66 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039896, upper bound: 0.0041041
time: 2.58 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0056660, 0.0073688, 0.0057050, 0.0074111, -0.0017451, 0.0016638
1: -0.0022043, 0.0023547, -0.0018596, 0.0024366, -0.0046410, 0.0042143
2: -0.0020649, 0.0261201, -0.0027258, 0.0251730, -0.0272379, 0.0288459
3: -0.0047041, -0.0023281, -0.0046497, -0.0022691, -0.0024350, 0.0023216
4: -0.0022958, 0.0100404, -0.0012234, 0.0103268, -0.0126226, 0.0112639
5: -0.0020186, 0.0034716, -0.0020613, 0.0004752, -0.0024937, 0.0055329
6: 0.9865234, 0.9939553, 0.9887463, 0.9940338, -0.0075103, 0.0052090
7: -0.0165971, 0.0047921, -0.0160078, 0.0053105, -0.0217979, 0.0207998
8: -0.0117557, 0.0024897, -0.0099604, 0.0026521, -0.0144077, 0.0124501
9: -0.0122982, 0.0017756, -0.0126223, 0.0012550, -0.0135532, 0.0143979

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039228, upper bound: 0.0038848
time: 2.38 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039228, upper bound: 0.0038721
time: 2.04 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0056603, 0.0074185, 0.0056908, 0.0074302, -0.0017699, 0.0017277
1: -0.0022543, 0.0024511, -0.0019845, 0.0024738, -0.0047281, 0.0044356
2: -0.0028426, 0.0262574, -0.0030253, 0.0255160, -0.0283587, 0.0292827
3: -0.0047120, -0.0022587, -0.0046694, -0.0022424, -0.0024697, 0.0024107
4: -0.0024057, 0.0103775, -0.0013190, 0.0104566, -0.0128623, 0.0116965
5: -0.0020689, 0.0037519, -0.0020807, 0.0005693, -0.0026382, 0.0058326
6: 0.9862899, 0.9940475, 0.9885111, 0.9940692, -0.0077794, 0.0055364
7: -0.0166825, 0.0054021, -0.0162212, 0.0055454, -0.0222278, 0.0216233
8: -0.0120159, 0.0026808, -0.0106106, 0.0027257, -0.0147415, 0.0132914
9: -0.0126796, 0.0018508, -0.0127692, 0.0014422, -0.0141219, 0.0146200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=58, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039653, upper bound: 0.0040124
time: 2.52 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039653, upper bound: 0.0040139
time: 2.51 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0056364, 0.0078402, 0.0057724, 0.0073846, -0.0017482, 0.0020678
1: -0.0024653, 0.0024445, -0.0012633, 0.0023854, -0.0048507, 0.0037078
2: -0.0027893, 0.0268372, -0.0023126, 0.0235346, -0.0263239, 0.0291498
3: -0.0047453, -0.0022634, -0.0045556, -0.0023060, -0.0024393, 0.0022921
4: -0.0028701, 0.0103543, -0.0007667, 0.0101478, -0.0130179, 0.0111210
5: -0.0020654, 0.0049360, -0.0020346, 0.0000256, -0.0020910, 0.0069706
6: 0.9853030, 0.9940412, 0.9898691, 0.9939846, -0.0086816, 0.0041721
7: -0.0170432, 0.0053603, -0.0149883, 0.0049863, -0.0220296, 0.0203486
8: -0.0131148, 0.0026677, -0.0068548, 0.0025505, -0.0156654, 0.0095225
9: -0.0126535, 0.0021686, -0.0124197, 0.0003606, -0.0130141, 0.0145883

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038331, upper bound: 0.0039371
time: 2.33 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038331, upper bound: 0.0039371
time: 2.39 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0056448, 0.0076865, 0.0057826, 0.0073840, -0.0017392, 0.0019039
1: -0.0023913, 0.0024748, -0.0011733, 0.0023843, -0.0047757, 0.0036481
2: -0.0030335, 0.0266339, -0.0023040, 0.0232874, -0.0263209, 0.0289379
3: -0.0047337, -0.0022416, -0.0045414, -0.0023068, -0.0024269, 0.0022997
4: -0.0027073, 0.0104602, -0.0006978, 0.0101440, -0.0128513, 0.0111580
5: -0.0020812, 0.0045208, -0.0020340, -0.0000423, -0.0020390, 0.0065548
6: 0.9856491, 0.9940702, 0.9900386, 0.9939836, -0.0083345, 0.0040316
7: -0.0169167, 0.0055519, -0.0148345, 0.0049796, -0.0218963, 0.0203864
8: -0.0127294, 0.0027277, -0.0063863, 0.0025484, -0.0152779, 0.0091140
9: -0.0127733, 0.0020572, -0.0124154, 0.0002257, -0.0129989, 0.0144726

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038199, upper bound: 0.0039371
time: 1.84 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038199, upper bound: 0.0039371
time: 2.48 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0055941, 0.0086184, 0.0057619, 0.0074311, -0.0018370, 0.0028565
1: -0.0028399, 0.0023638, -0.0013563, 0.0024754, -0.0053154, 0.0037202
2: -0.0021383, 0.0278664, -0.0030388, 0.0237903, -0.0259285, 0.0309052
3: -0.0048045, -0.0023216, -0.0045703, -0.0022412, -0.0025633, 0.0022487
4: -0.0036944, 0.0100722, -0.0008380, 0.0104625, -0.0141569, 0.0109102
5: -0.0020233, 0.0070379, -0.0020816, 0.0000957, -0.0021190, 0.0091194
6: 0.9835514, 0.9939640, 0.9896939, 0.9940708, -0.0105194, 0.0042701
7: -0.0176836, 0.0048496, -0.0151474, 0.0055560, -0.0231567, 0.0199970
8: -0.0150656, 0.0025077, -0.0073395, 0.0027290, -0.0177946, 0.0098472
9: -0.0123342, 0.0027328, -0.0127759, 0.0005002, -0.0128343, 0.0155087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=56, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039106, upper bound: 0.0039831
time: 2.44 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039106, upper bound: 0.0039830
time: 2.58 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0055880, 0.0087297, 0.0057619, 0.0074311, -0.0018431, 0.0029678
1: -0.0028935, 0.0024569, -0.0013563, 0.0024754, -0.0053689, 0.0038133
2: -0.0028894, 0.0280135, -0.0030388, 0.0237903, -0.0266796, 0.0310523
3: -0.0048129, -0.0022545, -0.0045703, -0.0022412, -0.0025717, 0.0023158
4: -0.0038123, 0.0103977, -0.0008380, 0.0104625, -0.0142747, 0.0112357
5: -0.0020719, 0.0073383, -0.0020816, 0.0000957, -0.0021676, 0.0094198
6: 0.9833012, 0.9940531, 0.9896939, 0.9940708, -0.0107696, 0.0043592
7: -0.0177751, 0.0054388, -0.0151474, 0.0055560, -0.0233311, 0.0205862
8: -0.0153444, 0.0026923, -0.0073395, 0.0027290, -0.0180734, 0.0100318
9: -0.0127026, 0.0028134, -0.0127759, 0.0005002, -0.0132027, 0.0155893

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=56, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039106, upper bound: 0.0040303
time: 2.60 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039106, upper bound: 0.0040303
time: 2.26 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0056660, 0.0073688, 0.0056643, 0.0074141, -0.0017481, 0.0017045
1: -0.0022043, 0.0023547, -0.0022193, 0.0024424, -0.0046468, 0.0045740
2: -0.0020649, 0.0261201, -0.0027725, 0.0261612, -0.0282261, 0.0288927
3: -0.0047041, -0.0023281, -0.0047065, -0.0022649, -0.0024392, 0.0023784
4: -0.0022958, 0.0100404, -0.0014989, 0.0103471, -0.0126429, 0.0115393
5: -0.0020186, 0.0034716, -0.0020643, 0.0007463, -0.0027649, 0.0055359
6: 0.9865234, 0.9939553, 0.9880690, 0.9940392, -0.0075158, 0.0058863
7: -0.0165971, 0.0047921, -0.0166226, 0.0053471, -0.0209839, 0.0214147
8: -0.0117557, 0.0024897, -0.0118335, 0.0026636, -0.0144192, 0.0143232
9: -0.0122982, 0.0017756, -0.0126453, 0.0017944, -0.0140926, 0.0144208

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=53, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039491, upper bound: 0.0039326
time: 2.49 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039491, upper bound: 0.0039184
time: 2.47 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0056603, 0.0074185, 0.0056499, 0.0074340, -0.0017737, 0.0017686
1: -0.0022543, 0.0024511, -0.0023460, 0.0024811, -0.0047353, 0.0047972
2: -0.0028426, 0.0262574, -0.0030840, 0.0265095, -0.0293521, 0.0293414
3: -0.0047120, -0.0022587, -0.0047265, -0.0022371, -0.0024749, 0.0024678
4: -0.0024057, 0.0103775, -0.0015960, 0.0104821, -0.0128878, 0.0119734
5: -0.0020689, 0.0037519, -0.0020845, 0.0008419, -0.0029108, 0.0058364
6: 0.9862899, 0.9940475, 0.9878303, 0.9940761, -0.0077863, 0.0062172
7: -0.0166825, 0.0054021, -0.0168393, 0.0055915, -0.0222739, 0.0222414
8: -0.0120159, 0.0026808, -0.0124937, 0.0027401, -0.0147560, 0.0151744
9: -0.0126796, 0.0018508, -0.0127980, 0.0019845, -0.0146642, 0.0146488

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039896, upper bound: 0.0040602
time: 2.23 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039896, upper bound: 0.0040722
time: 2.23 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0056660, 0.0073688, 0.0055920, 0.0074182, -0.0017523, 0.0017767
1: -0.0022043, 0.0023547, -0.0028579, 0.0024505, -0.0046549, 0.0052126
2: -0.0020649, 0.0261201, -0.0028380, 0.0279159, -0.0299808, 0.0289581
3: -0.0047041, -0.0023281, -0.0048073, -0.0022591, -0.0024450, 0.0024792
4: -0.0022958, 0.0100404, -0.0019880, 0.0103754, -0.0126712, 0.0120284
5: -0.0020186, 0.0034716, -0.0020686, 0.0012278, -0.0032463, 0.0055402
6: 0.9865234, 0.9939553, 0.9868666, 0.9940470, -0.0075235, 0.0070887
7: -0.0165971, 0.0047921, -0.0177143, 0.0053985, -0.0211063, 0.0225064
8: -0.0117557, 0.0024897, -0.0151593, 0.0026796, -0.0144353, 0.0176490
9: -0.0122982, 0.0017756, -0.0126773, 0.0027522, -0.0150504, 0.0144529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=56, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039228, upper bound: 0.0038548
time: 2.18 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039228, upper bound: 0.0038401
time: 2.24 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 5.82 seconds
IS_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0040540, upper bound: 0.0040210
IS_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0040540, upper bound: 0.0040078
IS_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0040984, upper bound: 0.0041433
IS_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0040984, upper bound: 0.0041472
IS_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0040130, upper bound: 0.0039186
IS_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0040130, upper bound: 0.0039053
IS_A1_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0040634, upper bound: 0.0040467
IS_A1_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0040634, upper bound: 0.0040482
IS_A1_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039186, upper bound: 0.0039707
IS_A1_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039186, upper bound: 0.0039706
IS_A1_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039053, upper bound: 0.0039706
IS_A1_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039053, upper bound: 0.0039707
IS_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039999, upper bound: 0.0040190
IS_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039999, upper bound: 0.0040191
IS_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039999, upper bound: 0.0040657
IS_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039999, upper bound: 0.0040657
IS_A1_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039579, upper bound: 0.0038092
IS_A1_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039579, upper bound: 0.0038093
IS_A1_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039579, upper bound: 0.0038576
IS_A1_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039579, upper bound: 0.0038575
IS_A1_B1_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0038155, upper bound: 0.0037384
IS_A1_B1_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0038141, upper bound: 0.0037203
IS_A1_B1_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0038155, upper bound: 0.0037919
IS_A1_B1_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0038141, upper bound: 0.0037740
IS_A1_B1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0038093, upper bound: 0.0039891
IS_A1_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0038093, upper bound: 0.0039900
IS_A1_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0038093, upper bound: 0.0039892
IS_A1_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0038093, upper bound: 0.0039900
IS_A1_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0038093, upper bound: 0.0039382
IS_A1_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0038093, upper bound: 0.0039387
IS_A1_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0038093, upper bound: 0.0039383
IS_A1_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0038093, upper bound: 0.0039387
IS_A1_B2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0040102, upper bound: 0.0039118
IS_A1_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0040102, upper bound: 0.0038990
IS_A1_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0040514, upper bound: 0.0040370
IS_A1_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0040514, upper bound: 0.0040464
IS_A1_B2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039737, upper bound: 0.0038330
IS_A1_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039737, upper bound: 0.0038198
IS_A1_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0040223, upper bound: 0.0039622
IS_A1_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0040223, upper bound: 0.0039654
IS_A1_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039633, upper bound: 0.0039296
IS_A1_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039633, upper bound: 0.0039295
IS_A1_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039633, upper bound: 0.0039803
IS_A1_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039633, upper bound: 0.0039803
IS_A1_B2_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0038846, upper bound: 0.0037129
IS_A1_B2_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0038845, upper bound: 0.0036939
IS_A1_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039434, upper bound: 0.0038481
IS_A1_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039434, upper bound: 0.0038514
IS_A1_B2_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0038397, upper bound: 0.0036039
IS_A1_B2_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0038393, upper bound: 0.0035852
IS_A1_B2_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039041, upper bound: 0.0037484
IS_A1_B2_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039041, upper bound: 0.0037490
IS_A2_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039491, upper bound: 0.0039729
IS_A2_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039491, upper bound: 0.0039616
IS_A2_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039896, upper bound: 0.0040979
IS_A2_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039896, upper bound: 0.0041041
IS_A2_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039228, upper bound: 0.0038848
IS_A2_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039228, upper bound: 0.0038721
IS_A2_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039653, upper bound: 0.0040124
IS_A2_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039653, upper bound: 0.0040139
IS_A2_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0038331, upper bound: 0.0039371
IS_A2_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0038331, upper bound: 0.0039371
IS_A2_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0038199, upper bound: 0.0039371
IS_A2_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0038199, upper bound: 0.0039371
IS_A2_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039106, upper bound: 0.0039831
IS_A2_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039106, upper bound: 0.0039830
IS_A2_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039106, upper bound: 0.0040303
IS_A2_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039106, upper bound: 0.0040303
IS_A2_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039491, upper bound: 0.0039326
IS_A2_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039491, upper bound: 0.0039184
IS_A2_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039896, upper bound: 0.0040602
IS_A2_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039896, upper bound: 0.0040722
IS_A2_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039228, upper bound: 0.0038548
IS_A2_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.82
Output dim: 6, lower bound: -0.0039228, upper bound: 0.0038401
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.82
Output dim: 6, lower bound: -0.0040162, upper bound: 0.0039889
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.82
Output dim: 6, lower bound: -0.0038541, upper bound: 0.0039113
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.82
Output dim: 6, lower bound: -0.0038398, upper bound: 0.0039112
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.82
Output dim: 6, lower bound: -0.0039761, upper bound: 0.0039523
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.82
Output dim: 6, lower bound: -0.0039761, upper bound: 0.0040061

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 5.11 + 596.02 = 601.12 seconds
