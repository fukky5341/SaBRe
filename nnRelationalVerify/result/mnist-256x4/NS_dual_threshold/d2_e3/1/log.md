## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00390744


## IAR start

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
execution time: IAR + RelationalAnalysis = 1.99 + 3.76 = 5.76 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0043416, upper bound: 0.0043415

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0042269, upper bound: 0.0041713
time: 1.88 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0041964, upper bound: 0.0041964
time: 2.55 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 4.61 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 4.61
Output dim: 6, lower bound: -0.0042269, upper bound: 0.0041713
NS_A2, status: Status.UNKNOWN, split count: 1, time: 4.61
Output dim: 6, lower bound: -0.0041964, upper bound: 0.0041964

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0041714, upper bound: 0.0041714
time: 2.40 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0041714, upper bound: 0.0041714
time: 2.37 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 97

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039521, upper bound: 0.0040220
time: 1.99 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039568, upper bound: 0.0039566
time: 2.52 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 6.36 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 6.36
Output dim: 6, lower bound: -0.0041714, upper bound: 0.0041714
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 6.36
Output dim: 6, lower bound: -0.0041714, upper bound: 0.0041714
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 6.36
Output dim: 6, lower bound: -0.0039521, upper bound: 0.0040220
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 6.36
Output dim: 6, lower bound: -0.0039568, upper bound: 0.0039566

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040539, upper bound: 0.0039249
time: 2.35 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039990, upper bound: 0.0039348
time: 2.09 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 97

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040539, upper bound: 0.0039249
time: 2.36 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039990, upper bound: 0.0039348
time: 1.93 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038616, upper bound: 0.0038747
time: 2.47 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038089, upper bound: 0.0038857
time: 2.37 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038655, upper bound: 0.0037970
time: 2.48 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038098, upper bound: 0.0038097
time: 2.19 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 6.70 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.70
Output dim: 6, lower bound: -0.0040539, upper bound: 0.0039249
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.70
Output dim: 6, lower bound: -0.0039990, upper bound: 0.0039348
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.70
Output dim: 6, lower bound: -0.0040539, upper bound: 0.0039249
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.70
Output dim: 6, lower bound: -0.0039990, upper bound: 0.0039348
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 6.70
Output dim: 6, lower bound: -0.0038616, upper bound: 0.0038747
NS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 6.70
Output dim: 6, lower bound: -0.0038089, upper bound: 0.0038857
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 6.70
Output dim: 6, lower bound: -0.0038655, upper bound: 0.0037970
NS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 6.70
Output dim: 6, lower bound: -0.0038098, upper bound: 0.0038097

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 97

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039609, upper bound: 0.0039560
time: 2.22 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039746, upper bound: 0.0038990
time: 2.43 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039657, upper bound: 0.0038919
time: 3.41 seconds

## Relational analysis of NS_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0039073, upper bound: 0.0039073
time: 2.27 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039077, upper bound: 0.0038366
time: 2.29 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039189, upper bound: 0.0037860
time: 1.94 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038384, upper bound: 0.0038451
time: 1.96 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038521, upper bound: 0.0037920
time: 2.69 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 6.52 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.52
Output dim: 6, lower bound: -0.0039609, upper bound: 0.0039560
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.52
Output dim: 6, lower bound: -0.0039746, upper bound: 0.0038990
NS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 6.52
Output dim: 6, lower bound: -0.0039657, upper bound: 0.0038919
NS_A1_B1_A2_A2, status: Status.VERIFIED, split count: 4, time: 6.52
Output dim: 6, lower bound: -0.0039073, upper bound: 0.0039073
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.52
Output dim: 6, lower bound: -0.0039077, upper bound: 0.0038366
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.52
Output dim: 6, lower bound: -0.0039189, upper bound: 0.0037860
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 6.52
Output dim: 6, lower bound: -0.0038384, upper bound: 0.0038451
NS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 6.52
Output dim: 6, lower bound: -0.0038521, upper bound: 0.0037920

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0057467, 0.0074528, 0.0057522, 0.0074592, -0.0017126, 0.0017006
1: -0.0014912, 0.0025174, -0.0014423, 0.0025299, -0.0040211, 0.0039596
2: -0.0033771, 0.0241607, -0.0034782, 0.0240263, -0.0274034, 0.0276389
3: -0.0045916, -0.0022109, -0.0045838, -0.0022019, -0.0023896, 0.0023729
4: -0.0009412, 0.0106091, -0.0009038, 0.0106529, -0.0115941, 0.0115128
5: -0.0021034, 0.0001974, -0.0021100, 0.0001605, -0.0022639, 0.0023074
6: 0.9894400, 0.9941109, 0.9895321, 0.9941230, -0.0046830, 0.0045788
7: -0.0153779, 0.0058214, -0.0152943, 0.0059006, -0.0212786, 0.0211157
8: -0.0080417, 0.0028121, -0.0077869, 0.0028370, -0.0108787, 0.0105990
9: -0.0129418, 0.0007024, -0.0129914, 0.0006290, -0.0135708, 0.0136938

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039609, upper bound: 0.0039560
time: 2.30 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039609, upper bound: 0.0039560
time: 2.29 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0057517, 0.0074473, 0.0056758, 0.0074648, -0.0017131, 0.0017715
1: -0.0014462, 0.0025069, -0.0021171, 0.0025408, -0.0039870, 0.0046240
2: -0.0032925, 0.0240371, -0.0035660, 0.0258804, -0.0291729, 0.0276030
3: -0.0045845, -0.0022185, -0.0046904, -0.0021941, -0.0023904, 0.0024719
4: -0.0009068, 0.0105724, -0.0021037, 0.0106909, -0.0115977, 0.0126762
5: -0.0020980, 0.0001635, -0.0021157, 0.0029820, -0.0050799, 0.0022791
6: 0.9895248, 0.9941009, 0.9869315, 0.9941334, -0.0046086, 0.0071694
7: -0.0153010, 0.0057550, -0.0164479, 0.0059695, -0.0212705, 0.0222029
8: -0.0078073, 0.0027914, -0.0113012, 0.0028586, -0.0106659, 0.0140926
9: -0.0129003, 0.0006349, -0.0130344, 0.0016441, -0.0145444, 0.0136693

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039746, upper bound: 0.0038989
time: 2.43 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039746, upper bound: 0.0038990
time: 2.63 seconds

## BFS NS instance: NS_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: 0.0056833, 0.0074078, 0.0057415, 0.0074697, -0.0017864, 0.0016663
1: -0.0020511, 0.0024304, -0.0015366, 0.0025502, -0.0046013, 0.0039670
2: -0.0026753, 0.0256993, -0.0036414, 0.0242854, -0.0269608, 0.0293407
3: -0.0046800, -0.0022736, -0.0045987, -0.0021873, -0.0024926, 0.0023251
4: -0.0019587, 0.0103050, -0.0009760, 0.0107236, -0.0126823, 0.0112810
5: -0.0020581, 0.0026121, -0.0021205, 0.0002316, -0.0022897, 0.0047326
6: 0.9872397, 0.9940277, 0.9893545, 0.9941422, -0.0069026, 0.0046732
7: -0.0163352, 0.0052709, -0.0154555, 0.0060287, -0.0223639, 0.0207264
8: -0.0109579, 0.0026397, -0.0082781, 0.0028771, -0.0138350, 0.0109178
9: -0.0125976, 0.0015448, -0.0130714, 0.0007705, -0.0133681, 0.0146163

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 213

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_A1_A1

### Relational analysis result of NS_A1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039383, upper bound: 0.0038226
time: 2.35 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2

### Relational analysis result of NS_A1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039387, upper bound: 0.0038666
time: 2.60 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0057467, 0.0074528, 0.0056359, 0.0078505, -0.0021038, 0.0018169
1: -0.0014912, 0.0025174, -0.0024702, 0.0025463, -0.0040375, 0.0049876
2: -0.0033771, 0.0241607, -0.0036105, 0.0268508, -0.0302279, 0.0277712
3: -0.0045916, -0.0022109, -0.0047461, -0.0021901, -0.0024015, 0.0025352
4: -0.0009412, 0.0106091, -0.0028810, 0.0107102, -0.0116514, 0.0134900
5: -0.0021034, 0.0001974, -0.0021185, 0.0049637, -0.0070671, 0.0023159
6: 0.9894400, 0.9941109, 0.9852800, 0.9941387, -0.0046986, 0.0088310
7: -0.0153779, 0.0058214, -0.0170516, 0.0060044, -0.0213824, 0.0228730
8: -0.0080417, 0.0028121, -0.0131405, 0.0028695, -0.0109112, 0.0159527
9: -0.0129418, 0.0007024, -0.0130563, 0.0021760, -0.0151178, 0.0137587

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A1_B1_B1

### Relational analysis result of NS_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039077, upper bound: 0.0038366
time: 2.51 seconds

## Relational analysis of NS_A1_B2_A1_B1_B2

### Relational analysis result of NS_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039077, upper bound: 0.0038366
time: 2.34 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0057517, 0.0074473, 0.0055624, 0.0092006, -0.0034488, 0.0018850
1: -0.0014462, 0.0025069, -0.0031201, 0.0025535, -0.0039996, 0.0056270
2: -0.0032925, 0.0240371, -0.0036680, 0.0286362, -0.0319287, 0.0277050
3: -0.0045845, -0.0022185, -0.0048487, -0.0021850, -0.0023995, 0.0026302
4: -0.0009068, 0.0105724, -0.0043110, 0.0107351, -0.0116419, 0.0148834
5: -0.0020980, 0.0001635, -0.0021223, 0.0086100, -0.0107079, 0.0022857
6: 0.9895248, 0.9941009, 0.9822412, 0.9941454, -0.0046206, 0.0118598
7: -0.0153010, 0.0057550, -0.0181625, 0.0060495, -0.0213505, 0.0239175
8: -0.0078073, 0.0027914, -0.0165247, 0.0028836, -0.0106909, 0.0193161
9: -0.0129003, 0.0006349, -0.0130845, 0.0031548, -0.0160551, 0.0137193

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A1_B2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039189, upper bound: 0.0037860
time: 2.48 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039189, upper bound: 0.0037860
time: 5.97 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 10.32 seconds
NS_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 10.32
Output dim: 6, lower bound: -0.0039609, upper bound: 0.0039560
NS_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 10.32
Output dim: 6, lower bound: -0.0039609, upper bound: 0.0039560
NS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 10.32
Output dim: 6, lower bound: -0.0039746, upper bound: 0.0038989
NS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 10.32
Output dim: 6, lower bound: -0.0039746, upper bound: 0.0038990
NS_A1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 10.32
Output dim: 6, lower bound: -0.0039383, upper bound: 0.0038226
NS_A1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 10.32
Output dim: 6, lower bound: -0.0039387, upper bound: 0.0038666
NS_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 10.32
Output dim: 6, lower bound: -0.0039077, upper bound: 0.0038366
NS_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 10.32
Output dim: 6, lower bound: -0.0039077, upper bound: 0.0038366
NS_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 10.32
Output dim: 6, lower bound: -0.0039189, upper bound: 0.0037860
NS_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 10.32
Output dim: 6, lower bound: -0.0039189, upper bound: 0.0037860

## BFS NS instance: NS_A1_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: 0.0057467, 0.0074528, 0.0057653, 0.0074242, -0.0016775, 0.0016875
1: -0.0014912, 0.0025174, -0.0013262, 0.0024621, -0.0039532, 0.0038436
2: -0.0033771, 0.0241607, -0.0029308, 0.0237074, -0.0270845, 0.0270915
3: -0.0045916, -0.0022109, -0.0045655, -0.0022508, -0.0023408, 0.0023546
4: -0.0009412, 0.0106091, -0.0008149, 0.0104156, -0.0113569, 0.0114239
5: -0.0021034, 0.0001974, -0.0020746, 0.0000730, -0.0021764, 0.0022720
6: 0.9894400, 0.9941109, 0.9897507, 0.9940580, -0.0046180, 0.0043602
7: -0.0153779, 0.0058214, -0.0150959, 0.0054713, -0.0208492, 0.0209172
8: -0.0080417, 0.0028121, -0.0071824, 0.0027024, -0.0107441, 0.0099946
9: -0.0129418, 0.0007024, -0.0127229, 0.0004549, -0.0133967, 0.0134253

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039339, upper bound: 0.0038877
time: 2.17 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039342, upper bound: 0.0039296
time: 2.49 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: 0.0057467, 0.0074528, 0.0056833, 0.0074078, -0.0016612, 0.0017695
1: -0.0014912, 0.0025174, -0.0020511, 0.0024304, -0.0039216, 0.0045685
2: -0.0033771, 0.0241607, -0.0026753, 0.0256993, -0.0290764, 0.0268360
3: -0.0045916, -0.0022109, -0.0046800, -0.0022736, -0.0023179, 0.0024690
4: -0.0009412, 0.0106091, -0.0013701, 0.0103050, -0.0112462, 0.0119792
5: -0.0021034, 0.0001974, -0.0020581, 0.0006196, -0.0027230, 0.0022554
6: 0.9894400, 0.9941109, 0.9883857, 0.9940277, -0.0045877, 0.0057253
7: -0.0153779, 0.0058214, -0.0163352, 0.0052709, -0.0206488, 0.0221566
8: -0.0080417, 0.0028121, -0.0109579, 0.0026397, -0.0106814, 0.0137701
9: -0.0129418, 0.0007024, -0.0125976, 0.0015422, -0.0144840, 0.0133000

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 213

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038982, upper bound: 0.0039296
time: 2.30 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039342, upper bound: 0.0039299
time: 2.59 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: 0.0057517, 0.0074473, 0.0056898, 0.0074302, -0.0016785, 0.0017576
1: -0.0014462, 0.0025069, -0.0019940, 0.0024738, -0.0039199, 0.0045009
2: -0.0032925, 0.0240371, -0.0030253, 0.0255422, -0.0288347, 0.0270623
3: -0.0045845, -0.0022185, -0.0046709, -0.0022424, -0.0023421, 0.0024524
4: -0.0009068, 0.0105724, -0.0018329, 0.0104566, -0.0113634, 0.0124053
5: -0.0020980, 0.0001635, -0.0020807, 0.0022914, -0.0043893, 0.0022441
6: 0.9895248, 0.9941009, 0.9875071, 0.9940692, -0.0045444, 0.0065938
7: -0.0153010, 0.0057550, -0.0162375, 0.0055454, -0.0208464, 0.0219925
8: -0.0078073, 0.0027914, -0.0106602, 0.0027257, -0.0105330, 0.0134516
9: -0.0129003, 0.0006349, -0.0127692, 0.0014588, -0.0143591, 0.0134041

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039140, upper bound: 0.0038720
time: 2.37 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039342, upper bound: 0.0038734
time: 6.33 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: 0.0057517, 0.0074473, 0.0056111, 0.0083063, -0.0025545, 0.0018363
1: -0.0014462, 0.0025069, -0.0026897, 0.0024414, -0.0038876, 0.0051966
2: -0.0032925, 0.0240371, -0.0027642, 0.0274536, -0.0307461, 0.0268013
3: -0.0045845, -0.0022185, -0.0047807, -0.0022657, -0.0023188, 0.0025622
4: -0.0009068, 0.0105724, -0.0033638, 0.0103435, -0.0112502, 0.0139362
5: -0.0020980, 0.0001635, -0.0020638, 0.0061947, -0.0082927, 0.0022273
6: 0.9895248, 0.9941009, 0.9842541, 0.9940383, -0.0045135, 0.0098469
7: -0.0153010, 0.0057550, -0.0174267, 0.0053406, -0.0206416, 0.0231817
8: -0.0078073, 0.0027914, -0.0142831, 0.0026615, -0.0104688, 0.0170745
9: -0.0129003, 0.0006349, -0.0126412, 0.0025065, -0.0154068, 0.0132761

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039140, upper bound: 0.0038721
time: 2.52 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039342, upper bound: 0.0038733
time: 5.06 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A1

### Backsubstitution after applying NS history:
0: 0.0056898, 0.0073454, 0.0057560, 0.0074506, -0.0017608, 0.0015893
1: -0.0019941, 0.0023094, -0.0014082, 0.0025131, -0.0045072, 0.0037175
2: -0.0016993, 0.0255425, -0.0033428, 0.0239327, -0.0256320, 0.0288853
3: -0.0046709, -0.0023608, -0.0045785, -0.0022140, -0.0024569, 0.0022177
4: -0.0018331, 0.0098820, -0.0008777, 0.0105942, -0.0124273, 0.0107597
5: -0.0019949, 0.0022919, -0.0021012, 0.0001348, -0.0021297, 0.0043931
6: 0.9875066, 0.9939119, 0.9895963, 0.9941069, -0.0066003, 0.0043156
7: -0.0162376, 0.0045053, -0.0152360, 0.0057945, -0.0215839, 0.0197413
8: -0.0106607, 0.0023998, -0.0076094, 0.0028037, -0.0134644, 0.0100093
9: -0.0121189, 0.0014589, -0.0129250, 0.0005779, -0.0126968, 0.0143838

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of NS_A1_B1_A2_A1_A1_A1

### Relational analysis result of NS_A1_B1_A2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038608, upper bound: 0.0037167
time: 2.36 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_A2

### Relational analysis result of NS_A1_B1_A2_A1_A1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038398, upper bound: 0.0037133
time: 2.41 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2

### Backsubstitution after applying NS history:
0: 0.0056933, 0.0073912, 0.0057415, 0.0074697, -0.0017764, 0.0016497
1: -0.0019629, 0.0023982, -0.0015366, 0.0025502, -0.0045130, 0.0039348
2: -0.0024161, 0.0254567, -0.0036414, 0.0242854, -0.0267015, 0.0290981
3: -0.0046660, -0.0022968, -0.0045987, -0.0021873, -0.0024787, 0.0023019
4: -0.0017644, 0.0101926, -0.0009760, 0.0107236, -0.0124880, 0.0111686
5: -0.0020413, 0.0021167, -0.0021205, 0.0002316, -0.0022729, 0.0042373
6: 0.9876527, 0.9939969, 0.9893545, 0.9941422, -0.0064896, 0.0046424
7: -0.0161843, 0.0050675, -0.0154555, 0.0060287, -0.0222130, 0.0205231
8: -0.0104981, 0.0025760, -0.0082781, 0.0028771, -0.0133752, 0.0108540
9: -0.0124704, 0.0014119, -0.0130714, 0.0007705, -0.0132409, 0.0144833

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 213

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of NS_A1_B1_A2_A1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038624, upper bound: 0.0037676
time: 2.45 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A1_A2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038437, upper bound: 0.0037669
time: 1.95 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: 0.0057467, 0.0074528, 0.0056499, 0.0075929, -0.0018463, 0.0018029
1: -0.0014912, 0.0025174, -0.0023463, 0.0024811, -0.0039722, 0.0048637
2: -0.0033771, 0.0241607, -0.0030840, 0.0265102, -0.0298873, 0.0272447
3: -0.0045916, -0.0022109, -0.0047265, -0.0022371, -0.0023544, 0.0025156
4: -0.0009412, 0.0106091, -0.0026082, 0.0104821, -0.0114233, 0.0132172
5: -0.0021034, 0.0001974, -0.0020845, 0.0042681, -0.0063715, 0.0022819
6: 0.9894400, 0.9941109, 0.9858598, 0.9940761, -0.0046361, 0.0082512
7: -0.0153779, 0.0058214, -0.0168397, 0.0055915, -0.0209694, 0.0226611
8: -0.0080417, 0.0028121, -0.0124949, 0.0027401, -0.0107818, 0.0153071
9: -0.0129418, 0.0007024, -0.0127980, 0.0019893, -0.0149311, 0.0135004

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B1_B1_B1

### Relational analysis result of NS_A1_B2_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038468, upper bound: 0.0038091
time: 2.22 seconds

## Relational analysis of NS_A1_B2_A1_B1_B1_B2

### Relational analysis result of NS_A1_B2_A1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038804, upper bound: 0.0038099
time: 2.51 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: 0.0057467, 0.0074528, 0.0055728, 0.0090099, -0.0032633, 0.0018800
1: -0.0014912, 0.0025174, -0.0030283, 0.0024353, -0.0039265, 0.0055457
2: -0.0033771, 0.0241607, -0.0027152, 0.0283841, -0.0317612, 0.0268759
3: -0.0045916, -0.0022109, -0.0048342, -0.0022701, -0.0023215, 0.0026233
4: -0.0009412, 0.0106091, -0.0041091, 0.0103222, -0.0112635, 0.0147182
5: -0.0021034, 0.0001974, -0.0020606, 0.0080951, -0.0101986, 0.0022580
6: 0.9894400, 0.9941109, 0.9826703, 0.9940324, -0.0045924, 0.0114406
7: -0.0153779, 0.0058214, -0.0180057, 0.0053021, -0.0206801, 0.0238270
8: -0.0080417, 0.0028121, -0.0160469, 0.0026495, -0.0106911, 0.0188590
9: -0.0129418, 0.0007024, -0.0126171, 0.0030166, -0.0159584, 0.0133195

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 97

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B1_B2_B1

### Relational analysis result of NS_A1_B2_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038468, upper bound: 0.0038091
time: 2.21 seconds

## Relational analysis of NS_A1_B2_A1_B1_B2_B2

### Relational analysis result of NS_A1_B2_A1_B1_B2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038804, upper bound: 0.0038100
time: 2.42 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: 0.0057517, 0.0074473, 0.0055761, 0.0089478, -0.0031961, 0.0018712
1: -0.0014462, 0.0025069, -0.0029984, 0.0024884, -0.0039346, 0.0055053
2: -0.0032925, 0.0240371, -0.0031433, 0.0283020, -0.0315945, 0.0271803
3: -0.0045845, -0.0022185, -0.0048295, -0.0022318, -0.0023526, 0.0026110
4: -0.0009068, 0.0105724, -0.0040433, 0.0105077, -0.0114145, 0.0146157
5: -0.0020980, 0.0001635, -0.0020883, 0.0079274, -0.0100254, 0.0022518
6: 0.9895248, 0.9941009, 0.9828101, 0.9940832, -0.0045584, 0.0112908
7: -0.0153010, 0.0057550, -0.0179546, 0.0056380, -0.0209389, 0.0237096
8: -0.0078073, 0.0027914, -0.0158912, 0.0027547, -0.0105620, 0.0186826
9: -0.0129003, 0.0006349, -0.0128271, 0.0029716, -0.0158719, 0.0134620

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B2_B1_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038600, upper bound: 0.0037598
time: 2.79 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_B2

### Relational analysis result of NS_A1_B2_A1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038913, upper bound: 0.0037608
time: 2.95 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: 0.0057517, 0.0074473, 0.0055015, 0.0103181, -0.0045664, 0.0019458
1: -0.0014462, 0.0025069, -0.0036580, 0.0024405, -0.0038866, 0.0061649
2: -0.0032925, 0.0240371, -0.0027568, 0.0301142, -0.0334067, 0.0267939
3: -0.0045845, -0.0022185, -0.0049336, -0.0016650, -0.0029194, 0.0027151
4: -0.0009068, 0.0105724, -0.0054948, 0.0103402, -0.0112470, 0.0160672
5: -0.0020980, 0.0001635, -0.0020633, 0.0116283, -0.0137262, 0.0022268
6: 0.9895248, 0.9941009, 0.9797258, 0.9940374, -0.0045126, 0.0143752
7: -0.0153010, 0.0057550, -0.0190821, 0.0053348, -0.0206358, 0.0248371
8: -0.0078073, 0.0027914, -0.0193261, 0.0026597, -0.0104670, 0.0221175
9: -0.0129003, 0.0006349, -0.0126375, 0.0039650, -0.0168653, 0.0132724

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B2_B2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038600, upper bound: 0.0037598
time: 2.54 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038913, upper bound: 0.0037608
time: 2.40 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 6.88 seconds
NS_A1_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.88
Output dim: 6, lower bound: -0.0039339, upper bound: 0.0038877
NS_A1_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.88
Output dim: 6, lower bound: -0.0039342, upper bound: 0.0039296
NS_A1_B1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.88
Output dim: 6, lower bound: -0.0038982, upper bound: 0.0039296
NS_A1_B1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.88
Output dim: 6, lower bound: -0.0039342, upper bound: 0.0039299
NS_A1_B1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.88
Output dim: 6, lower bound: -0.0039140, upper bound: 0.0038720
NS_A1_B1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.88
Output dim: 6, lower bound: -0.0039342, upper bound: 0.0038734
NS_A1_B1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.88
Output dim: 6, lower bound: -0.0039140, upper bound: 0.0038721
NS_A1_B1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.88
Output dim: 6, lower bound: -0.0039342, upper bound: 0.0038733
NS_A1_B1_A2_A1_A1_A1, status: Status.VERIFIED, split count: 6, time: 6.88
Output dim: 6, lower bound: -0.0038608, upper bound: 0.0037167
NS_A1_B1_A2_A1_A1_A2, status: Status.VERIFIED, split count: 6, time: 6.88
Output dim: 6, lower bound: -0.0038398, upper bound: 0.0037133
NS_A1_B1_A2_A1_A2_A1, status: Status.VERIFIED, split count: 6, time: 6.88
Output dim: 6, lower bound: -0.0038624, upper bound: 0.0037676
NS_A1_B1_A2_A1_A2_A2, status: Status.VERIFIED, split count: 6, time: 6.88
Output dim: 6, lower bound: -0.0038437, upper bound: 0.0037669
NS_A1_B2_A1_B1_B1_B1, status: Status.VERIFIED, split count: 6, time: 6.88
Output dim: 6, lower bound: -0.0038468, upper bound: 0.0038091
NS_A1_B2_A1_B1_B1_B2, status: Status.VERIFIED, split count: 6, time: 6.88
Output dim: 6, lower bound: -0.0038804, upper bound: 0.0038099
NS_A1_B2_A1_B1_B2_B1, status: Status.VERIFIED, split count: 6, time: 6.88
Output dim: 6, lower bound: -0.0038468, upper bound: 0.0038091
NS_A1_B2_A1_B1_B2_B2, status: Status.VERIFIED, split count: 6, time: 6.88
Output dim: 6, lower bound: -0.0038804, upper bound: 0.0038100
NS_A1_B2_A1_B2_B1_B1, status: Status.VERIFIED, split count: 6, time: 6.88
Output dim: 6, lower bound: -0.0038600, upper bound: 0.0037598
NS_A1_B2_A1_B2_B1_B2, status: Status.VERIFIED, split count: 6, time: 6.88
Output dim: 6, lower bound: -0.0038913, upper bound: 0.0037608
NS_A1_B2_A1_B2_B2_B1, status: Status.VERIFIED, split count: 6, time: 6.88
Output dim: 6, lower bound: -0.0038600, upper bound: 0.0037598
NS_A1_B2_A1_B2_B2_B2, status: Status.VERIFIED, split count: 6, time: 6.88
Output dim: 6, lower bound: -0.0038913, upper bound: 0.0037608

## BFS NS instance: NS_A1_B1_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0057605, 0.0073899, 0.0057796, 0.0074047, -0.0016442, 0.0016104
1: -0.0013684, 0.0023956, -0.0012003, 0.0024244, -0.0037928, 0.0035960
2: -0.0023951, 0.0238234, -0.0026271, 0.0233616, -0.0257567, 0.0264505
3: -0.0045722, -0.0022986, -0.0045456, -0.0022779, -0.0022942, 0.0022470
4: -0.0008472, 0.0101835, -0.0007185, 0.0102840, -0.0111313, 0.0109020
5: -0.0020399, 0.0001048, -0.0020549, -0.0000219, -0.0020181, 0.0021598
6: 0.9896712, 0.9939945, 0.9899877, 0.9940220, -0.0043507, 0.0040068
7: -0.0151681, 0.0050511, -0.0148807, 0.0052330, -0.0200932, 0.0199318
8: -0.0074024, 0.0025708, -0.0065270, 0.0026278, -0.0100302, 0.0090979
9: -0.0124601, 0.0005183, -0.0125739, 0.0002662, -0.0127263, 0.0130922

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 213

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039566, upper bound: 0.0039830
time: 2.08 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039566, upper bound: 0.0039702
time: 2.43 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0057568, 0.0074365, 0.0057653, 0.0074242, -0.0016674, 0.0016712
1: -0.0014015, 0.0024860, -0.0013262, 0.0024621, -0.0038636, 0.0038122
2: -0.0031239, 0.0239144, -0.0029308, 0.0237074, -0.0268313, 0.0268452
3: -0.0045774, -0.0022336, -0.0045655, -0.0022508, -0.0023266, 0.0023320
4: -0.0008726, 0.0104993, -0.0008149, 0.0104156, -0.0112882, 0.0113142
5: -0.0020871, 0.0001298, -0.0020746, 0.0000730, -0.0021601, 0.0022044
6: 0.9896089, 0.9940810, 0.9897507, 0.9940580, -0.0044491, 0.0043302
7: -0.0152247, 0.0056227, -0.0150959, 0.0054713, -0.0206959, 0.0207186
8: -0.0075748, 0.0027499, -0.0071824, 0.0027024, -0.0102773, 0.0099323
9: -0.0128176, 0.0005679, -0.0127229, 0.0004549, -0.0132725, 0.0132908

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039610, upper bound: 0.0040333
time: 2.61 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039610, upper bound: 0.0040207
time: 2.58 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_B1

### Backsubstitution after applying NS history:
0: 0.0057611, 0.0074335, 0.0056898, 0.0073454, -0.0015842, 0.0017437
1: -0.0013632, 0.0024801, -0.0019941, 0.0023094, -0.0036726, 0.0044741
2: -0.0030761, 0.0238091, -0.0016993, 0.0255425, -0.0286186, 0.0255084
3: -0.0045714, -0.0022378, -0.0046709, -0.0023608, -0.0022106, 0.0024331
4: -0.0008432, 0.0104786, -0.0013264, 0.0098820, -0.0107252, 0.0118050
5: -0.0020840, 0.0001009, -0.0019949, 0.0005765, -0.0026605, 0.0020958
6: 0.9896809, 0.9940752, 0.9884931, 0.9939119, -0.0042310, 0.0055822
7: -0.0151592, 0.0055853, -0.0162376, 0.0045053, -0.0196645, 0.0214192
8: -0.0073753, 0.0027382, -0.0106607, 0.0023998, -0.0097751, 0.0133989
9: -0.0127941, 0.0005105, -0.0121189, 0.0014567, -0.0142508, 0.0126293

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of NS_A1_B1_A1_B1_B2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038032, upper bound: 0.0038539
time: 2.26 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038026, upper bound: 0.0038350
time: 2.53 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_B2

### Backsubstitution after applying NS history:
0: 0.0057467, 0.0074528, 0.0056933, 0.0073912, -0.0016446, 0.0017595
1: -0.0014912, 0.0025174, -0.0019629, 0.0023982, -0.0038894, 0.0044803
2: -0.0033771, 0.0241607, -0.0024161, 0.0254567, -0.0288338, 0.0265768
3: -0.0045916, -0.0022109, -0.0046660, -0.0022968, -0.0022948, 0.0024551
4: -0.0009412, 0.0106091, -0.0013025, 0.0101926, -0.0111338, 0.0119116
5: -0.0021034, 0.0001974, -0.0020413, 0.0005530, -0.0026565, 0.0022387
6: 0.9894400, 0.9941109, 0.9885518, 0.9939969, -0.0045569, 0.0055591
7: -0.0153779, 0.0058214, -0.0161843, 0.0050675, -0.0204455, 0.0220057
8: -0.0080417, 0.0028121, -0.0104981, 0.0025760, -0.0106176, 0.0133103
9: -0.0129418, 0.0007024, -0.0124704, 0.0014098, -0.0143516, 0.0131728

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 213

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of NS_A1_B1_A1_B1_B2_B2_B1

### Relational analysis result of NS_A1_B1_A1_B1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038431, upper bound: 0.0038552
time: 2.53 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_B2_B2

### Relational analysis result of NS_A1_B1_A1_B1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038425, upper bound: 0.0038379
time: 2.59 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: 0.0057662, 0.0074282, 0.0057036, 0.0073688, -0.0016026, 0.0017247
1: -0.0013186, 0.0024699, -0.0018721, 0.0023547, -0.0036733, 0.0043420
2: -0.0029938, 0.0236865, -0.0020651, 0.0252075, -0.0282012, 0.0257516
3: -0.0045643, -0.0022452, -0.0046517, -0.0023281, -0.0022362, 0.0024065
4: -0.0008091, 0.0104429, -0.0015648, 0.0100405, -0.0108496, 0.0120077
5: -0.0020786, 0.0000673, -0.0020186, 0.0016078, -0.0036864, 0.0020858
6: 0.9897650, 0.9940655, 0.9880767, 0.9939553, -0.0041903, 0.0059887
7: -0.0150829, 0.0055207, -0.0160292, 0.0047922, -0.0198751, 0.0213020
8: -0.0071428, 0.0027179, -0.0100258, 0.0024897, -0.0096325, 0.0127437
9: -0.0127538, 0.0004435, -0.0122983, 0.0012752, -0.0140290, 0.0127418

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 213

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of NS_A1_B1_A1_B2_B1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039250, upper bound: 0.0039877
time: 2.91 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039053, upper bound: 0.0039707
time: 2.51 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: 0.0057517, 0.0074473, 0.0057005, 0.0074138, -0.0016621, 0.0017468
1: -0.0014462, 0.0025069, -0.0018990, 0.0024420, -0.0038882, 0.0044059
2: -0.0032925, 0.0240371, -0.0027691, 0.0252812, -0.0285737, 0.0268061
3: -0.0045845, -0.0022185, -0.0046559, -0.0022653, -0.0023192, 0.0024374
4: -0.0009068, 0.0105724, -0.0016239, 0.0103456, -0.0112523, 0.0121963
5: -0.0020980, 0.0001635, -0.0020641, 0.0017584, -0.0038564, 0.0022276
6: 0.9895248, 0.9941009, 0.9879513, 0.9940388, -0.0045140, 0.0061496
7: -0.0153010, 0.0057550, -0.0160751, 0.0053444, -0.0206454, 0.0218301
8: -0.0078073, 0.0027914, -0.0101656, 0.0026627, -0.0104700, 0.0129569
9: -0.0129003, 0.0006349, -0.0126436, 0.0013157, -0.0142160, 0.0132784

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039250, upper bound: 0.0039920
time: 2.38 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039752, upper bound: 0.0039752
time: 2.11 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: 0.0057662, 0.0074282, 0.0056182, 0.0081750, -0.0024088, 0.0018100
1: -0.0013186, 0.0024699, -0.0026265, 0.0023250, -0.0036436, 0.0050963
2: -0.0029938, 0.0236865, -0.0018255, 0.0272799, -0.0302737, 0.0255120
3: -0.0045643, -0.0022452, -0.0047708, -0.0023495, -0.0022148, 0.0025256
4: -0.0008091, 0.0104429, -0.0032247, 0.0099367, -0.0107457, 0.0136328
5: -0.0020786, 0.0000673, -0.0020031, 0.0058402, -0.0079188, 0.0020703
6: 0.9897650, 0.9940655, 0.9845495, 0.9939269, -0.0041619, 0.0095159
7: -0.0150829, 0.0055207, -0.0173187, 0.0046043, -0.0196871, 0.0224941
8: -0.0071428, 0.0027179, -0.0139540, 0.0024308, -0.0095737, 0.0166719
9: -0.0127538, 0.0004435, -0.0121807, 0.0024113, -0.0151651, 0.0126243

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of NS_A1_B1_A1_B2_B2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038154, upper bound: 0.0037902
time: 3.00 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038026, upper bound: 0.0037706
time: 2.48 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: 0.0057517, 0.0074473, 0.0056212, 0.0081196, -0.0023679, 0.0018261
1: -0.0014462, 0.0025069, -0.0025998, 0.0024077, -0.0038538, 0.0051067
2: -0.0032925, 0.0240371, -0.0024920, 0.0272067, -0.0304992, 0.0265290
3: -0.0045845, -0.0022185, -0.0047666, -0.0022900, -0.0022944, 0.0025481
4: -0.0009068, 0.0105724, -0.0031661, 0.0102255, -0.0111323, 0.0137385
5: -0.0020980, 0.0001635, -0.0020462, 0.0056906, -0.0077886, 0.0022096
6: 0.9895248, 0.9941009, 0.9846743, 0.9940059, -0.0044811, 0.0094267
7: -0.0153010, 0.0057550, -0.0172731, 0.0051271, -0.0204281, 0.0230281
8: -0.0078073, 0.0027914, -0.0138152, 0.0025946, -0.0104019, 0.0166065
9: -0.0129003, 0.0006349, -0.0125076, 0.0023712, -0.0152715, 0.0131425

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 97

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of NS_A1_B1_A1_B2_B2_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038521, upper bound: 0.0037919
time: 2.47 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038519, upper bound: 0.0037740
time: 2.70 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 7.11 seconds
NS_A1_B1_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 7.11
Output dim: 6, lower bound: -0.0039566, upper bound: 0.0039830
NS_A1_B1_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 7.11
Output dim: 6, lower bound: -0.0039566, upper bound: 0.0039702
NS_A1_B1_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 7.11
Output dim: 6, lower bound: -0.0039610, upper bound: 0.0040333
NS_A1_B1_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 7.11
Output dim: 6, lower bound: -0.0039610, upper bound: 0.0040207
NS_A1_B1_A1_B1_B2_B1_B1, status: Status.VERIFIED, split count: 7, time: 7.11
Output dim: 6, lower bound: -0.0038032, upper bound: 0.0038539
NS_A1_B1_A1_B1_B2_B1_B2, status: Status.VERIFIED, split count: 7, time: 7.11
Output dim: 6, lower bound: -0.0038026, upper bound: 0.0038350
NS_A1_B1_A1_B1_B2_B2_B1, status: Status.VERIFIED, split count: 7, time: 7.11
Output dim: 6, lower bound: -0.0038431, upper bound: 0.0038552
NS_A1_B1_A1_B1_B2_B2_B2, status: Status.VERIFIED, split count: 7, time: 7.11
Output dim: 6, lower bound: -0.0038425, upper bound: 0.0038379
NS_A1_B1_A1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 7.11
Output dim: 6, lower bound: -0.0039250, upper bound: 0.0039877
NS_A1_B1_A1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 7.11
Output dim: 6, lower bound: -0.0039053, upper bound: 0.0039707
NS_A1_B1_A1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 7.11
Output dim: 6, lower bound: -0.0039250, upper bound: 0.0039920
NS_A1_B1_A1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 7.11
Output dim: 6, lower bound: -0.0039752, upper bound: 0.0039752
NS_A1_B1_A1_B2_B2_B1_B1, status: Status.VERIFIED, split count: 7, time: 7.11
Output dim: 6, lower bound: -0.0038154, upper bound: 0.0037902
NS_A1_B1_A1_B2_B2_B1_B2, status: Status.VERIFIED, split count: 7, time: 7.11
Output dim: 6, lower bound: -0.0038026, upper bound: 0.0037706
NS_A1_B1_A1_B2_B2_B2_B1, status: Status.VERIFIED, split count: 7, time: 7.11
Output dim: 6, lower bound: -0.0038521, upper bound: 0.0037919
NS_A1_B1_A1_B2_B2_B2_B2, status: Status.VERIFIED, split count: 7, time: 7.11
Output dim: 6, lower bound: -0.0038519, upper bound: 0.0037740

## BFS NS instance: NS_A1_B1_A1_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0057675, 0.0073894, 0.0058232, 0.0074016, -0.0016341, 0.0015663
1: -0.0013071, 0.0023948, -0.0008147, 0.0024183, -0.0037254, 0.0032095
2: -0.0023882, 0.0236551, -0.0025775, 0.0223022, -0.0246904, 0.0262326
3: -0.0045625, -0.0022993, -0.0044848, -0.0022824, -0.0022801, 0.0021855
4: -0.0008003, 0.0101805, -0.0004232, 0.0102626, -0.0110629, 0.0106037
5: -0.0020395, 0.0000586, -0.0020517, -0.0003126, -0.0017269, 0.0021104
6: 0.9897866, 0.9939936, 0.9907137, 0.9940161, -0.0042294, 0.0032799
7: -0.0150633, 0.0050457, -0.0142216, 0.0051941, -0.0198848, 0.0192673
8: -0.0070833, 0.0025691, -0.0045190, 0.0026156, -0.0096989, 0.0070881
9: -0.0124568, 0.0004264, -0.0125496, -0.0003121, -0.0121446, 0.0129760

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 213

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039566, upper bound: 0.0039830
time: 2.14 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039566, upper bound: 0.0039830
time: 1.99 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0057776, 0.0073889, 0.0058289, 0.0074180, -0.0016404, 0.0015600
1: -0.0012175, 0.0023937, -0.0007641, 0.0024500, -0.0036675, 0.0031578
2: -0.0023796, 0.0234088, -0.0028337, 0.0221630, -0.0245427, 0.0262426
3: -0.0045484, -0.0023000, -0.0044768, -0.0022595, -0.0022889, 0.0021768
4: -0.0007317, 0.0101768, -0.0003844, 0.0103736, -0.0111053, 0.0105612
5: -0.0020389, -0.0000089, -0.0020683, -0.0003508, -0.0016882, 0.0020594
6: 0.9899552, 0.9939926, 0.9908091, 0.9940466, -0.0040913, 0.0031835
7: -0.0149101, 0.0050389, -0.0141350, 0.0053952, -0.0201420, 0.0191238
8: -0.0066166, 0.0025670, -0.0042552, 0.0026786, -0.0092952, 0.0068222
9: -0.0124525, 0.0002920, -0.0126753, -0.0003881, -0.0120645, 0.0129672

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 213

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039566, upper bound: 0.0039702
time: 2.45 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039566, upper bound: 0.0039704
time: 2.42 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0057644, 0.0074360, 0.0058099, 0.0074208, -0.0016564, 0.0016261
1: -0.0013344, 0.0024849, -0.0009325, 0.0024555, -0.0037899, 0.0034174
2: -0.0031152, 0.0237299, -0.0028781, 0.0226257, -0.0257409, 0.0266080
3: -0.0045668, -0.0022343, -0.0045034, -0.0022555, -0.0023113, 0.0022690
4: -0.0008212, 0.0104956, -0.0005134, 0.0103928, -0.0112140, 0.0110089
5: -0.0020865, 0.0000792, -0.0020712, -0.0002238, -0.0018627, 0.0021503
6: 0.9897352, 0.9940799, 0.9904920, 0.9940518, -0.0043165, 0.0035879
7: -0.0151099, 0.0056159, -0.0144229, 0.0054299, -0.0205398, 0.0200388
8: -0.0072251, 0.0027478, -0.0051322, 0.0026895, -0.0099146, 0.0078800
9: -0.0128133, 0.0004672, -0.0126970, -0.0001355, -0.0126778, 0.0131643

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039610, upper bound: 0.0040334
time: 1.99 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039610, upper bound: 0.0040333
time: 2.39 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0057762, 0.0074353, 0.0058162, 0.0074369, -0.0016607, 0.0016191
1: -0.0012299, 0.0024836, -0.0008760, 0.0024867, -0.0037166, 0.0033596
2: -0.0031045, 0.0234430, -0.0031296, 0.0224705, -0.0255750, 0.0265726
3: -0.0045503, -0.0022353, -0.0044945, -0.0022331, -0.0023173, 0.0022592
4: -0.0007412, 0.0104909, -0.0004701, 0.0105018, -0.0112430, 0.0109610
5: -0.0020858, 0.0000004, -0.0020874, -0.0002664, -0.0018194, 0.0020879
6: 0.9899319, 0.9940786, 0.9905984, 0.9940816, -0.0041497, 0.0034802
7: -0.0149314, 0.0056075, -0.0143263, 0.0056272, -0.0205586, 0.0199338
8: -0.0066812, 0.0027451, -0.0048381, 0.0027513, -0.0094326, 0.0075832
9: -0.0128081, 0.0003106, -0.0128204, -0.0002202, -0.0125878, 0.0131310

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039610, upper bound: 0.0040209
time: 2.27 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039610, upper bound: 0.0040207
time: 2.29 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_B1_B1

### Backsubstitution after applying NS history:
0: 0.0057737, 0.0074277, 0.0057436, 0.0073660, -0.0015923, 0.0016840
1: -0.0012517, 0.0024688, -0.0015179, 0.0023494, -0.0036010, 0.0039867
2: -0.0029853, 0.0235026, -0.0020219, 0.0242341, -0.0272193, 0.0255246
3: -0.0045537, -0.0022459, -0.0045958, -0.0023320, -0.0022218, 0.0023498
4: -0.0007578, 0.0104393, -0.0009617, 0.0100218, -0.0107796, 0.0114010
5: -0.0020781, 0.0000168, -0.0020158, 0.0002175, -0.0022956, 0.0020326
6: 0.9898911, 0.9940645, 0.9893898, 0.9939502, -0.0040591, 0.0046747
7: -0.0149685, 0.0055140, -0.0154236, 0.0047584, -0.0197268, 0.0206541
8: -0.0067943, 0.0027158, -0.0081807, 0.0024791, -0.0092734, 0.0108966
9: -0.0127496, 0.0003432, -0.0122771, 0.0007424, -0.0134920, 0.0126203

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 213

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of NS_A1_B1_A1_B2_B1_B1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038470, upper bound: 0.0039416
time: 2.60 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038462, upper bound: 0.0039089
time: 2.24 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_B1_B2

### Backsubstitution after applying NS history:
0: 0.0057856, 0.0074270, 0.0057477, 0.0073827, -0.0015971, 0.0016792
1: -0.0011468, 0.0024675, -0.0014816, 0.0023818, -0.0035286, 0.0039491
2: -0.0029744, 0.0232146, -0.0022832, 0.0241345, -0.0271089, 0.0254978
3: -0.0045372, -0.0022469, -0.0045901, -0.0023087, -0.0022285, 0.0023431
4: -0.0006775, 0.0104346, -0.0009339, 0.0101351, -0.0108126, 0.0113685
5: -0.0020774, -0.0000622, -0.0020327, 0.0001902, -0.0022676, 0.0019704
6: 0.9900885, 0.9940631, 0.9894580, 0.9939812, -0.0038927, 0.0046052
7: -0.0147892, 0.0055055, -0.0153616, 0.0049633, -0.0197526, 0.0207656
8: -0.0062483, 0.0027132, -0.0079920, 0.0025433, -0.0087916, 0.0107052
9: -0.0127443, 0.0001859, -0.0124053, 0.0006881, -0.0134324, 0.0125912

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of NS_A1_B1_A1_B2_B1_B1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038459, upper bound: 0.0039261
time: 2.39 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B1_B2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038454, upper bound: 0.0038976
time: 2.35 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_B2_B1

### Backsubstitution after applying NS history:
0: 0.0057593, 0.0074468, 0.0057448, 0.0074106, -0.0016513, 0.0017020
1: -0.0013792, 0.0025058, -0.0015079, 0.0024357, -0.0038149, 0.0040137
2: -0.0032838, 0.0238531, -0.0027182, 0.0242067, -0.0274905, 0.0265713
3: -0.0045739, -0.0022193, -0.0045942, -0.0022698, -0.0023041, 0.0023749
4: -0.0008555, 0.0105686, -0.0009541, 0.0103235, -0.0111790, 0.0115227
5: -0.0020974, 0.0001130, -0.0020608, 0.0002100, -0.0023074, 0.0021738
6: 0.9896508, 0.9941000, 0.9894086, 0.9940328, -0.0043820, 0.0046914
7: -0.0151866, 0.0057482, -0.0154066, 0.0053045, -0.0204911, 0.0211547
8: -0.0074586, 0.0027892, -0.0081289, 0.0026502, -0.0101089, 0.0109181
9: -0.0128960, 0.0005345, -0.0126186, 0.0007275, -0.0136235, 0.0131531

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039250, upper bound: 0.0039378
time: 2.58 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039250, upper bound: 0.0039919
time: 2.97 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_B2_B2

### Backsubstitution after applying NS history:
0: 0.0057711, 0.0074461, 0.0057517, 0.0074257, -0.0016546, 0.0016944
1: -0.0012747, 0.0025045, -0.0014467, 0.0024650, -0.0037397, 0.0039511
2: -0.0032728, 0.0235659, -0.0029544, 0.0240384, -0.0273112, 0.0265203
3: -0.0045574, -0.0022203, -0.0045845, -0.0022487, -0.0023087, 0.0023643
4: -0.0007754, 0.0105639, -0.0009072, 0.0104259, -0.0112013, 0.0114710
5: -0.0020967, 0.0000342, -0.0020761, 0.0001638, -0.0022605, 0.0021103
6: 0.9898477, 0.9940985, 0.9895239, 0.9940609, -0.0042132, 0.0045747
7: -0.0150078, 0.0057395, -0.0153018, 0.0054898, -0.0204977, 0.0210414
8: -0.0069143, 0.0027865, -0.0078099, 0.0027083, -0.0096225, 0.0105964
9: -0.0128906, 0.0003777, -0.0127345, 0.0006356, -0.0135263, 0.0131122

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039240, upper bound: 0.0039240
time: 4.87 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039240, upper bound: 0.0039751
time: 2.59 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 9.84 seconds
NS_A1_B1_A1_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 9.84
Output dim: 6, lower bound: -0.0039566, upper bound: 0.0039830
NS_A1_B1_A1_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 9.84
Output dim: 6, lower bound: -0.0039566, upper bound: 0.0039830
NS_A1_B1_A1_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 9.84
Output dim: 6, lower bound: -0.0039566, upper bound: 0.0039702
NS_A1_B1_A1_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 9.84
Output dim: 6, lower bound: -0.0039566, upper bound: 0.0039704
NS_A1_B1_A1_B1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 9.84
Output dim: 6, lower bound: -0.0039610, upper bound: 0.0040334
NS_A1_B1_A1_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 9.84
Output dim: 6, lower bound: -0.0039610, upper bound: 0.0040333
NS_A1_B1_A1_B1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 9.84
Output dim: 6, lower bound: -0.0039610, upper bound: 0.0040209
NS_A1_B1_A1_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 9.84
Output dim: 6, lower bound: -0.0039610, upper bound: 0.0040207
NS_A1_B1_A1_B2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 9.84
Output dim: 6, lower bound: -0.0038470, upper bound: 0.0039416
NS_A1_B1_A1_B2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 9.84
Output dim: 6, lower bound: -0.0038462, upper bound: 0.0039089
NS_A1_B1_A1_B2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 9.84
Output dim: 6, lower bound: -0.0038459, upper bound: 0.0039261
NS_A1_B1_A1_B2_B1_B1_B2_B2, status: Status.VERIFIED, split count: 8, time: 9.84
Output dim: 6, lower bound: -0.0038454, upper bound: 0.0038976
NS_A1_B1_A1_B2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 9.84
Output dim: 6, lower bound: -0.0039250, upper bound: 0.0039378
NS_A1_B1_A1_B2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 9.84
Output dim: 6, lower bound: -0.0039250, upper bound: 0.0039919
NS_A1_B1_A1_B2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 9.84
Output dim: 6, lower bound: -0.0039240, upper bound: 0.0039240
NS_A1_B1_A1_B2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 9.84
Output dim: 6, lower bound: -0.0039240, upper bound: 0.0039751

## BFS NS instance: NS_A1_B1_A1_B1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0057851, 0.0073606, 0.0058232, 0.0074016, -0.0016165, 0.0015374
1: -0.0011515, 0.0023388, -0.0008147, 0.0024183, -0.0035698, 0.0031536
2: -0.0019368, 0.0232275, -0.0025775, 0.0223022, -0.0242390, 0.0258050
3: -0.0045379, -0.0023396, -0.0044848, -0.0022824, -0.0022556, 0.0021452
4: -0.0006811, 0.0099849, -0.0004232, 0.0102626, -0.0109437, 0.0104081
5: -0.0020103, -0.0000587, -0.0020517, -0.0003126, -0.0016977, 0.0019930
6: 0.9900796, 0.9939401, 0.9907137, 0.9940161, -0.0039365, 0.0032264
7: -0.0147973, 0.0046916, -0.0142216, 0.0051941, -0.0195816, 0.0189132
8: -0.0062729, 0.0024582, -0.0045190, 0.0026156, -0.0088885, 0.0069772
9: -0.0122353, 0.0001930, -0.0125496, -0.0003121, -0.0119232, 0.0127426

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 184

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038737, upper bound: 0.0039309
time: 2.41 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038841, upper bound: 0.0039150
time: 2.59 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0057104, 0.0073683, 0.0058232, 0.0074016, -0.0016911, 0.0015451
1: -0.0018112, 0.0023538, -0.0008147, 0.0024183, -0.0042294, 0.0031686
2: -0.0020577, 0.0250399, -0.0025775, 0.0223022, -0.0243599, 0.0276174
3: -0.0046421, -0.0023288, -0.0044848, -0.0022824, -0.0023597, 0.0021560
4: -0.0011863, 0.0100373, -0.0004232, 0.0102626, -0.0114489, 0.0104605
5: -0.0020181, 0.0004386, -0.0020517, -0.0003126, -0.0017055, 0.0024904
6: 0.9888374, 0.9939544, 0.9907137, 0.9940161, -0.0051786, 0.0032407
7: -0.0159250, 0.0047864, -0.0142216, 0.0051941, -0.0208060, 0.0190080
8: -0.0097082, 0.0024879, -0.0045190, 0.0026156, -0.0123238, 0.0070069
9: -0.0122947, 0.0011823, -0.0125496, -0.0003121, -0.0119825, 0.0137319

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 184

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038737, upper bound: 0.0039307
time: 2.30 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038841, upper bound: 0.0039147
time: 2.31 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0057952, 0.0073600, 0.0058289, 0.0074180, -0.0016228, 0.0015311
1: -0.0010618, 0.0023378, -0.0007641, 0.0024500, -0.0035118, 0.0031018
2: -0.0019282, 0.0229810, -0.0028337, 0.0221630, -0.0240912, 0.0258147
3: -0.0045238, -0.0023404, -0.0044768, -0.0022595, -0.0022643, 0.0021364
4: -0.0006124, 0.0099812, -0.0003844, 0.0103736, -0.0109860, 0.0103656
5: -0.0020097, -0.0001263, -0.0020683, -0.0003508, -0.0016590, 0.0019420
6: 0.9902486, 0.9939390, 0.9908091, 0.9940466, -0.0037980, 0.0031298
7: -0.0146439, 0.0046848, -0.0141350, 0.0053952, -0.0198366, 0.0187447
8: -0.0058055, 0.0024561, -0.0042552, 0.0026786, -0.0084842, 0.0067113
9: -0.0122311, 0.0000584, -0.0126753, -0.0003881, -0.0118430, 0.0127337

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 184

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038731, upper bound: 0.0039201
time: 2.59 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038841, upper bound: 0.0039051
time: 2.58 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0057207, 0.0073677, 0.0058289, 0.0074180, -0.0016973, 0.0015388
1: -0.0017205, 0.0023527, -0.0007641, 0.0024500, -0.0041705, 0.0031168
2: -0.0020486, 0.0247907, -0.0028337, 0.0221630, -0.0242117, 0.0276244
3: -0.0046278, -0.0023296, -0.0044768, -0.0022595, -0.0023683, 0.0021472
4: -0.0011169, 0.0100334, -0.0003844, 0.0103736, -0.0114905, 0.0104178
5: -0.0020175, 0.0003703, -0.0020683, -0.0003508, -0.0016667, 0.0024385
6: 0.9890082, 0.9939533, 0.9908091, 0.9940466, -0.0050383, 0.0031442
7: -0.0157699, 0.0047793, -0.0141350, 0.0053952, -0.0210590, 0.0189136
8: -0.0092358, 0.0024857, -0.0042552, 0.0026786, -0.0119144, 0.0067409
9: -0.0122902, 0.0010463, -0.0126753, -0.0003881, -0.0119021, 0.0137216

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038731, upper bound: 0.0039201
time: 2.33 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038841, upper bound: 0.0039051
time: 2.21 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0057829, 0.0074076, 0.0058099, 0.0074208, -0.0016379, 0.0015978
1: -0.0011709, 0.0024300, -0.0009325, 0.0024555, -0.0036264, 0.0033625
2: -0.0026719, 0.0232807, -0.0028781, 0.0226257, -0.0252976, 0.0261587
3: -0.0045410, -0.0022739, -0.0045034, -0.0022555, -0.0022855, 0.0022294
4: -0.0006959, 0.0103035, -0.0005134, 0.0103928, -0.0110887, 0.0108168
5: -0.0020578, -0.0000441, -0.0020712, -0.0002238, -0.0018340, 0.0020271
6: 0.9900432, 0.9940273, 0.9904920, 0.9940518, -0.0040085, 0.0035353
7: -0.0148304, 0.0052682, -0.0144229, 0.0054299, -0.0202603, 0.0196911
8: -0.0063736, 0.0026388, -0.0051322, 0.0026895, -0.0090631, 0.0077711
9: -0.0125959, 0.0002220, -0.0126970, -0.0001355, -0.0124604, 0.0129190

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039063, upper bound: 0.0040284
time: 2.42 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039063, upper bound: 0.0040334
time: 2.38 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0057081, 0.0074132, 0.0058099, 0.0074208, -0.0017127, 0.0016034
1: -0.0018318, 0.0024409, -0.0009325, 0.0024555, -0.0042874, 0.0033734
2: -0.0027599, 0.0250967, -0.0028781, 0.0226257, -0.0253856, 0.0279748
3: -0.0046453, -0.0022661, -0.0045034, -0.0022555, -0.0023898, 0.0022373
4: -0.0012022, 0.0103416, -0.0005134, 0.0103928, -0.0115950, 0.0108549
5: -0.0020635, 0.0004542, -0.0020712, -0.0002238, -0.0018397, 0.0025254
6: 0.9887987, 0.9940377, 0.9904920, 0.9940518, -0.0052531, 0.0035458
7: -0.0159603, 0.0053372, -0.0144229, 0.0054299, -0.0213902, 0.0197601
8: -0.0098158, 0.0026605, -0.0051322, 0.0026895, -0.0125054, 0.0077927
9: -0.0126390, 0.0012133, -0.0126970, -0.0001355, -0.0125035, 0.0139104

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039063, upper bound: 0.0040284
time: 2.44 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039063, upper bound: 0.0040333
time: 2.38 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0057946, 0.0074069, 0.0058162, 0.0074369, -0.0016423, 0.0015907
1: -0.0010671, 0.0024286, -0.0008760, 0.0024867, -0.0035538, 0.0033046
2: -0.0026611, 0.0229955, -0.0031296, 0.0224705, -0.0251317, 0.0261251
3: -0.0045246, -0.0022749, -0.0044945, -0.0022331, -0.0022916, 0.0022196
4: -0.0006164, 0.0102988, -0.0004701, 0.0105018, -0.0111183, 0.0107689
5: -0.0020571, -0.0001223, -0.0020874, -0.0002664, -0.0017907, 0.0019651
6: 0.9902386, 0.9940260, 0.9905984, 0.9940816, -0.0038430, 0.0034276
7: -0.0146530, 0.0052598, -0.0143263, 0.0056272, -0.0202802, 0.0195861
8: -0.0058331, 0.0026362, -0.0048381, 0.0027513, -0.0085844, 0.0074743
9: -0.0125906, 0.0000663, -0.0128204, -0.0002202, -0.0123704, 0.0128867

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039053, upper bound: 0.0040128
time: 2.36 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039053, upper bound: 0.0040208
time: 2.48 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0057200, 0.0074125, 0.0058162, 0.0074369, -0.0017169, 0.0015963
1: -0.0017266, 0.0024395, -0.0008760, 0.0024867, -0.0042133, 0.0033155
2: -0.0027487, 0.0248075, -0.0031296, 0.0224705, -0.0252192, 0.0279371
3: -0.0046287, -0.0022671, -0.0044945, -0.0022331, -0.0023957, 0.0022274
4: -0.0011215, 0.0103367, -0.0004701, 0.0105018, -0.0116234, 0.0108068
5: -0.0020628, 0.0003749, -0.0020874, -0.0002664, -0.0017964, 0.0024623
6: 0.9889967, 0.9940363, 0.9905984, 0.9940816, -0.0050849, 0.0034379
7: -0.0157804, 0.0053284, -0.0143263, 0.0056272, -0.0214076, 0.0196548
8: -0.0092677, 0.0026577, -0.0048381, 0.0027513, -0.0120190, 0.0074958
9: -0.0126336, 0.0010555, -0.0128204, -0.0002202, -0.0124133, 0.0138759

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039053, upper bound: 0.0040130
time: 2.43 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039053, upper bound: 0.0040207
time: 2.45 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_B1_B1_B1

### Backsubstitution after applying NS history:
0: 0.0057784, 0.0074229, 0.0057624, 0.0073439, -0.0015655, 0.0016604
1: -0.0012107, 0.0024595, -0.0013516, 0.0023065, -0.0035173, 0.0038112
2: -0.0029104, 0.0233902, -0.0016763, 0.0237773, -0.0266878, 0.0250665
3: -0.0045473, -0.0022526, -0.0045695, -0.0023629, -0.0021844, 0.0023169
4: -0.0007265, 0.0104068, -0.0008344, 0.0098721, -0.0105985, 0.0112412
5: -0.0020733, -0.0000140, -0.0019934, 0.0000922, -0.0021654, 0.0019794
6: 0.9899681, 0.9940555, 0.9897028, 0.9939092, -0.0039411, 0.0043527
7: -0.0148985, 0.0054553, -0.0151394, 0.0044873, -0.0193858, 0.0202519
8: -0.0065812, 0.0026975, -0.0073150, 0.0023942, -0.0089753, 0.0100125
9: -0.0127129, 0.0002818, -0.0121076, 0.0004931, -0.0132060, 0.0123894

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 97

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A1_B2_B1_B1_B1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038282, upper bound: 0.0039268
time: 2.35 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B1_B1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038282, upper bound: 0.0039416
time: 2.41 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_B1_B1_B2

### Backsubstitution after applying NS history:
0: 0.0057804, 0.0074214, 0.0057350, 0.0073449, -0.0015645, 0.0016864
1: -0.0011926, 0.0024566, -0.0015941, 0.0023085, -0.0035011, 0.0040507
2: -0.0028869, 0.0233403, -0.0016925, 0.0244436, -0.0273305, 0.0250328
3: -0.0045444, -0.0022547, -0.0046078, -0.0023614, -0.0021830, 0.0023531
4: -0.0007126, 0.0103966, -0.0010201, 0.0098791, -0.0105916, 0.0114167
5: -0.0020717, -0.0000277, -0.0019945, 0.0002750, -0.0023467, 0.0019668
6: 0.9900022, 0.9940528, 0.9892462, 0.9939111, -0.0039089, 0.0048066
7: -0.0148675, 0.0054368, -0.0155539, 0.0044999, -0.0193674, 0.0206028
8: -0.0064867, 0.0026917, -0.0085778, 0.0023981, -0.0088849, 0.0112695
9: -0.0127013, 0.0002546, -0.0121155, 0.0008568, -0.0135581, 0.0123701

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A1_B2_B1_B1_B1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038284, upper bound: 0.0038949
time: 2.58 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B1_B1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038284, upper bound: 0.0039089
time: 2.66 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_B1_B2_B1

### Backsubstitution after applying NS history:
0: 0.0057903, 0.0074222, 0.0057675, 0.0073607, -0.0015705, 0.0016547
1: -0.0011057, 0.0024582, -0.0013067, 0.0023391, -0.0034449, 0.0037649
2: -0.0028995, 0.0231017, -0.0019393, 0.0236539, -0.0265534, 0.0250410
3: -0.0045307, -0.0022536, -0.0045624, -0.0023394, -0.0021914, 0.0023088
4: -0.0006460, 0.0104021, -0.0008000, 0.0099860, -0.0106320, 0.0112021
5: -0.0020726, -0.0000932, -0.0020104, 0.0000583, -0.0021309, 0.0019172
6: 0.9901658, 0.9940543, 0.9897873, 0.9939403, -0.0037745, 0.0042669
7: -0.0147190, 0.0054468, -0.0150626, 0.0046936, -0.0194126, 0.0203460
8: -0.0060343, 0.0026948, -0.0070811, 0.0024588, -0.0084931, 0.0097759
9: -0.0127076, 0.0001243, -0.0122366, 0.0004257, -0.0131333, 0.0123609

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 97

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A1_B2_B1_B1_B2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038266, upper bound: 0.0039132
time: 2.37 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B1_B2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038266, upper bound: 0.0039262
time: 2.61 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_B2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0057723, 0.0073846, 0.0057448, 0.0074106, -0.0016383, 0.0016399
1: -0.0012645, 0.0023854, -0.0015079, 0.0024357, -0.0037002, 0.0038933
2: -0.0023126, 0.0235378, -0.0027182, 0.0242067, -0.0265193, 0.0262560
3: -0.0045558, -0.0023060, -0.0045942, -0.0022698, -0.0022860, 0.0022882
4: -0.0007676, 0.0101478, -0.0009541, 0.0103235, -0.0110911, 0.0111018
5: -0.0020346, 0.0000265, -0.0020608, 0.0002100, -0.0022446, 0.0020873
6: 0.9898670, 0.9939846, 0.9894086, 0.9940328, -0.0041658, 0.0045761
7: -0.0149904, 0.0049863, -0.0154066, 0.0053045, -0.0201646, 0.0203929
8: -0.0068610, 0.0025505, -0.0081289, 0.0026502, -0.0095112, 0.0106795
9: -0.0124197, 0.0003624, -0.0126186, 0.0007275, -0.0131472, 0.0129810

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 213

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038470, upper bound: 0.0038782
time: 2.63 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038462, upper bound: 0.0038532
time: 2.40 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0057694, 0.0074305, 0.0057448, 0.0074106, -0.0016411, 0.0016858
1: -0.0012896, 0.0024744, -0.0015079, 0.0024357, -0.0037253, 0.0039823
2: -0.0030302, 0.0236070, -0.0027182, 0.0242067, -0.0272369, 0.0263252
3: -0.0045597, -0.0022419, -0.0045942, -0.0022698, -0.0022900, 0.0023523
4: -0.0007869, 0.0104587, -0.0009541, 0.0103235, -0.0111104, 0.0114128
5: -0.0020810, 0.0000455, -0.0020608, 0.0002100, -0.0022910, 0.0021063
6: 0.9898196, 0.9940698, 0.9894086, 0.9940328, -0.0042132, 0.0046613
7: -0.0150334, 0.0055492, -0.0154066, 0.0053045, -0.0203380, 0.0209558
8: -0.0069922, 0.0027269, -0.0081289, 0.0026502, -0.0096424, 0.0108558
9: -0.0127716, 0.0004001, -0.0126186, 0.0007275, -0.0134991, 0.0130188

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038470, upper bound: 0.0039486
time: 2.62 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038462, upper bound: 0.0039258
time: 2.74 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_B2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0057825, 0.0073841, 0.0057517, 0.0074257, -0.0016432, 0.0016324
1: -0.0011745, 0.0023843, -0.0014467, 0.0024650, -0.0036395, 0.0038310
2: -0.0023040, 0.0232907, -0.0029544, 0.0240384, -0.0263424, 0.0262451
3: -0.0045416, -0.0023068, -0.0045845, -0.0022487, -0.0022929, 0.0022777
4: -0.0006987, 0.0101440, -0.0009072, 0.0104259, -0.0111246, 0.0110512
5: -0.0020340, -0.0000413, -0.0020761, 0.0001638, -0.0021979, 0.0020348
6: 0.9900364, 0.9939836, 0.9895239, 0.9940609, -0.0040245, 0.0044597
7: -0.0148366, 0.0049796, -0.0153018, 0.0054898, -0.0203264, 0.0202814
8: -0.0063926, 0.0025484, -0.0078099, 0.0027083, -0.0091008, 0.0103583
9: -0.0124154, 0.0002275, -0.0127345, 0.0006356, -0.0130511, 0.0129619

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 213

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038459, upper bound: 0.0038650
time: 2.87 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038454, upper bound: 0.0038452
time: 2.55 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0057813, 0.0074299, 0.0057517, 0.0074257, -0.0016444, 0.0016782
1: -0.0011851, 0.0024731, -0.0014467, 0.0024650, -0.0036501, 0.0039197
2: -0.0030195, 0.0233197, -0.0029544, 0.0240384, -0.0270579, 0.0262741
3: -0.0045432, -0.0022429, -0.0045845, -0.0022487, -0.0022945, 0.0023416
4: -0.0007068, 0.0104541, -0.0009072, 0.0104259, -0.0111327, 0.0113613
5: -0.0020803, -0.0000334, -0.0020761, 0.0001638, -0.0022442, 0.0020427
6: 0.9900165, 0.9940685, 0.9895239, 0.9940609, -0.0040444, 0.0045446
7: -0.0148547, 0.0055409, -0.0153018, 0.0054898, -0.0203445, 0.0208427
8: -0.0064476, 0.0027243, -0.0078099, 0.0027083, -0.0091559, 0.0105342
9: -0.0127664, 0.0002433, -0.0127345, 0.0006356, -0.0134020, 0.0129778

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038459, upper bound: 0.0039344
time: 2.57 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038454, upper bound: 0.0039170
time: 2.53 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 7.18 seconds
NS_A1_B1_A1_B1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 9, time: 7.18
Output dim: 6, lower bound: -0.0038737, upper bound: 0.0039309
NS_A1_B1_A1_B1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 7.18
Output dim: 6, lower bound: -0.0038841, upper bound: 0.0039150
NS_A1_B1_A1_B1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 7.18
Output dim: 6, lower bound: -0.0038737, upper bound: 0.0039307
NS_A1_B1_A1_B1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 7.18
Output dim: 6, lower bound: -0.0038841, upper bound: 0.0039147
NS_A1_B1_A1_B1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 9, time: 7.18
Output dim: 6, lower bound: -0.0038731, upper bound: 0.0039201
NS_A1_B1_A1_B1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 9, time: 7.18
Output dim: 6, lower bound: -0.0038841, upper bound: 0.0039051
NS_A1_B1_A1_B1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 7.18
Output dim: 6, lower bound: -0.0038731, upper bound: 0.0039201
NS_A1_B1_A1_B1_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 9, time: 7.18
Output dim: 6, lower bound: -0.0038841, upper bound: 0.0039051
NS_A1_B1_A1_B1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 9, time: 7.18
Output dim: 6, lower bound: -0.0039063, upper bound: 0.0040284
NS_A1_B1_A1_B1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 7.18
Output dim: 6, lower bound: -0.0039063, upper bound: 0.0040334
NS_A1_B1_A1_B1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 7.18
Output dim: 6, lower bound: -0.0039063, upper bound: 0.0040284
NS_A1_B1_A1_B1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 7.18
Output dim: 6, lower bound: -0.0039063, upper bound: 0.0040333
NS_A1_B1_A1_B1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 9, time: 7.18
Output dim: 6, lower bound: -0.0039053, upper bound: 0.0040128
NS_A1_B1_A1_B1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 7.18
Output dim: 6, lower bound: -0.0039053, upper bound: 0.0040208
NS_A1_B1_A1_B1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 7.18
Output dim: 6, lower bound: -0.0039053, upper bound: 0.0040130
NS_A1_B1_A1_B1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 7.18
Output dim: 6, lower bound: -0.0039053, upper bound: 0.0040207
NS_A1_B1_A1_B2_B1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 7.18
Output dim: 6, lower bound: -0.0038282, upper bound: 0.0039268
NS_A1_B1_A1_B2_B1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 7.18
Output dim: 6, lower bound: -0.0038282, upper bound: 0.0039416
NS_A1_B1_A1_B2_B1_B1_B1_B2_A1, status: Status.VERIFIED, split count: 9, time: 7.18
Output dim: 6, lower bound: -0.0038284, upper bound: 0.0038949
NS_A1_B1_A1_B2_B1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 7.18
Output dim: 6, lower bound: -0.0038284, upper bound: 0.0039089
NS_A1_B1_A1_B2_B1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 7.18
Output dim: 6, lower bound: -0.0038266, upper bound: 0.0039132
NS_A1_B1_A1_B2_B1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 7.18
Output dim: 6, lower bound: -0.0038266, upper bound: 0.0039262
NS_A1_B1_A1_B2_B1_B2_B1_A1_B1, status: Status.VERIFIED, split count: 9, time: 7.18
Output dim: 6, lower bound: -0.0038470, upper bound: 0.0038782
NS_A1_B1_A1_B2_B1_B2_B1_A1_B2, status: Status.VERIFIED, split count: 9, time: 7.18
Output dim: 6, lower bound: -0.0038462, upper bound: 0.0038532
NS_A1_B1_A1_B2_B1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 7.18
Output dim: 6, lower bound: -0.0038470, upper bound: 0.0039486
NS_A1_B1_A1_B2_B1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 7.18
Output dim: 6, lower bound: -0.0038462, upper bound: 0.0039258
NS_A1_B1_A1_B2_B1_B2_B2_A1_B1, status: Status.VERIFIED, split count: 9, time: 7.18
Output dim: 6, lower bound: -0.0038459, upper bound: 0.0038650
NS_A1_B1_A1_B2_B1_B2_B2_A1_B2, status: Status.VERIFIED, split count: 9, time: 7.18
Output dim: 6, lower bound: -0.0038454, upper bound: 0.0038452
NS_A1_B1_A1_B2_B1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 7.18
Output dim: 6, lower bound: -0.0038459, upper bound: 0.0039344
NS_A1_B1_A1_B2_B1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 7.18
Output dim: 6, lower bound: -0.0038454, upper bound: 0.0039170

## BFS NS instance: NS_A1_B1_A1_B1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0057891, 0.0073558, 0.0058416, 0.0073784, -0.0015893, 0.0015141
1: -0.0011160, 0.0023295, -0.0006517, 0.0023733, -0.0034892, 0.0029812
2: -0.0018619, 0.0231298, -0.0022148, 0.0218542, -0.0237161, 0.0253446
3: -0.0045323, -0.0023463, -0.0044590, -0.0023148, -0.0022176, 0.0021128
4: -0.0006539, 0.0099525, -0.0002983, 0.0101054, -0.0107593, 0.0102508
5: -0.0020054, -0.0000855, -0.0020283, -0.0004355, -0.0015699, 0.0019428
6: 0.9901466, 0.9939312, 0.9910207, 0.9939730, -0.0038264, 0.0029105
7: -0.0147365, 0.0046328, -0.0139429, 0.0049097, -0.0192014, 0.0185757
8: -0.0060876, 0.0024398, -0.0036698, 0.0025265, -0.0086141, 0.0061096
9: -0.0121986, 0.0001396, -0.0123717, -0.0005567, -0.0116419, 0.0125113

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 184

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038411, upper bound: 0.0039345
time: 2.48 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039651, upper bound: 0.0039537
time: 2.23 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0057904, 0.0073543, 0.0058065, 0.0073819, -0.0015915, 0.0015478
1: -0.0011043, 0.0023266, -0.0009623, 0.0023801, -0.0034844, 0.0032890
2: -0.0018386, 0.0230978, -0.0022699, 0.0227077, -0.0245463, 0.0253677
3: -0.0045305, -0.0023484, -0.0045081, -0.0023098, -0.0022207, 0.0021597
4: -0.0006450, 0.0099424, -0.0005362, 0.0101293, -0.0107742, 0.0104786
5: -0.0020039, -0.0000943, -0.0020318, -0.0002013, -0.0018026, 0.0019376
6: 0.9901685, 0.9939284, 0.9904358, 0.9939796, -0.0038111, 0.0034926
7: -0.0147166, 0.0046145, -0.0144739, 0.0049529, -0.0192858, 0.0190884
8: -0.0060270, 0.0024340, -0.0052877, 0.0025400, -0.0085670, 0.0077217
9: -0.0121872, 0.0001222, -0.0123987, -0.0000907, -0.0120964, 0.0125209

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 97

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038408, upper bound: 0.0038960
time: 2.75 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039758, upper bound: 0.0039314
time: 2.55 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0057145, 0.0073637, 0.0058416, 0.0073784, -0.0016638, 0.0015220
1: -0.0017754, 0.0023448, -0.0006517, 0.0023733, -0.0041486, 0.0029965
2: -0.0019854, 0.0249415, -0.0022148, 0.0218542, -0.0238396, 0.0271563
3: -0.0046364, -0.0023352, -0.0044590, -0.0023148, -0.0023217, 0.0021238
4: -0.0011589, 0.0100060, -0.0002983, 0.0101054, -0.0112643, 0.0103043
5: -0.0020134, 0.0004116, -0.0020283, -0.0004355, -0.0015779, 0.0024399
6: 0.9889048, 0.9939458, 0.9910207, 0.9939730, -0.0050682, 0.0029251
7: -0.0158637, 0.0047297, -0.0139429, 0.0049097, -0.0204254, 0.0186726
8: -0.0095217, 0.0024701, -0.0036698, 0.0025265, -0.0120482, 0.0061399
9: -0.0122592, 0.0011286, -0.0123717, -0.0005567, -0.0117025, 0.0135003

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036326, upper bound: 0.0038656
time: 2.66 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038548, upper bound: 0.0039157
time: 2.57 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0057162, 0.0073618, 0.0058065, 0.0073819, -0.0016657, 0.0015553
1: -0.0017603, 0.0023411, -0.0009623, 0.0023801, -0.0041404, 0.0033035
2: -0.0019554, 0.0249000, -0.0022699, 0.0227077, -0.0246631, 0.0271700
3: -0.0046340, -0.0023379, -0.0045081, -0.0023098, -0.0023242, 0.0021701
4: -0.0011473, 0.0099930, -0.0005362, 0.0101293, -0.0112766, 0.0105292
5: -0.0020115, 0.0004003, -0.0020318, -0.0002013, -0.0018102, 0.0024321
6: 0.9889334, 0.9939423, 0.9904358, 0.9939796, -0.0050462, 0.0035065
7: -0.0158379, 0.0047062, -0.0144739, 0.0049529, -0.0205029, 0.0191800
8: -0.0094430, 0.0024627, -0.0052877, 0.0025400, -0.0119831, 0.0077504
9: -0.0122445, 0.0011060, -0.0123987, -0.0000907, -0.0121537, 0.0135047

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036299, upper bound: 0.0038324
time: 2.46 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B1_A2_B2_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038663, upper bound: 0.0038993
time: 2.58 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0057992, 0.0073552, 0.0058485, 0.0073954, -0.0015961, 0.0015067
1: -0.0010264, 0.0023285, -0.0005905, 0.0024062, -0.0034326, 0.0029189
2: -0.0018533, 0.0228837, -0.0024805, 0.0216860, -0.0235392, 0.0253642
3: -0.0045182, -0.0023470, -0.0044494, -0.0022910, -0.0022272, 0.0021023
4: -0.0005853, 0.0099487, -0.0002514, 0.0102205, -0.0108058, 0.0102001
5: -0.0020049, -0.0001530, -0.0020454, -0.0004817, -0.0015232, 0.0018924
6: 0.9903153, 0.9939302, 0.9911360, 0.9940045, -0.0036893, 0.0027941
7: -0.0145834, 0.0046261, -0.0138382, 0.0051181, -0.0194719, 0.0182509
8: -0.0056211, 0.0024377, -0.0033509, 0.0025918, -0.0082129, 0.0057886
9: -0.0121944, 0.0000053, -0.0125020, -0.0006485, -0.0115459, 0.0125073

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 97

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B2_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038388, upper bound: 0.0039178
time: 2.34 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B2_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039649, upper bound: 0.0039408
time: 1.96 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0057248, 0.0073631, 0.0058485, 0.0073954, -0.0016706, 0.0015145
1: -0.0016847, 0.0023437, -0.0005905, 0.0024062, -0.0040909, 0.0029342
2: -0.0019763, 0.0246924, -0.0024805, 0.0216860, -0.0236622, 0.0271729
3: -0.0046221, -0.0023361, -0.0044494, -0.0022910, -0.0023311, 0.0021133
4: -0.0010894, 0.0100020, -0.0002514, 0.0102205, -0.0113100, 0.0102534
5: -0.0020128, 0.0003433, -0.0020454, -0.0004817, -0.0015311, 0.0023887
6: 0.9890757, 0.9939448, 0.9911360, 0.9940045, -0.0049289, 0.0028088
7: -0.0157087, 0.0047226, -0.0138382, 0.0051181, -0.0206939, 0.0184154
8: -0.0090494, 0.0024679, -0.0033509, 0.0025918, -0.0116412, 0.0058188
9: -0.0122547, 0.0009926, -0.0125020, -0.0006485, -0.0116062, 0.0134946

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B2_A2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036247, upper bound: 0.0038493
time: 2.25 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B2_A2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038538, upper bound: 0.0039046
time: 4.26 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0057829, 0.0074076, 0.0058188, 0.0073584, -0.0015755, 0.0015888
1: -0.0011709, 0.0024300, -0.0008535, 0.0023346, -0.0035055, 0.0032835
2: -0.0026719, 0.0232807, -0.0019027, 0.0224088, -0.0250807, 0.0251834
3: -0.0045410, -0.0022739, -0.0044909, -0.0023426, -0.0021984, 0.0022170
4: -0.0006959, 0.0103035, -0.0004529, 0.0099702, -0.0106661, 0.0107563
5: -0.0020578, -0.0000441, -0.0020081, -0.0002833, -0.0017745, 0.0019640
6: 0.9900432, 0.9940273, 0.9906408, 0.9939361, -0.0038928, 0.0033865
7: -0.0148304, 0.0052682, -0.0142879, 0.0046649, -0.0194952, 0.0192334
8: -0.0063736, 0.0026388, -0.0047210, 0.0024498, -0.0088234, 0.0073598
9: -0.0125959, 0.0002220, -0.0122186, -0.0002539, -0.0123420, 0.0124406

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 97

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039404, upper bound: 0.0040335
time: 2.66 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039394, upper bound: 0.0040008
time: 2.10 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0057829, 0.0074076, 0.0058198, 0.0074049, -0.0016220, 0.0015878
1: -0.0011709, 0.0024300, -0.0008443, 0.0024247, -0.0035956, 0.0032743
2: -0.0026719, 0.0232807, -0.0026296, 0.0223834, -0.0250553, 0.0259103
3: -0.0045410, -0.0022739, -0.0044894, -0.0022777, -0.0022633, 0.0022155
4: -0.0006959, 0.0103035, -0.0004458, 0.0102851, -0.0109811, 0.0107493
5: -0.0020578, -0.0000441, -0.0020551, -0.0002903, -0.0017675, 0.0020110
6: 0.9900432, 0.9940273, 0.9906581, 0.9940223, -0.0039790, 0.0033692
7: -0.0148304, 0.0052682, -0.0142721, 0.0052350, -0.0200654, 0.0195403
8: -0.0063736, 0.0026388, -0.0046730, 0.0026284, -0.0090020, 0.0073118
9: -0.0125959, 0.0002220, -0.0125752, -0.0002678, -0.0123281, 0.0127971

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039404, upper bound: 0.0040444
time: 2.53 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039394, upper bound: 0.0040302
time: 2.47 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0057081, 0.0074132, 0.0058188, 0.0073584, -0.0016503, 0.0015944
1: -0.0018318, 0.0024409, -0.0008535, 0.0023346, -0.0041664, 0.0032944
2: -0.0027599, 0.0250967, -0.0019027, 0.0224088, -0.0251687, 0.0269995
3: -0.0046453, -0.0022661, -0.0044909, -0.0023426, -0.0023027, 0.0022248
4: -0.0012022, 0.0103416, -0.0004529, 0.0099702, -0.0111723, 0.0107945
5: -0.0020635, 0.0004542, -0.0020081, -0.0002833, -0.0017802, 0.0024623
6: 0.9887987, 0.9940377, 0.9906408, 0.9939361, -0.0051374, 0.0033970
7: -0.0159603, 0.0053372, -0.0142879, 0.0046649, -0.0206251, 0.0193341
8: -0.0098158, 0.0026605, -0.0047210, 0.0024498, -0.0122657, 0.0073815
9: -0.0126390, 0.0012133, -0.0122186, -0.0002539, -0.0123851, 0.0134320

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 97

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038282, upper bound: 0.0039874
time: 2.44 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038284, upper bound: 0.0039603
time: 2.34 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0057081, 0.0074132, 0.0058198, 0.0074049, -0.0016968, 0.0015934
1: -0.0018318, 0.0024409, -0.0008443, 0.0024247, -0.0042566, 0.0032852
2: -0.0027599, 0.0250967, -0.0026296, 0.0223834, -0.0251433, 0.0277263
3: -0.0046453, -0.0022661, -0.0044894, -0.0022777, -0.0023676, 0.0022234
4: -0.0012022, 0.0103416, -0.0004458, 0.0102851, -0.0114873, 0.0107874
5: -0.0020635, 0.0004542, -0.0020551, -0.0002903, -0.0017732, 0.0025093
6: 0.9887987, 0.9940377, 0.9906581, 0.9940223, -0.0052236, 0.0033796
7: -0.0159603, 0.0053372, -0.0142721, 0.0052350, -0.0211953, 0.0196094
8: -0.0098158, 0.0026605, -0.0046730, 0.0026284, -0.0124443, 0.0073335
9: -0.0126390, 0.0012133, -0.0125752, -0.0002678, -0.0123713, 0.0137885

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038517, upper bound: 0.0039694
time: 2.67 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038284, upper bound: 0.0039797
time: 2.55 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0057946, 0.0074069, 0.0058220, 0.0073757, -0.0015810, 0.0015850
1: -0.0010671, 0.0024286, -0.0008255, 0.0023681, -0.0034352, 0.0032542
2: -0.0026611, 0.0229955, -0.0021731, 0.0223319, -0.0249930, 0.0251686
3: -0.0045246, -0.0022749, -0.0044865, -0.0023185, -0.0022061, 0.0022116
4: -0.0006164, 0.0102988, -0.0004314, 0.0100873, -0.0107037, 0.0107302
5: -0.0020571, -0.0001223, -0.0020256, -0.0003044, -0.0017527, 0.0019032
6: 0.9902386, 0.9940260, 0.9906934, 0.9939681, -0.0037295, 0.0033326
7: -0.0146530, 0.0052598, -0.0142401, 0.0048769, -0.0195299, 0.0193608
8: -0.0058331, 0.0026362, -0.0045752, 0.0025162, -0.0083494, 0.0072114
9: -0.0125906, 0.0000663, -0.0123512, -0.0002959, -0.0122947, 0.0124176

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 213

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B2_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039398, upper bound: 0.0040166
time: 2.64 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B2_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039388, upper bound: 0.0039906
time: 2.73 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0057946, 0.0074069, 0.0058263, 0.0074209, -0.0016263, 0.0015807
1: -0.0010671, 0.0024286, -0.0007875, 0.0024557, -0.0035228, 0.0032162
2: -0.0026611, 0.0229955, -0.0028798, 0.0222275, -0.0248886, 0.0258753
3: -0.0045246, -0.0022749, -0.0044805, -0.0022554, -0.0022693, 0.0022056
4: -0.0006164, 0.0102988, -0.0004023, 0.0103935, -0.0110100, 0.0107011
5: -0.0020571, -0.0001223, -0.0020713, -0.0003331, -0.0017240, 0.0019489
6: 0.9902386, 0.9940260, 0.9907650, 0.9940519, -0.0038133, 0.0032610
7: -0.0146530, 0.0052598, -0.0141751, 0.0054313, -0.0200842, 0.0194348
8: -0.0058331, 0.0026362, -0.0043773, 0.0026899, -0.0085230, 0.0070135
9: -0.0125906, 0.0000663, -0.0126979, -0.0003529, -0.0122377, 0.0127642

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B2_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039398, upper bound: 0.0040311
time: 2.49 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B2_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039389, upper bound: 0.0040187
time: 2.71 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0057200, 0.0074125, 0.0058220, 0.0073757, -0.0016557, 0.0015906
1: -0.0017266, 0.0024395, -0.0008255, 0.0023681, -0.0040947, 0.0032650
2: -0.0027487, 0.0248075, -0.0021731, 0.0223319, -0.0250805, 0.0269806
3: -0.0046287, -0.0022671, -0.0044865, -0.0023185, -0.0023102, 0.0022194
4: -0.0011215, 0.0103367, -0.0004314, 0.0100873, -0.0112089, 0.0107682
5: -0.0020628, 0.0003749, -0.0020256, -0.0003044, -0.0017584, 0.0024004
6: 0.9889967, 0.9940363, 0.9906934, 0.9939681, -0.0049713, 0.0033429
7: -0.0157804, 0.0053284, -0.0142401, 0.0048769, -0.0206573, 0.0194608
8: -0.0092677, 0.0026577, -0.0045752, 0.0025162, -0.0117840, 0.0072329
9: -0.0126336, 0.0010555, -0.0123512, -0.0002959, -0.0123376, 0.0134067

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 97

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B2_A2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038266, upper bound: 0.0039733
time: 2.55 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B2_A2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038272, upper bound: 0.0039512
time: 2.68 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0057200, 0.0074125, 0.0058263, 0.0074209, -0.0017009, 0.0015863
1: -0.0017266, 0.0024395, -0.0007875, 0.0024557, -0.0041823, 0.0032270
2: -0.0027487, 0.0248075, -0.0028798, 0.0222275, -0.0249761, 0.0276873
3: -0.0046287, -0.0022671, -0.0044805, -0.0022554, -0.0023734, 0.0022134
4: -0.0011215, 0.0103367, -0.0004023, 0.0103935, -0.0115151, 0.0107391
5: -0.0020628, 0.0003749, -0.0020713, -0.0003331, -0.0017297, 0.0024462
6: 0.9889967, 0.9940363, 0.9907650, 0.9940519, -0.0050552, 0.0032713
7: -0.0157804, 0.0053284, -0.0141751, 0.0054313, -0.0212116, 0.0195035
8: -0.0092677, 0.0026577, -0.0043773, 0.0026899, -0.0119576, 0.0070350
9: -0.0126336, 0.0010555, -0.0126979, -0.0003529, -0.0122806, 0.0137533

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B2_A2_B2_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038266, upper bound: 0.0039839
time: 2.46 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B2_A2_B2_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038272, upper bound: 0.0039707
time: 2.62 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_B1_B1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0057918, 0.0073993, 0.0057624, 0.0073439, -0.0015520, 0.0016369
1: -0.0010917, 0.0024139, -0.0013516, 0.0023065, -0.0033982, 0.0037655
2: -0.0025420, 0.0230632, -0.0016763, 0.0237773, -0.0263193, 0.0247396
3: -0.0045285, -0.0022855, -0.0045695, -0.0023629, -0.0021657, 0.0022840
4: -0.0006353, 0.0102472, -0.0008344, 0.0098721, -0.0105074, 0.0110815
5: -0.0020494, -0.0001037, -0.0019934, 0.0000922, -0.0021416, 0.0018897
6: 0.9901922, 0.9940119, 0.9897028, 0.9939092, -0.0037171, 0.0043091
7: -0.0146951, 0.0051663, -0.0151394, 0.0044873, -0.0191824, 0.0199634
8: -0.0059615, 0.0026069, -0.0073150, 0.0023942, -0.0083556, 0.0099219
9: -0.0125322, 0.0001033, -0.0121076, 0.0004931, -0.0130253, 0.0122109

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_B1_B1_B1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B1_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035941, upper bound: 0.0038192
time: 1.98 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B1_B1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0035941, upper bound: 0.0039104
time: 2.22 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_B1_B1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0057171, 0.0074057, 0.0057624, 0.0073439, -0.0016268, 0.0016433
1: -0.0017522, 0.0024262, -0.0013516, 0.0023065, -0.0040587, 0.0037779
2: -0.0026418, 0.0248778, -0.0016763, 0.0237773, -0.0264192, 0.0265542
3: -0.0046328, -0.0022766, -0.0045695, -0.0023629, -0.0022699, 0.0022929
4: -0.0011411, 0.0102904, -0.0008344, 0.0098721, -0.0110132, 0.0111248
5: -0.0020559, 0.0003942, -0.0019934, 0.0000922, -0.0021481, 0.0023876
6: 0.9889486, 0.9940237, 0.9897028, 0.9939092, -0.0049607, 0.0043209
7: -0.0158241, 0.0052446, -0.0151394, 0.0044873, -0.0203114, 0.0196921
8: -0.0094009, 0.0026314, -0.0073150, 0.0023942, -0.0117951, 0.0099465
9: -0.0125812, 0.0010938, -0.0121076, 0.0004931, -0.0130743, 0.0132014

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_B1_B1_B1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B1_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035941, upper bound: 0.0038310
time: 2.18 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B1_B1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0035941, upper bound: 0.0039270
time: 2.30 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_B1_B1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0057193, 0.0074041, 0.0057350, 0.0073449, -0.0016257, 0.0016691
1: -0.0017333, 0.0024233, -0.0015941, 0.0023085, -0.0040419, 0.0040174
2: -0.0026179, 0.0248261, -0.0016925, 0.0244436, -0.0270614, 0.0265186
3: -0.0046298, -0.0022788, -0.0046078, -0.0023614, -0.0022684, 0.0023291
4: -0.0011267, 0.0102801, -0.0010201, 0.0098791, -0.0110058, 0.0113001
5: -0.0020543, 0.0003800, -0.0019945, 0.0002750, -0.0023293, 0.0023744
6: 0.9889841, 0.9940208, 0.9892462, 0.9939111, -0.0049270, 0.0047746
7: -0.0157919, 0.0052258, -0.0155539, 0.0044999, -0.0202919, 0.0200692
8: -0.0093029, 0.0026256, -0.0085778, 0.0023981, -0.0117010, 0.0112034
9: -0.0125694, 0.0010656, -0.0121155, 0.0008568, -0.0134262, 0.0131811

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_B1_B1_B1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B1_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035804, upper bound: 0.0037351
time: 2.31 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B1_B1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B1_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038084, upper bound: 0.0038933
time: 2.87 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_B1_B2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0058036, 0.0073986, 0.0057675, 0.0073607, -0.0015571, 0.0016311
1: -0.0009877, 0.0024125, -0.0013067, 0.0023391, -0.0033268, 0.0037192
2: -0.0025310, 0.0227774, -0.0019393, 0.0236539, -0.0261849, 0.0247168
3: -0.0045121, -0.0022865, -0.0045624, -0.0023394, -0.0021727, 0.0022759
4: -0.0005557, 0.0102424, -0.0008000, 0.0099860, -0.0105417, 0.0110424
5: -0.0020487, -0.0001822, -0.0020104, 0.0000583, -0.0021070, 0.0018283
6: 0.9903881, 0.9940106, 0.9897873, 0.9939403, -0.0035522, 0.0042232
7: -0.0145173, 0.0051577, -0.0150626, 0.0046936, -0.0192109, 0.0200580
8: -0.0054198, 0.0026042, -0.0070811, 0.0024588, -0.0078786, 0.0096853
9: -0.0125268, -0.0000527, -0.0122366, 0.0004257, -0.0129525, 0.0121839

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_B1_B1_B2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B1_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035840, upper bound: 0.0037890
time: 2.64 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B1_B2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B1_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038065, upper bound: 0.0038969
time: 2.28 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0057291, 0.0074050, 0.0057675, 0.0073607, -0.0016317, 0.0016375
1: -0.0016466, 0.0024250, -0.0013067, 0.0023391, -0.0039858, 0.0037317
2: -0.0026317, 0.0245879, -0.0019393, 0.0236539, -0.0262856, 0.0265273
3: -0.0046161, -0.0022775, -0.0045624, -0.0023394, -0.0022767, 0.0022849
4: -0.0010603, 0.0102861, -0.0008000, 0.0099860, -0.0110463, 0.0110860
5: -0.0020552, 0.0003146, -0.0020104, 0.0000583, -0.0021136, 0.0023251
6: 0.9891473, 0.9940225, 0.9897873, 0.9939403, -0.0047930, 0.0042352
7: -0.0156437, 0.0052367, -0.0150626, 0.0046936, -0.0203373, 0.0197912
8: -0.0088514, 0.0026290, -0.0070811, 0.0024588, -0.0113102, 0.0097101
9: -0.0125762, 0.0009356, -0.0122366, 0.0004257, -0.0130019, 0.0131722

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_B1_B1_B2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B1_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035840, upper bound: 0.0037990
time: 2.65 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B1_B2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038065, upper bound: 0.0039114
time: 2.66 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0057741, 0.0074256, 0.0057668, 0.0073872, -0.0016131, 0.0016588
1: -0.0012486, 0.0024648, -0.0013130, 0.0023903, -0.0036389, 0.0037778
2: -0.0029531, 0.0234942, -0.0023522, 0.0236712, -0.0266243, 0.0258464
3: -0.0045533, -0.0022488, -0.0045634, -0.0023025, -0.0022508, 0.0023146
4: -0.0007554, 0.0104253, -0.0008048, 0.0101649, -0.0109204, 0.0112301
5: -0.0020760, 0.0000145, -0.0020371, 0.0000630, -0.0021391, 0.0020516
6: 0.9898969, 0.9940606, 0.9897755, 0.9939893, -0.0040925, 0.0042850
7: -0.0149632, 0.0054888, -0.0150733, 0.0050174, -0.0199806, 0.0205621
8: -0.0067783, 0.0027079, -0.0071138, 0.0025603, -0.0093386, 0.0098217
9: -0.0127338, 0.0003385, -0.0124391, 0.0004352, -0.0131690, 0.0127776

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038615, upper bound: 0.0039326
time: 2.58 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038615, upper bound: 0.0039485
time: 2.27 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0057760, 0.0074245, 0.0057367, 0.0073911, -0.0016151, 0.0016878
1: -0.0012319, 0.0024626, -0.0015794, 0.0023980, -0.0036300, 0.0040419
2: -0.0029351, 0.0234484, -0.0024145, 0.0244030, -0.0273381, 0.0258629
3: -0.0045506, -0.0022504, -0.0046055, -0.0022969, -0.0022537, 0.0023551
4: -0.0007427, 0.0104175, -0.0010088, 0.0101919, -0.0109346, 0.0114263
5: -0.0020749, 0.0000019, -0.0020412, 0.0002639, -0.0023387, 0.0020431
6: 0.9899282, 0.9940585, 0.9892741, 0.9939967, -0.0040686, 0.0047844
7: -0.0149348, 0.0054747, -0.0155287, 0.0050663, -0.0200010, 0.0210034
8: -0.0066916, 0.0027035, -0.0085009, 0.0025756, -0.0092672, 0.0112044
9: -0.0127250, 0.0003136, -0.0124696, 0.0008346, -0.0135597, 0.0127832

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038648, upper bound: 0.0039085
time: 3.22 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038648, upper bound: 0.0039257
time: 2.91 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0057859, 0.0074249, 0.0057750, 0.0074030, -0.0016171, 0.0016500
1: -0.0011438, 0.0024635, -0.0012409, 0.0024211, -0.0035649, 0.0037044
2: -0.0029425, 0.0232063, -0.0026003, 0.0234730, -0.0264155, 0.0258066
3: -0.0045367, -0.0022498, -0.0045520, -0.0022803, -0.0022564, 0.0023023
4: -0.0006752, 0.0104207, -0.0007495, 0.0102725, -0.0109476, 0.0111703
5: -0.0020753, -0.0000645, -0.0020532, 0.0000087, -0.0020840, 0.0019887
6: 0.9900941, 0.9940594, 0.9899114, 0.9940187, -0.0039247, 0.0041479
7: -0.0147841, 0.0054804, -0.0149500, 0.0052121, -0.0199962, 0.0204305
8: -0.0062327, 0.0027053, -0.0067382, 0.0026212, -0.0088539, 0.0094435
9: -0.0127286, 0.0001814, -0.0125608, 0.0003270, -0.0130556, 0.0127422

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038611, upper bound: 0.0039208
time: 2.42 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038611, upper bound: 0.0039344
time: 2.34 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0057877, 0.0074238, 0.0057461, 0.0074053, -0.0016175, 0.0016777
1: -0.0011280, 0.0024613, -0.0014964, 0.0024255, -0.0035534, 0.0039577
2: -0.0029245, 0.0231627, -0.0026355, 0.0241751, -0.0270995, 0.0257983
3: -0.0045342, -0.0022514, -0.0045924, -0.0022772, -0.0022570, 0.0023410
4: -0.0006631, 0.0104129, -0.0009452, 0.0102877, -0.0109508, 0.0113582
5: -0.0020742, -0.0000765, -0.0020555, 0.0002013, -0.0022755, 0.0019790
6: 0.9901240, 0.9940572, 0.9894301, 0.9940229, -0.0038989, 0.0046271
7: -0.0147570, 0.0054664, -0.0153868, 0.0052397, -0.0199967, 0.0208532
8: -0.0061501, 0.0027009, -0.0080689, 0.0026299, -0.0087800, 0.0107698
9: -0.0127198, 0.0001576, -0.0125781, 0.0007102, -0.0134300, 0.0127357

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0038644, upper bound: 0.0039012
time: 2.74 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038644, upper bound: 0.0039172
time: 2.68 seconds

## Summary of splitting at layer (split count: 9)
- Time for NS candidates: 7.49 seconds
NS_A1_B1_A1_B1_B1_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0038411, upper bound: 0.0039345
NS_A1_B1_A1_B1_B1_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0039651, upper bound: 0.0039537
NS_A1_B1_A1_B1_B1_A1_B1_A1_B2_B1, status: Status.VERIFIED, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0038408, upper bound: 0.0038960
NS_A1_B1_A1_B1_B1_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0039758, upper bound: 0.0039314
NS_A1_B1_A1_B1_B1_A1_B1_A2_B1_B1, status: Status.VERIFIED, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0036326, upper bound: 0.0038656
NS_A1_B1_A1_B1_B1_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0038548, upper bound: 0.0039157
NS_A1_B1_A1_B1_B1_A1_B1_A2_B2_B1, status: Status.VERIFIED, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0036299, upper bound: 0.0038324
NS_A1_B1_A1_B1_B1_A1_B1_A2_B2_B2, status: Status.VERIFIED, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0038663, upper bound: 0.0038993
NS_A1_B1_A1_B1_B1_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0038388, upper bound: 0.0039178
NS_A1_B1_A1_B1_B1_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0039649, upper bound: 0.0039408
NS_A1_B1_A1_B1_B1_A1_B2_A2_B1_B1, status: Status.VERIFIED, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0036247, upper bound: 0.0038493
NS_A1_B1_A1_B1_B1_A1_B2_A2_B1_B2, status: Status.VERIFIED, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0038538, upper bound: 0.0039046
NS_A1_B1_A1_B1_B1_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0039404, upper bound: 0.0040335
NS_A1_B1_A1_B1_B1_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0039394, upper bound: 0.0040008
NS_A1_B1_A1_B1_B1_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0039404, upper bound: 0.0040444
NS_A1_B1_A1_B1_B1_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0039394, upper bound: 0.0040302
NS_A1_B1_A1_B1_B1_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0038282, upper bound: 0.0039874
NS_A1_B1_A1_B1_B1_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0038284, upper bound: 0.0039603
NS_A1_B1_A1_B1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0038517, upper bound: 0.0039694
NS_A1_B1_A1_B1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0038284, upper bound: 0.0039797
NS_A1_B1_A1_B1_B1_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0039398, upper bound: 0.0040166
NS_A1_B1_A1_B1_B1_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0039388, upper bound: 0.0039906
NS_A1_B1_A1_B1_B1_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0039398, upper bound: 0.0040311
NS_A1_B1_A1_B1_B1_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0039389, upper bound: 0.0040187
NS_A1_B1_A1_B1_B1_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0038266, upper bound: 0.0039733
NS_A1_B1_A1_B1_B1_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0038272, upper bound: 0.0039512
NS_A1_B1_A1_B1_B1_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0038266, upper bound: 0.0039839
NS_A1_B1_A1_B1_B1_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0038272, upper bound: 0.0039707
NS_A1_B1_A1_B2_B1_B1_B1_B1_A1_B1, status: Status.VERIFIED, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0035941, upper bound: 0.0038192
NS_A1_B1_A1_B2_B1_B1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0035941, upper bound: 0.0039104
NS_A1_B1_A1_B2_B1_B1_B1_B1_A2_B1, status: Status.VERIFIED, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0035941, upper bound: 0.0038310
NS_A1_B1_A1_B2_B1_B1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0035941, upper bound: 0.0039270
NS_A1_B1_A1_B2_B1_B1_B1_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0035804, upper bound: 0.0037351
NS_A1_B1_A1_B2_B1_B1_B1_B2_A2_B2, status: Status.VERIFIED, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0038084, upper bound: 0.0038933
NS_A1_B1_A1_B2_B1_B1_B2_B1_A1_B1, status: Status.VERIFIED, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0035840, upper bound: 0.0037890
NS_A1_B1_A1_B2_B1_B1_B2_B1_A1_B2, status: Status.VERIFIED, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0038065, upper bound: 0.0038969
NS_A1_B1_A1_B2_B1_B1_B2_B1_A2_B1, status: Status.VERIFIED, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0035840, upper bound: 0.0037990
NS_A1_B1_A1_B2_B1_B1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0038065, upper bound: 0.0039114
NS_A1_B1_A1_B2_B1_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0038615, upper bound: 0.0039326
NS_A1_B1_A1_B2_B1_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0038615, upper bound: 0.0039485
NS_A1_B1_A1_B2_B1_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0038648, upper bound: 0.0039085
NS_A1_B1_A1_B2_B1_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0038648, upper bound: 0.0039257
NS_A1_B1_A1_B2_B1_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0038611, upper bound: 0.0039208
NS_A1_B1_A1_B2_B1_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0038611, upper bound: 0.0039344
NS_A1_B1_A1_B2_B1_B2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0038644, upper bound: 0.0039012
NS_A1_B1_A1_B2_B1_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 7.49
Output dim: 6, lower bound: -0.0038644, upper bound: 0.0039172

## BFS NS instance: NS_A1_B1_A1_B1_B1_A1_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: 0.0057992, 0.0073404, 0.0058573, 0.0073237, -0.0012588, 0.0014831
1: -0.0010266, 0.0022998, -0.0005728, 0.0022674, -0.0032940, 0.0023505
2: -0.0016220, 0.0228844, -0.0013605, 0.0215483, -0.0189588, 0.0242449
3: -0.0045182, -0.0023677, -0.0044372, -0.0023911, -0.0017565, 0.0020695
4: -0.0005855, 0.0098485, -0.0001921, 0.0097352, -0.0085221, 0.0100406
5: -0.0019899, -0.0001528, -0.0019730, -0.0004911, -0.0012264, 0.0018202
6: 0.9903148, 0.9939027, 0.9911538, 0.9938717, -0.0035570, 0.0022493
7: -0.0145838, 0.0044447, -0.0137305, 0.0042395, -0.0184070, 0.0148717
8: -0.0056225, 0.0023808, -0.0033133, 0.0023165, -0.0079391, 0.0046592
9: -0.0120810, 0.0000057, -0.0119527, -0.0007162, -0.0092991, 0.0119583

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 97

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B1_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038411, upper bound: 0.0039345
time: 2.43 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B1_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038411, upper bound: 0.0039345
time: 2.17 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A1_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: 0.0057891, 0.0073558, 0.0058494, 0.0073712, -0.0012958, 0.0015064
1: -0.0011160, 0.0023295, -0.0005882, 0.0023595, -0.0034755, 0.0023891
2: -0.0018619, 0.0231298, -0.0021036, 0.0216721, -0.0192700, 0.0252333
3: -0.0045323, -0.0023463, -0.0044482, -0.0023247, -0.0018081, 0.0021019
4: -0.0006539, 0.0099525, -0.0002458, 0.0100572, -0.0087725, 0.0101982
5: -0.0020054, -0.0000855, -0.0020211, -0.0004831, -0.0012465, 0.0019356
6: 0.9901466, 0.9939312, 0.9911391, 0.9939598, -0.0038132, 0.0022863
7: -0.0147365, 0.0046328, -0.0138277, 0.0048224, -0.0189597, 0.0151157
8: -0.0060876, 0.0024398, -0.0033438, 0.0024992, -0.0085868, 0.0047357
9: -0.0121986, 0.0001396, -0.0123171, -0.0006554, -0.0094517, 0.0124568

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 97

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B1_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039651, upper bound: 0.0039537
time: 2.54 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B1_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039651, upper bound: 0.0039537
time: 2.06 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A1_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: 0.0057904, 0.0073543, 0.0058157, 0.0073748, -0.0015844, 0.0015386
1: -0.0011043, 0.0023266, -0.0008807, 0.0023665, -0.0034708, 0.0032073
2: -0.0018386, 0.0230978, -0.0021603, 0.0224833, -0.0243219, 0.0252581
3: -0.0045305, -0.0023484, -0.0044952, -0.0023196, -0.0022109, 0.0021468
4: -0.0006450, 0.0099424, -0.0004737, 0.0100818, -0.0107267, 0.0104160
5: -0.0020039, -0.0000943, -0.0020247, -0.0002629, -0.0017410, 0.0019305
6: 0.9901685, 0.9939284, 0.9905896, 0.9939666, -0.0037981, 0.0033388
7: -0.0147166, 0.0046145, -0.0143343, 0.0048669, -0.0190468, 0.0189488
8: -0.0060270, 0.0024340, -0.0048623, 0.0025131, -0.0085401, 0.0072964
9: -0.0121872, 0.0001222, -0.0123450, -0.0002132, -0.0119739, 0.0124671

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 97

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B1_A1_B2_B2_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039225, upper bound: 0.0039312
time: 2.81 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B1_A1_B2_B2_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039225, upper bound: 0.0039313
time: 2.43 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A1_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: 0.0057145, 0.0073637, 0.0058494, 0.0073712, -0.0013732, 0.0015143
1: -0.0017754, 0.0023448, -0.0005882, 0.0023595, -0.0041348, 0.0024105
2: -0.0019854, 0.0249415, -0.0021036, 0.0216721, -0.0194428, 0.0270451
3: -0.0046364, -0.0023352, -0.0044482, -0.0023247, -0.0019161, 0.0021130
4: -0.0011589, 0.0100060, -0.0002458, 0.0100572, -0.0092967, 0.0102517
5: -0.0020134, 0.0004116, -0.0020211, -0.0004831, -0.0012577, 0.0024327
6: 0.9889048, 0.9939458, 0.9911391, 0.9939598, -0.0050550, 0.0023068
7: -0.0158637, 0.0047297, -0.0138277, 0.0048224, -0.0200974, 0.0152513
8: -0.0095217, 0.0024701, -0.0033438, 0.0024992, -0.0120208, 0.0047781
9: -0.0122592, 0.0011286, -0.0123171, -0.0006554, -0.0095365, 0.0134457

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 97

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B1_A2_B1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038548, upper bound: 0.0039157
time: 2.52 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B1_A2_B1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038548, upper bound: 0.0039157
time: 2.33 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A1_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: 0.0058093, 0.0073399, 0.0058637, 0.0073405, -0.0012765, 0.0014762
1: -0.0009370, 0.0022987, -0.0005605, 0.0023000, -0.0032370, 0.0023566
2: -0.0016134, 0.0226381, -0.0016240, 0.0214486, -0.0190083, 0.0242621
3: -0.0045041, -0.0023685, -0.0044283, -0.0023675, -0.0017812, 0.0020598
4: -0.0005168, 0.0098448, -0.0001489, 0.0098494, -0.0086419, 0.0099937
5: -0.0019894, -0.0002204, -0.0019900, -0.0004975, -0.0012296, 0.0017696
6: 0.9904836, 0.9939017, 0.9911655, 0.9939030, -0.0034194, 0.0022552
7: -0.0144306, 0.0044379, -0.0136524, 0.0044462, -0.0187121, 0.0149105
8: -0.0051557, 0.0023787, -0.0032889, 0.0023813, -0.0075370, 0.0046713
9: -0.0120767, -0.0001287, -0.0120819, -0.0007650, -0.0093234, 0.0119532

Time for backsubstitution: 1.75 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 5.76 + 594.75 = 600.51 seconds
