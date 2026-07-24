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
Threshold: 173.89956106530002


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=61, inp2_unstable=61, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329)
1: (-79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183)
2: (-104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471)
3: (-110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219)
4: (-101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509)
5: (-90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867)
6: (-86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223)
7: (-95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773)
8: (-114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580)
9: (-86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.49 + 10.61 = 12.10 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -174.0736347, upper bound: 174.0736347

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9962271, upper bound: 174.0008380
time: 8.67 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9869953, upper bound: 173.9869953
time: 6.03 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 14.86 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 14.86
Output dim: 7, lower bound: -173.9962271, upper bound: 174.0008380
IS_A2, status: Status.UNKNOWN, split count: 1, time: 14.86
Output dim: 7, lower bound: -173.9869953, upper bound: 173.9869953

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -94.2460861, 74.8623047, -94.5060349, 75.0702057, -169.3162842, 169.3683319
1: -78.9821320, 66.4131470, -79.2014389, 66.5985794, -145.5807037, 145.6145935
2: -104.0140457, 67.8887329, -104.3030472, 68.0764999, -172.0905304, 172.1917725
3: -110.3552094, 58.0360298, -110.6649246, 58.1981163, -168.5533295, 168.7009277
4: -100.8139877, 77.5684891, -101.0963440, 77.7846146, -178.5986023, 178.6648254
5: -90.4393463, 70.3454666, -90.6905060, 70.5433807, -160.9827271, 161.0359802
6: -86.6970139, 83.5235748, -86.9384842, 83.7556839, -170.4526825, 170.4620209
7: -94.8713150, 79.9647064, -95.1351624, 80.1866226, -175.0579376, 175.0998688
8: -114.1304474, 77.2884140, -114.4460297, 77.5040588, -191.6344910, 191.7344360
9: -86.4736252, 84.5179214, -86.7146835, 84.7555695, -171.2291870, 171.2326050

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=61, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9760411, upper bound: 173.9775508
time: 8.68 seconds

## Relational analysis of IS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9939648, upper bound: 173.9980546
time: 8.12 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9867920, upper bound: 173.9918835
time: 8.41 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -97.1398468, 77.0113602, -93.4849319, 74.2527542, -171.3925629, 170.4962921
1: -81.2423706, 68.2438812, -78.3389893, 65.8683929, -147.1107635, 146.5828705
2: -106.9815216, 69.6908875, -103.1663513, 67.3365936, -174.3181152, 172.8572235
3: -113.4796600, 59.5425911, -109.4469452, 57.5597343, -171.0393677, 168.9895325
4: -103.7134476, 79.6397400, -99.9858551, 76.9331055, -180.6465302, 179.6255798
5: -93.0929413, 72.0213623, -89.7031631, 69.7627335, -162.8556671, 161.7245178
6: -89.2468872, 85.9368439, -85.9888000, 82.8421173, -172.0890045, 171.9256287
7: -97.5071945, 82.1068726, -94.0961227, 79.3124466, -176.8196259, 176.2030029
8: -117.5732346, 79.3855209, -113.2059174, 76.6553879, -194.2286224, 192.5914307
9: -88.7992706, 86.7623596, -85.7651672, 83.8187866, -172.6180573, 172.5275116

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=61, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=199, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9785582, upper bound: 173.9776385
time: 6.85 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9762404, upper bound: 173.9762404
time: 5.75 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 34.83 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 34.83
Output dim: 7, lower bound: -173.9939648, upper bound: 173.9980546
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 34.83
Output dim: 7, lower bound: -173.9867920, upper bound: 173.9918835
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 34.83
Output dim: 7, lower bound: -173.9785582, upper bound: 173.9776385
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 34.83
Output dim: 7, lower bound: -173.9762404, upper bound: 173.9762404

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -93.1831741, 74.0235596, -91.0917587, 72.3756943, -165.5588684, 165.1153107
1: -78.0881729, 65.6629715, -76.3301773, 64.1884613, -142.2766113, 141.9931335
2: -102.8402481, 67.1276932, -100.5319138, 65.6316681, -168.4719086, 167.6596069
3: -109.1036530, 57.3801155, -106.6444931, 56.0903130, -165.1939697, 164.0246124
4: -99.6721725, 76.6927032, -97.4282455, 74.9719315, -174.6441040, 174.1209412
5: -89.4221573, 69.5592346, -87.4257812, 68.0162354, -157.4383850, 156.9850159
6: -85.7197876, 82.5819702, -83.7994003, 80.7310104, -166.4508057, 166.3813629
7: -93.8023300, 79.0722809, -91.6993408, 77.3193970, -171.1217041, 170.7716217
8: -112.8417816, 76.4119949, -110.3076019, 74.6895065, -187.5312805, 186.7196045
9: -85.5054398, 83.5662079, -83.6045609, 81.6996155, -167.2050476, 167.1707611

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=187, inp2_unstable=185, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9867920, upper bound: 173.9918835
time: 9.16 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9867920, upper bound: 173.9918835
time: 7.99 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -92.7955475, 73.7162628, -105.6991653, 83.9540863, -176.7496338, 179.4153748
1: -77.7604523, 65.3890305, -88.4945602, 74.4153595, -152.1758118, 153.8835754
2: -102.4129868, 66.8508759, -116.5678711, 75.9626389, -178.3755951, 183.4187469
3: -108.6444321, 57.1402359, -123.7401199, 64.9252014, -173.5696411, 180.8803558
4: -99.2515335, 76.3713913, -113.1579437, 86.9111328, -186.1626587, 189.5293274
5: -89.0489426, 69.2704544, -101.4842300, 78.8699799, -167.9188995, 170.7546844
6: -85.3612442, 82.2373886, -97.2234116, 93.5800781, -178.9413147, 179.4608002
7: -93.4111328, 78.7459106, -106.2618103, 89.4792633, -182.8903961, 185.0076752
8: -112.3742676, 76.0912933, -127.8211594, 86.6612167, -199.0354767, 203.9124451
9: -85.1495361, 83.2177429, -96.8313675, 94.7412720, -179.8908081, 180.0491028

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=186, inp2_unstable=189, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=256, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9867920, upper bound: 173.9918835
time: 7.78 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9867920, upper bound: 173.9918835
time: 8.50 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -96.0779190, 76.1727448, -90.0792007, 71.5648804, -167.6427917, 166.2519531
1: -80.3486328, 67.4941788, -75.4747543, 63.4642639, -143.8128815, 142.9689178
2: -105.8093796, 68.9302673, -99.4046555, 64.8977280, -170.7070923, 168.3349152
3: -112.2275467, 58.8868752, -105.4364014, 55.4570618, -167.6845856, 164.3232727
4: -102.5730591, 78.7647552, -96.3269577, 74.1274948, -176.7005310, 175.0917053
5: -92.0758133, 71.2353439, -86.4467239, 67.2417755, -159.3175812, 157.6820679
6: -88.2704697, 84.9959488, -82.8577499, 79.8249359, -168.0953979, 167.8536987
7: -96.4392624, 81.2143250, -90.6687317, 76.4521484, -172.8913879, 171.8830566
8: -116.2873611, 78.5121002, -109.0782318, 73.8483047, -190.1356506, 187.5903015
9: -87.8318253, 85.8111038, -82.6627808, 80.7705231, -168.6023254, 168.4738770

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=198, inp2_unstable=185, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9762404, upper bound: 173.9762404
time: 6.33 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9762404, upper bound: 173.9762404
time: 6.21 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -95.6868286, 75.8625793, -104.7123032, 83.1642075, -178.8510437, 180.5748901
1: -80.0181351, 67.2176895, -87.6610413, 73.7097702, -153.7279053, 154.8787231
2: -105.3788452, 68.6516953, -115.4694977, 75.2475204, -180.6263733, 184.1211853
3: -111.7636566, 58.6447220, -122.5632095, 64.3080902, -176.0717468, 181.2079315
4: -102.1490784, 78.4407272, -112.0848312, 86.0884171, -188.2374878, 190.5255585
5: -91.6992874, 70.9437561, -100.5304031, 78.1152420, -169.8145294, 171.4741516
6: -87.9084625, 84.6485291, -96.3057022, 92.6974335, -180.6058960, 180.9542236
7: -96.0447388, 80.8852386, -105.2575607, 88.6342392, -184.6789856, 186.1427917
8: -115.8163528, 78.1894073, -126.6229782, 85.8415298, -201.6578827, 204.8123779
9: -87.4729233, 85.4597778, -95.9136963, 93.8362885, -181.3092041, 181.3734131

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=198, inp2_unstable=189, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=256, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9762404, upper bound: 173.9762404
time: 5.32 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9762404, upper bound: 173.9762404
time: 6.10 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 25.46 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 25.46
Output dim: 7, lower bound: -173.9867920, upper bound: 173.9918835
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 25.46
Output dim: 7, lower bound: -173.9867920, upper bound: 173.9918835
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 25.46
Output dim: 7, lower bound: -173.9867920, upper bound: 173.9918835
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 25.46
Output dim: 7, lower bound: -173.9867920, upper bound: 173.9918835
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 25.46
Output dim: 7, lower bound: -173.9762404, upper bound: 173.9762404
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 25.46
Output dim: 7, lower bound: -173.9762404, upper bound: 173.9762404
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 25.46
Output dim: 7, lower bound: -173.9762404, upper bound: 173.9762404
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 25.46
Output dim: 7, lower bound: -173.9762404, upper bound: 173.9762404

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -90.8334122, 72.1690445, -91.0917587, 72.3756943, -163.2091064, 163.2607727
1: -76.1121597, 64.0041580, -76.3301773, 64.1884613, -140.3006134, 140.3343048
2: -100.2447052, 65.4450836, -100.5319138, 65.6316681, -165.8763428, 165.9769897
3: -106.3366089, 55.9291992, -106.6444931, 56.0903130, -162.4269257, 162.5737000
4: -97.1475983, 74.7571487, -97.4282455, 74.9719315, -172.1195374, 172.1853943
5: -87.1761780, 67.8195038, -87.4257812, 68.0162354, -155.1924133, 155.2452850
6: -83.5594559, 80.5003052, -83.7994003, 80.7310104, -164.2904205, 164.2996826
7: -91.4370575, 77.0987625, -91.6993408, 77.3193970, -168.7564545, 168.7980957
8: -109.9940720, 74.4753036, -110.3076019, 74.6895065, -184.6835632, 184.7828979
9: -83.3649826, 81.4634323, -83.6045609, 81.6996155, -165.0645905, 165.0679626

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=60, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=185, inp2_unstable=185, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9719896, upper bound: 173.9736990
time: 7.97 seconds

## Relational analysis of IS_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9921561, upper bound: 173.9962535
time: 7.70 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9939648, upper bound: 173.9980547
time: 9.69 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -105.4372940, 83.7445374, -91.0917587, 72.3756943, -177.8129578, 174.8362732
1: -88.2734985, 74.2285385, -76.3301773, 64.1884613, -152.4619598, 150.5586853
2: -116.2766953, 75.7733078, -100.5319138, 65.6316681, -181.9083557, 176.3052216
3: -123.4280319, 64.7618713, -106.6444931, 56.0903130, -179.5183411, 171.4063721
4: -112.8734436, 86.6933899, -97.4282455, 74.9719315, -187.8453674, 184.1216431
5: -101.2312012, 78.6704330, -87.4257812, 68.0162354, -169.2474213, 166.0962219
6: -96.9802322, 93.3461151, -83.7994003, 80.7310104, -177.7112274, 177.1455078
7: -105.9958954, 89.2556381, -91.6993408, 77.3193970, -183.3152771, 180.9549866
8: -127.5032578, 86.4439621, -110.3076019, 74.6895065, -202.1927490, 196.7515564
9: -96.5884933, 94.5018539, -83.6045609, 81.6996155, -178.2881165, 178.1064148

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=60, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=189, inp2_unstable=185, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9934280, upper bound: 173.9977270
time: 8.11 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9939648, upper bound: 173.9980546
time: 8.97 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -90.8334122, 72.1690445, -105.6991653, 83.9540863, -174.7875061, 177.8681641
1: -76.1121597, 64.0041580, -88.4945602, 74.4153595, -150.5275269, 152.4987183
2: -100.2447052, 65.4450836, -116.5678711, 75.9626389, -176.2072906, 182.0129547
3: -106.3366089, 55.9291992, -123.7401199, 64.9252014, -171.2618103, 179.6693115
4: -97.1475983, 74.7571487, -113.1579437, 86.9111328, -184.0587311, 187.9150848
5: -87.1761780, 67.8195038, -101.4842300, 78.8699799, -166.0461578, 169.3037415
6: -83.5594559, 80.5003052, -97.2234116, 93.5800781, -177.1394958, 177.7237091
7: -91.4370575, 77.0987625, -106.2618103, 89.4792633, -180.9163208, 183.3605652
8: -109.9940720, 74.4753036, -127.8211594, 86.6612167, -196.6552887, 202.2964630
9: -83.3649826, 81.4634323, -96.8313675, 94.7412720, -178.1062317, 178.2947845

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=60, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=185, inp2_unstable=189, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=256, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9624625, upper bound: 173.9636519
time: 8.69 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9800736, upper bound: 173.9855334
time: 8.63 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9867920, upper bound: 173.9918835
time: 7.35 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -105.4372940, 83.7445374, -105.6991653, 83.9540863, -189.3913574, 189.4436646
1: -88.2734985, 74.2285385, -88.4945602, 74.4153595, -162.6888580, 162.7230835
2: -116.2766953, 75.7733078, -116.5678711, 75.9626389, -192.2393036, 192.3411865
3: -123.4280319, 64.7618713, -123.7401199, 64.9252014, -188.3532410, 188.5019836
4: -112.8734436, 86.6933899, -113.1579437, 86.9111328, -199.7845764, 199.8513184
5: -101.2312012, 78.6704330, -101.4842300, 78.8699799, -180.1011658, 180.1546631
6: -96.9802322, 93.3461151, -97.2234116, 93.5800781, -190.5602875, 190.5695190
7: -105.9958954, 89.2556381, -106.2618103, 89.4792633, -195.4751587, 195.5174561
8: -127.5032578, 86.4439621, -127.8211594, 86.6612167, -214.1644592, 214.2651215
9: -96.5884933, 94.5018539, -96.8313675, 94.7412720, -191.3297729, 191.3332214

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=60, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=189, inp2_unstable=189, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9624625, upper bound: 173.9636519
time: 9.38 seconds

## Relational analysis of IS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9800736, upper bound: 173.9855333
time: 9.15 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9867920, upper bound: 173.9918835
time: 8.30 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -93.7403030, 74.3259430, -90.0792007, 71.5648804, -165.3051758, 164.4051514
1: -78.3804855, 65.8434143, -75.4747543, 63.4642639, -141.8447571, 141.3181458
2: -103.2282486, 67.2559814, -99.4046555, 64.8977280, -168.1259613, 166.6606140
3: -109.4676819, 57.4434204, -105.4364014, 55.4570618, -164.9247284, 162.8798218
4: -100.0625229, 76.8396683, -96.3269577, 74.1274948, -174.1900024, 173.1666260
5: -89.8383331, 69.5029449, -86.4467239, 67.2417755, -157.0800781, 155.9496765
6: -86.1222763, 82.9249039, -82.8577499, 79.8249359, -165.9471741, 165.7826538
7: -94.0857697, 79.2475433, -90.6687317, 76.4521484, -170.5378876, 169.9162750
8: -113.4624100, 76.5956497, -109.0782318, 73.8483047, -187.3107147, 185.6738892
9: -85.7015686, 83.7190170, -82.6627808, 80.7705231, -166.4720917, 166.3817749

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=60, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=196, inp2_unstable=185, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9754658, upper bound: 173.9745976
time: 7.23 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9785582, upper bound: 173.9776385
time: 6.65 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -108.2859344, 85.8609009, -90.0792007, 71.5648804, -179.8508148, 175.9400940
1: -90.4996490, 76.0357590, -75.4747543, 63.4642639, -153.9638977, 151.5104828
2: -119.2036591, 77.5540924, -99.4046555, 64.8977280, -184.1013641, 176.9587402
3: -126.5062027, 66.2461929, -105.4364014, 55.4570618, -181.9632263, 171.6825867
4: -115.7342758, 88.7373047, -96.3269577, 74.1274948, -189.8617706, 185.0642548
5: -103.8441162, 80.3245316, -86.4467239, 67.2417755, -171.0858765, 166.7712402
6: -99.4937592, 95.7221069, -82.8577499, 79.8249359, -179.3186646, 178.5798492
7: -108.5926208, 91.3672485, -90.6687317, 76.4521484, -185.0447235, 182.0359802
8: -130.9031067, 88.5163498, -109.0782318, 73.8483047, -204.7514038, 197.5945740
9: -98.8762436, 96.7144318, -82.6627808, 80.7705231, -179.6467590, 179.3771973

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=60, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=202, inp2_unstable=185, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9716812, upper bound: 173.9705325
time: 6.96 seconds

## Relational analysis of IS_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9785582, upper bound: 173.9776385
time: 7.26 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -93.7403030, 74.3259430, -104.7123032, 83.1642075, -176.9045105, 179.0382385
1: -78.3804855, 65.8434143, -87.6610413, 73.7097702, -152.0902557, 153.5044556
2: -103.2282486, 67.2559814, -115.4694977, 75.2475204, -178.4757690, 182.7254791
3: -109.4676819, 57.4434204, -122.5632095, 64.3080902, -173.7757721, 180.0066223
4: -100.0625229, 76.8396683, -112.0848312, 86.0884171, -186.1509247, 188.9244995
5: -89.8383331, 69.5029449, -100.5304031, 78.1152420, -167.9535522, 170.0333557
6: -86.1222763, 82.9249039, -96.3057022, 92.6974335, -178.8197021, 179.2306061
7: -94.0857697, 79.2475433, -105.2575607, 88.6342392, -182.7199860, 184.5050964
8: -113.4624100, 76.5956497, -126.6229782, 85.8415298, -199.3039398, 203.2186279
9: -85.7015686, 83.7190170, -95.9136963, 93.8362885, -179.5378571, 179.6326752

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=60, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=196, inp2_unstable=189, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=256, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 73

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 12.10 + 589.24 = 601.34 seconds
