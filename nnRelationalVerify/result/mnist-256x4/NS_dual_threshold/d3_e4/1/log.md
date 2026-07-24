## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 173.89956106530002


## IAR start

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
execution time: IAR + RelationalAnalysis = 0.89 + 10.36 = 11.25 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -174.0736347, upper bound: 174.0736347

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9962271, upper bound: 174.0008380
time: 8.99 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9869953, upper bound: 173.9869953
time: 5.87 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 14.94 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 14.94
Output dim: 7, lower bound: -173.9962271, upper bound: 174.0008380
NS_A2, status: Status.UNKNOWN, split count: 1, time: 14.94
Output dim: 7, lower bound: -173.9869953, upper bound: 173.9869953

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 185

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9760411, upper bound: 173.9775508
time: 8.63 seconds

## Relational analysis of NS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9939648, upper bound: 173.9980546
time: 8.14 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9867920, upper bound: 173.9918835
time: 8.52 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 185

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9785582, upper bound: 173.9776385
time: 6.55 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9762404, upper bound: 173.9762404
time: 5.58 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 22.47 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 22.47
Output dim: 7, lower bound: -173.9939648, upper bound: 173.9980546
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 22.47
Output dim: 7, lower bound: -173.9867920, upper bound: 173.9918835
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 22.47
Output dim: 7, lower bound: -173.9785582, upper bound: 173.9776385
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 22.47
Output dim: 7, lower bound: -173.9762404, upper bound: 173.9762404

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9867920, upper bound: 173.9918835
time: 8.95 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9867920, upper bound: 173.9918835
time: 7.96 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9867920, upper bound: 173.9918835
time: 7.69 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9867920, upper bound: 173.9918835
time: 8.41 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9762404, upper bound: 173.9762404
time: 6.14 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9762404, upper bound: 173.9762404
time: 6.09 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9762404, upper bound: 173.9762404
time: 5.10 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9762404, upper bound: 173.9762404
time: 5.91 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 14.12 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 14.12
Output dim: 7, lower bound: -173.9867920, upper bound: 173.9918835
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 14.12
Output dim: 7, lower bound: -173.9867920, upper bound: 173.9918835
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 14.12
Output dim: 7, lower bound: -173.9867920, upper bound: 173.9918835
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 14.12
Output dim: 7, lower bound: -173.9867920, upper bound: 173.9918835
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 14.12
Output dim: 7, lower bound: -173.9762404, upper bound: 173.9762404
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 14.12
Output dim: 7, lower bound: -173.9762404, upper bound: 173.9762404
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 14.12
Output dim: 7, lower bound: -173.9762404, upper bound: 173.9762404
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 14.12
Output dim: 7, lower bound: -173.9762404, upper bound: 173.9762404

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 185

### Candidate
type: A, layer: 1, pos: 185

### Candidate
type: B, layer: 1, pos: 250

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 251

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 251

### Candidate
type: B, layer: 1, pos: 197

### Candidate
type: A, layer: 1, pos: 197

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 96

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9921561, upper bound: 173.9962535
time: 7.51 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9939648, upper bound: 173.9980547
time: 9.46 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 197

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of NS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 185

### Candidate
type: B, layer: 1, pos: 185

### Candidate
type: A, layer: 1, pos: 251

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9934280, upper bound: 173.9977270
time: 7.91 seconds

## Relational analysis of NS_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9939648, upper bound: 173.9980546
time: 8.68 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 197

### Candidate
type: B, layer: 1, pos: 250

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 185

### Candidate
type: A, layer: 1, pos: 185

### Candidate
type: B, layer: 1, pos: 251

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9800736, upper bound: 173.9855334
time: 8.38 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9867920, upper bound: 173.9918835
time: 7.12 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 185

### Candidate
type: A, layer: 1, pos: 185

### Candidate
type: B, layer: 1, pos: 251

### Candidate
type: A, layer: 1, pos: 251

### Candidate
type: B, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 197

### Candidate
type: A, layer: 1, pos: 197

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9800736, upper bound: 173.9855333
time: 8.73 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9867920, upper bound: 173.9918835
time: 7.81 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 185

### Candidate
type: A, layer: 1, pos: 185

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9754658, upper bound: 173.9745976
time: 7.31 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9785582, upper bound: 173.9776385
time: 6.68 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of NS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 185

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 185

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of NS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of NS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9716812, upper bound: 173.9705325
time: 6.95 seconds

## Relational analysis of NS_A2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9785582, upper bound: 173.9776385
time: 7.26 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 197

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9693656, upper bound: 173.9699542
time: 6.97 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9762404, upper bound: 173.9762404
time: 6.15 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -108.2859344, 85.8609009, -104.7123032, 83.1642075, -191.4501343, 190.5732117
1: -90.4996490, 76.0357590, -87.6610413, 73.7097702, -164.2094116, 163.6967926
2: -119.2036591, 77.5540924, -115.4694977, 75.2475204, -194.4511719, 193.0235901
3: -126.5062027, 66.2461929, -122.5632095, 64.3080902, -190.8143005, 188.8094025
4: -115.7342758, 88.7373047, -112.0848312, 86.0884171, -201.8226929, 200.8221436
5: -103.8441162, 80.3245316, -100.5304031, 78.1152420, -181.9593506, 180.8549347
6: -99.4937592, 95.7221069, -96.3057022, 92.6974335, -192.1911926, 192.0278015
7: -108.5926208, 91.3672485, -105.2575607, 88.6342392, -197.2268372, 196.6247711
8: -130.9031067, 88.5163498, -126.6229782, 85.8415298, -216.7446289, 215.1393280
9: -98.8762436, 96.7144318, -95.9136963, 93.8362885, -192.7125244, 192.6280975

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
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
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 185

### Candidate
type: A, layer: 1, pos: 185

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9693656, upper bound: 173.9699542
time: 6.14 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9762404, upper bound: 173.9762404
time: 6.77 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 24.50 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.50
Output dim: 7, lower bound: -173.9921561, upper bound: 173.9962535
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.50
Output dim: 7, lower bound: -173.9939648, upper bound: 173.9980547
NS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 24.50
Output dim: 7, lower bound: -173.9934280, upper bound: 173.9977270
NS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 24.50
Output dim: 7, lower bound: -173.9939648, upper bound: 173.9980546
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.50
Output dim: 7, lower bound: -173.9800736, upper bound: 173.9855334
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.50
Output dim: 7, lower bound: -173.9867920, upper bound: 173.9918835
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.50
Output dim: 7, lower bound: -173.9800736, upper bound: 173.9855333
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.50
Output dim: 7, lower bound: -173.9867920, upper bound: 173.9918835
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.50
Output dim: 7, lower bound: -173.9754658, upper bound: 173.9745976
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.50
Output dim: 7, lower bound: -173.9785582, upper bound: 173.9776385
NS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 24.50
Output dim: 7, lower bound: -173.9716812, upper bound: 173.9705325
NS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 24.50
Output dim: 7, lower bound: -173.9785582, upper bound: 173.9776385
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.50
Output dim: 7, lower bound: -173.9693656, upper bound: 173.9699542
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.50
Output dim: 7, lower bound: -173.9762404, upper bound: 173.9762404
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.50
Output dim: 7, lower bound: -173.9693656, upper bound: 173.9699542
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.50
Output dim: 7, lower bound: -173.9762404, upper bound: 173.9762404

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -89.2786560, 70.9336929, -85.8327560, 68.2132339, -157.4918823, 156.7664490
1: -74.8124237, 62.9082603, -71.9396057, 60.4833603, -135.2957764, 134.8478699
2: -98.5339355, 64.3487930, -94.7567902, 61.9295082, -160.4634399, 159.1055756
3: -104.5019760, 54.9684753, -100.4537582, 52.8393555, -157.3413239, 155.4222260
4: -95.4807892, 73.4818344, -91.8001785, 70.6590805, -166.1398621, 165.2820129
5: -85.6778641, 66.6640625, -82.3655396, 64.1124725, -149.7903442, 149.0295868
6: -82.1303406, 79.1258698, -78.9549484, 76.0818863, -158.2122040, 158.0807953
7: -89.8783493, 75.7923279, -86.4394531, 72.9127731, -162.7911224, 162.2317810
8: -108.1262360, 73.2069855, -103.9950714, 70.3880157, -178.5142517, 177.2020569
9: -81.9471512, 80.0796967, -78.8168488, 77.0000534, -158.9471741, 158.8965149

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9942923, upper bound: 173.9987485
time: 7.27 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9942923, upper bound: 173.9987485
time: 8.68 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -90.2932358, 71.7403793, -89.2600250, 70.9226837, -161.2159119, 161.0003815
1: -75.6616898, 63.6242294, -74.8030090, 62.9004250, -138.5621033, 138.4272003
2: -99.6520691, 65.0642014, -98.5229111, 64.3406143, -163.9926758, 163.5870972
3: -105.7008133, 55.5959358, -104.4889069, 54.9603119, -160.6611176, 160.0848236
4: -96.5704956, 74.3148499, -95.4715118, 73.4725418, -170.0430298, 169.7863617
5: -86.6565018, 67.4175034, -85.6643524, 66.6528397, -153.3093109, 153.0818481
6: -83.0643082, 80.0239563, -82.1212845, 79.1160812, -162.1803894, 162.1452179
7: -90.8969421, 76.6452560, -89.8689575, 75.7819595, -166.6788788, 166.5142212
8: -109.3459930, 74.0346680, -108.1104126, 73.1956482, -182.5416260, 182.1450806
9: -82.8727036, 80.9837341, -81.9355164, 80.0737457, -162.9464264, 162.9192505

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9960114, upper bound: 174.0007394
time: 7.37 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9960114, upper bound: 174.0008380
time: 7.46 seconds

## BFS NS instance: NS_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -100.1949768, 79.5902710, -89.5372009, 71.1405258, -171.3355103, 169.1274567
1: -83.8938675, 70.5323029, -75.0306091, 63.0927162, -146.9865875, 145.5629120
2: -110.5122833, 72.0789261, -98.8213959, 64.5355377, -175.0478210, 170.9003143
3: -117.2536697, 61.5142593, -104.8101044, 55.1297379, -172.3834076, 166.3243713
4: -107.2626877, 82.3863602, -95.7616501, 73.6968002, -180.9594879, 178.1480103
5: -96.1841736, 74.7729797, -85.9276810, 66.8609924, -163.0451355, 160.7006378
6: -92.1473007, 88.7046967, -82.3705063, 79.3567657, -171.5040588, 171.0751953
7: -100.7457275, 84.8563232, -90.1408539, 76.0131226, -176.7588196, 174.9971771
8: -121.1972733, 82.1510849, -108.4400558, 73.4213791, -194.6186523, 190.5911407
9: -91.8075790, 89.8104477, -82.1869278, 80.3160934, -172.1236572, 171.9973755

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 197

### Candidate
type: A, layer: 1, pos: 105

### Candidate
type: A, layer: 1, pos: 185

### Candidate
type: B, layer: 1, pos: 185

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A2_A1_B1

### Relational analysis result of NS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9921340, upper bound: 173.9962535
time: 10.01 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2

### Relational analysis result of NS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9921340, upper bound: 173.9977270
time: 9.11 seconds

## BFS NS instance: NS_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -103.6005707, 82.2880325, -90.5516205, 71.9470673, -175.5476379, 172.8396606
1: -86.7420731, 72.9369888, -75.8797226, 63.8085594, -150.5506134, 148.8167114
2: -114.2624054, 74.4783173, -99.9393463, 65.2508240, -179.5131836, 174.4176636
3: -121.2664795, 63.6286545, -106.0087204, 55.7570877, -177.0235443, 169.6373749
4: -110.9112015, 85.1897202, -96.8511734, 74.5296707, -185.4408722, 182.0408936
5: -99.4648285, 77.3036957, -86.9061432, 67.6142883, -167.0790863, 164.2098236
6: -95.2970734, 91.7267914, -83.3042831, 80.2546768, -175.5517578, 175.0310516
7: -104.1606598, 87.7138138, -91.1592560, 76.8659134, -181.0265656, 178.8730774
8: -125.2995682, 84.9462662, -109.6595840, 74.2489090, -199.5484772, 194.6058350
9: -94.9145508, 92.8713989, -83.1123276, 81.2199783, -176.1345215, 175.9837341

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 197

### Candidate
type: A, layer: 1, pos: 105

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A2_A2_B1

### Relational analysis result of NS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9921561, upper bound: 173.9962535
time: 9.57 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2

### Relational analysis result of NS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9921561, upper bound: 173.9980547
time: 8.82 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -89.2786560, 70.9336929, -100.4557114, 79.7988586, -169.0774689, 171.3894043
1: -74.8124237, 62.9082603, -84.1139374, 70.7182770, -145.5307007, 147.0222015
2: -98.5339355, 64.3487930, -110.8021317, 72.2673111, -170.8012390, 175.1509247
3: -104.5019760, 54.9684753, -117.5643845, 61.6768341, -166.1788025, 172.5328522
4: -95.4807892, 73.4818344, -107.5459290, 82.6030884, -178.0838623, 181.0277710
5: -85.6778641, 66.6640625, -96.4361267, 74.9715652, -160.6494293, 163.1001740
6: -82.1303406, 79.1258698, -92.3894424, 88.9375305, -171.0678711, 171.5153046
7: -89.8783493, 75.7923279, -101.0104370, 85.0789337, -174.9572754, 176.8027649
8: -108.1262360, 73.2069855, -121.5136795, 82.3673401, -190.4935608, 194.7206726
9: -81.9471512, 80.0796967, -92.0493546, 90.0487137, -171.9958496, 172.1290588

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
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
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 197

### Candidate
type: B, layer: 1, pos: 250

### Candidate
type: B, layer: 1, pos: 105

### Candidate
type: B, layer: 1, pos: 185

### Candidate
type: A, layer: 1, pos: 185

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9801000, upper bound: 173.9855730
time: 8.65 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9801000, upper bound: 173.9855729
time: 8.29 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -90.2932358, 71.7403793, -103.8624954, 82.4976349, -172.7908630, 175.6028595
1: -75.6616898, 63.6242294, -86.9631882, 73.1238937, -148.7855835, 150.5873871
2: -99.6520691, 65.0642014, -114.5536346, 74.6676941, -174.3197632, 179.6178284
3: -105.7008133, 55.5959358, -121.5786133, 63.7920189, -169.4928284, 177.1745453
4: -96.5704956, 74.3148499, -111.1957855, 85.4075165, -181.9780121, 185.5106201
5: -86.6565018, 67.4175034, -99.7179337, 77.5032578, -164.1597290, 167.1354370
6: -83.0643082, 80.0239563, -95.5403290, 91.9608231, -175.0251312, 175.5642548
7: -90.8969421, 76.6452560, -104.4266281, 87.9375076, -178.8344269, 181.0718842
8: -109.3459930, 74.0346680, -125.6175308, 85.1635666, -194.5095520, 199.6521912
9: -82.8727036, 80.9837341, -95.1574936, 93.1108627, -175.9835663, 176.1412048

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 197

### Candidate
type: B, layer: 1, pos: 105

### Candidate
type: B, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9865720, upper bound: 173.9917482
time: 8.01 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9865720, upper bound: 173.9919234
time: 7.73 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -103.8840485, 82.5110016, -100.4557114, 79.7988586, -183.6828613, 182.9667053
1: -86.9753036, 73.1341324, -84.1139374, 70.7182770, -157.6935577, 157.2480774
2: -114.5680161, 74.6783447, -110.8021317, 72.2673111, -186.8353271, 185.4804688
3: -121.5959015, 63.8023109, -117.5643845, 61.6768341, -183.2726746, 181.3666992
4: -111.2086639, 85.4198456, -107.5459290, 82.6030884, -193.8116913, 192.9657745
5: -99.7347488, 77.5163498, -96.4361267, 74.9715652, -174.7063141, 173.9524536
6: -95.5529099, 91.9731979, -92.3894424, 88.9375305, -184.4904480, 184.3626404
7: -104.4392395, 87.9508820, -101.0104370, 85.0789337, -189.5181427, 188.9613190
8: -125.6373062, 85.1772995, -121.5136795, 82.3673401, -208.0046234, 206.6909332
9: -95.1721725, 93.1197205, -92.0493546, 90.0487137, -185.2208710, 185.1690674

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9800736, upper bound: 173.9855334
time: 7.59 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9800736, upper bound: 173.9855334
time: 8.68 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -104.8955765, 83.3147507, -103.8624954, 82.4976349, -187.3932037, 187.1772461
1: -87.8215637, 73.8474655, -86.9631882, 73.1238937, -160.9454651, 160.8106232
2: -115.6823349, 75.3911591, -114.5536346, 74.6676941, -190.3500061, 189.9447937
3: -122.7902298, 64.4275360, -121.5786133, 63.7920189, -186.5822449, 186.0061340
4: -112.2945633, 86.2497101, -111.1957855, 85.4075165, -197.7020874, 197.4454651
5: -100.7099152, 78.2674026, -99.7179337, 77.5032578, -178.2131653, 177.9853210
6: -96.4834442, 92.8683167, -95.5403290, 91.9608231, -188.4442749, 188.4086456
7: -105.4542313, 88.8007278, -104.4266281, 87.9375076, -193.3917389, 193.2273560
8: -126.8530655, 86.0020294, -125.6175308, 85.1635666, -212.0166321, 211.6195679
9: -96.0946579, 94.0206070, -95.1574936, 93.1108627, -189.2055206, 189.1780853

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9865267, upper bound: 173.9916923
time: 9.22 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9865267, upper bound: 173.9918835
time: 7.96 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -92.0950928, 73.0158463, -84.8143005, 67.3976364, -159.4927216, 157.8301392
1: -77.0024414, 64.6811905, -71.0791397, 59.7547226, -136.7571411, 135.7603302
2: -101.4136429, 66.0933228, -93.6229324, 61.1911964, -162.6048431, 159.7162476
3: -107.5220947, 56.4257317, -99.2381897, 52.2022858, -159.7243652, 155.6639099
4: -98.2966995, 75.4873810, -90.6923904, 69.8093491, -168.1060333, 166.1797638
5: -88.2497101, 68.2800446, -81.3807983, 63.3333359, -151.5830383, 149.6608124
6: -84.6057358, 81.4677505, -78.0076981, 75.1703491, -159.7760620, 159.4754181
7: -92.4325867, 77.8630447, -85.4027481, 72.0401382, -164.4726868, 163.2657776
8: -111.4800186, 75.2515945, -102.7581711, 69.5417862, -181.0218048, 178.0097656
9: -84.2001724, 82.2492599, -77.8693619, 76.0652237, -160.2653961, 160.1186218

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9849823, upper bound: 173.9849823
time: 6.18 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9849823, upper bound: 173.9853952
time: 5.80 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -93.1767197, 73.8778152, -88.2456055, 70.1103363, -163.2870178, 162.1233826
1: -77.9096222, 65.4461746, -73.9459763, 62.1748390, -140.0844574, 139.3921356
2: -102.6084213, 66.8578720, -97.3935089, 63.6053123, -166.2137299, 164.2513428
3: -108.8029099, 57.0952072, -103.2785339, 54.3258057, -163.1287079, 160.3737335
4: -99.4595108, 76.3773117, -94.3681564, 72.6265030, -172.0860138, 170.7454376
5: -89.2952042, 69.0834351, -84.6834564, 65.8769531, -155.1721497, 153.7668915
6: -85.6045837, 82.4269028, -81.1778793, 78.2082977, -163.8128662, 163.6047821
7: -93.5212326, 78.7737732, -88.8364258, 74.9130325, -168.4342651, 167.6101990
8: -112.7840271, 76.1354599, -106.8787003, 72.3528290, -185.1368561, 183.0141602
9: -85.1876602, 83.2167969, -80.9920044, 79.1429138, -164.3305664, 164.2088013

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9853952, upper bound: 173.9852847
time: 7.15 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9853952, upper bound: 173.9869953
time: 6.58 seconds

## BFS NS instance: NS_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -102.7896500, 81.4954071, -88.5245132, 70.3295288, -173.1191711, 170.0199280
1: -85.9012985, 72.1511002, -74.1749802, 62.3683624, -148.2696533, 146.3260803
2: -113.1469727, 73.6645584, -97.6939163, 63.8014412, -176.9484100, 171.3584747
3: -120.0292740, 62.8438759, -103.6018066, 54.4963455, -174.5256195, 166.4456329
4: -109.8447876, 84.2120132, -94.6601410, 72.8521729, -182.6969604, 178.8721619
5: -98.5452652, 76.2342987, -84.9484482, 66.0863647, -164.6316223, 161.1827393
6: -94.4166565, 90.8490753, -81.4286957, 78.4504929, -172.8671265, 172.2777710
7: -103.0710907, 86.7449951, -89.1100540, 75.1456528, -178.2167358, 175.8550415
8: -124.2748337, 84.0079117, -107.2104492, 72.5800171, -196.8548431, 191.2183533
9: -93.8563538, 91.7773132, -81.2449799, 79.3868256, -173.2431793, 173.0222931

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_B1_A2_A1_B1

### Relational analysis result of NS_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9714023, upper bound: 173.9703100
time: 6.77 seconds

## Relational analysis of NS_A2_B1_A2_A1_B2

### Relational analysis result of NS_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9714023, upper bound: 173.9705325
time: 6.29 seconds

## BFS NS instance: NS_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -106.3637085, 84.3330307, -89.5385284, 71.1358109, -177.4995117, 173.8715515
1: -88.8937073, 74.6816940, -75.0238419, 63.0839653, -151.9776611, 149.7055359
2: -117.0896606, 76.1957092, -98.8114624, 64.5164948, -181.6061554, 175.0071411
3: -124.2422256, 65.0608749, -104.7999649, 55.1234665, -179.3656921, 169.8608246
4: -113.6769562, 87.1605911, -95.7493057, 73.6847763, -187.3617249, 182.9098816
5: -101.9914322, 78.8944473, -85.9265747, 66.8394012, -168.8308411, 164.8210144
6: -97.7284241, 94.0237427, -82.3621216, 79.3481216, -177.0765381, 176.3858643
7: -106.6670456, 89.7513733, -90.1281052, 75.9981918, -182.6652222, 179.8794861
8: -128.5891876, 86.9469452, -108.4295425, 73.4072495, -201.9964142, 195.3764648
9: -97.1227875, 95.0028152, -82.1700439, 80.2903748, -177.4131622, 177.1728516

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_B1_A2_A2_B1

### Relational analysis result of NS_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9754658, upper bound: 173.9745977
time: 6.99 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2

### Relational analysis result of NS_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9754658, upper bound: 173.9776385
time: 9.88 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -92.0950928, 73.0158463, -99.4617538, 79.0029831, -171.0980530, 172.4775848
1: -77.0024414, 64.6811905, -83.2742691, 70.0072479, -147.0096893, 147.9554596
2: -101.4136429, 66.0933228, -109.6957474, 71.5466537, -172.9602966, 175.7890625
3: -107.5220947, 56.4257317, -116.3787003, 61.0548782, -168.5769653, 172.8044281
4: -98.2966995, 75.4873810, -106.4649048, 81.7741318, -180.0708160, 181.9522552
5: -88.2497101, 68.2800446, -95.4752884, 74.2111664, -162.4608612, 163.7553406
6: -84.6057358, 81.4677505, -91.4650574, 88.0481262, -172.6538391, 172.9327850
7: -92.4325867, 77.8630447, -99.9987335, 84.2274551, -176.6600342, 177.8617706
8: -111.4800186, 75.2515945, -120.3063736, 81.5414200, -193.0214386, 195.5579681
9: -84.2001724, 82.2492599, -91.1248474, 89.1367569, -173.3369141, 173.3741150

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 197

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 105

### Candidate
type: B, layer: 1, pos: 250

### Candidate
type: B, layer: 1, pos: 251

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 185

### Candidate
type: A, layer: 1, pos: 185

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9703100, upper bound: 173.9714023
time: 7.10 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9703100, upper bound: 173.9716812
time: 6.82 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -93.1767197, 73.8778152, -102.8739700, 81.7063599, -174.8830566, 176.7517548
1: -77.9096222, 65.4461746, -86.1282349, 72.4170609, -150.3266907, 151.5744019
2: -102.6084213, 66.8578720, -113.4533615, 73.9513092, -176.5597229, 180.3112030
3: -108.8029099, 57.0952072, -120.3996811, 63.1737900, -171.9766998, 177.4948730
4: -99.4595108, 76.3773117, -110.1208038, 84.5833969, -184.0428772, 186.4980774
5: -89.2952042, 69.0834351, -98.7624664, 76.7472687, -166.0424805, 167.8459015
6: -85.6045837, 82.4269028, -94.6210098, 91.0766449, -176.6812286, 177.0479126
7: -93.5212326, 78.7737732, -103.4206619, 87.0909958, -180.6122131, 182.1944275
8: -112.7840271, 76.1354599, -124.4172440, 84.3424225, -197.1264343, 200.5526886
9: -85.1876602, 83.2167969, -94.2382126, 92.2043610, -177.3920288, 177.4550171

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 73

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 197

### Candidate
type: B, layer: 1, pos: 251

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9745977, upper bound: 173.9754658
time: 8.10 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9745977, upper bound: 173.9785582
time: 7.41 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -106.6389618, 84.5502930, -99.4617538, 79.0029831, -185.6419220, 184.0120544
1: -89.1205826, 74.8732605, -83.2742691, 70.0072479, -159.1278381, 158.1475220
2: -117.3879852, 76.3902740, -109.6957474, 71.5466537, -188.9346313, 186.0860138
3: -124.5611954, 65.2285309, -116.3787003, 61.0548782, -185.6160736, 181.6072388
4: -113.9665527, 87.3846512, -106.4649048, 81.7741318, -195.7406616, 193.8495026
5: -102.2542877, 79.1007156, -95.4752884, 74.2111664, -176.4654388, 174.5760040
6: -97.9763412, 94.2637863, -91.4650574, 88.0481262, -186.0244446, 185.7288513
7: -106.9383774, 89.9817505, -99.9987335, 84.2274551, -191.1658325, 189.9804688
8: -128.9189911, 87.1714554, -120.3063736, 81.5414200, -210.4604187, 207.4778290
9: -97.3733749, 95.2443085, -91.1248474, 89.1367569, -186.5101013, 186.3691254

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9684781, upper bound: 173.9684782
time: 5.61 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9684781, upper bound: 173.9699542
time: 6.20 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -107.7213058, 85.4121399, -102.8739700, 81.7063599, -189.4276581, 188.2861023
1: -90.0277786, 75.6378479, -86.1282349, 72.4170609, -162.4448395, 161.7660675
2: -118.5827637, 77.1550980, -113.4533615, 73.9513092, -192.5340729, 190.6084595
3: -125.8402481, 65.8972092, -120.3996811, 63.1737900, -189.0140381, 186.2968750
4: -115.1302032, 88.2740631, -110.1208038, 84.5833969, -199.7135773, 198.3948517
5: -103.2999344, 79.9042740, -98.7624664, 76.7472687, -180.0471802, 178.6667480
6: -98.9749222, 95.2232056, -94.6210098, 91.0766449, -190.0515747, 189.8442078
7: -108.0271759, 90.8925323, -103.4206619, 87.0909958, -195.1181488, 194.3132019
8: -130.2233887, 88.0552979, -124.4172440, 84.3424225, -214.5658112, 212.4725189
9: -98.3612976, 96.2114182, -94.2382126, 92.2043610, -190.5656586, 190.4496307

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9699542, upper bound: 173.9693658
time: 6.16 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9699542, upper bound: 173.9762404
time: 6.01 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 13.07 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 7, lower bound: -173.9942923, upper bound: 173.9987485
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 7, lower bound: -173.9942923, upper bound: 173.9987485
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 7, lower bound: -173.9960114, upper bound: 174.0007394
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 7, lower bound: -173.9960114, upper bound: 174.0008380
NS_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 7, lower bound: -173.9921340, upper bound: 173.9962535
NS_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 7, lower bound: -173.9921340, upper bound: 173.9977270
NS_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 7, lower bound: -173.9921561, upper bound: 173.9962535
NS_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 7, lower bound: -173.9921561, upper bound: 173.9980547
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 7, lower bound: -173.9801000, upper bound: 173.9855730
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 7, lower bound: -173.9801000, upper bound: 173.9855729
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 7, lower bound: -173.9865720, upper bound: 173.9917482
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 7, lower bound: -173.9865720, upper bound: 173.9919234
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 7, lower bound: -173.9800736, upper bound: 173.9855334
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 7, lower bound: -173.9800736, upper bound: 173.9855334
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 7, lower bound: -173.9865267, upper bound: 173.9916923
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 7, lower bound: -173.9865267, upper bound: 173.9918835
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 7, lower bound: -173.9849823, upper bound: 173.9849823
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 7, lower bound: -173.9849823, upper bound: 173.9853952
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 7, lower bound: -173.9853952, upper bound: 173.9852847
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 7, lower bound: -173.9853952, upper bound: 173.9869953
NS_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 7, lower bound: -173.9714023, upper bound: 173.9703100
NS_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 7, lower bound: -173.9714023, upper bound: 173.9705325
NS_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 7, lower bound: -173.9754658, upper bound: 173.9745977
NS_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 7, lower bound: -173.9754658, upper bound: 173.9776385
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 7, lower bound: -173.9703100, upper bound: 173.9714023
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 7, lower bound: -173.9703100, upper bound: 173.9716812
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 7, lower bound: -173.9745977, upper bound: 173.9754658
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 7, lower bound: -173.9745977, upper bound: 173.9785582
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 7, lower bound: -173.9684781, upper bound: 173.9684782
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 7, lower bound: -173.9684781, upper bound: 173.9699542
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 7, lower bound: -173.9699542, upper bound: 173.9693658
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 7, lower bound: -173.9699542, upper bound: 173.9762404

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -85.5749893, 68.0070419, -85.8327560, 68.2132339, -153.7882233, 153.8397980
1: -71.7220688, 60.2994690, -71.9396057, 60.4833603, -132.2054291, 132.2390747
2: -94.4702454, 61.7433624, -94.7567902, 61.9295082, -156.3997498, 156.5001221
3: -100.1464844, 52.6785698, -100.4537582, 52.8393555, -152.9858093, 153.1323090
4: -91.5201950, 70.4448166, -91.8001785, 70.6590805, -162.1792755, 162.2449799
5: -82.1164398, 63.9162254, -82.3655396, 64.1124725, -146.2289124, 146.2817230
6: -78.7155304, 75.8516998, -78.9549484, 76.0818863, -154.7973785, 154.8066406
7: -86.1777802, 72.6926804, -86.4394531, 72.9127731, -159.0905457, 159.1321411
8: -103.6823044, 70.1743546, -103.9950714, 70.3880157, -174.0702820, 174.1694183
9: -78.5778427, 76.7643967, -78.8168488, 77.0000534, -155.5778503, 155.5812378

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
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
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 185

### Candidate
type: A, layer: 1, pos: 185

### Candidate
type: B, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 251

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 197

### Candidate
type: A, layer: 1, pos: 251

### Candidate
type: B, layer: 1, pos: 105

### Candidate
type: A, layer: 1, pos: 197

### Candidate
type: A, layer: 1, pos: 105

### Candidate
type: B, layer: 1, pos: 96

### Candidate
type: A, layer: 1, pos: 96

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 195

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 11.25 + 590.41 = 601.66 seconds
