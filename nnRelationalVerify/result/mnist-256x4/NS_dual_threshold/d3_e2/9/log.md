## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.851544332


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.3877775, 0.4621432, -0.3877775, 0.4621432, -0.8499205, 0.8499205)
1: (-0.5284067, 2.0988278, -0.5284067, 2.0988278, -2.6272345, 2.6272345)
2: (-0.3809817, 0.5539122, -0.3809817, 0.5539122, -0.9348937, 0.9348938)
3: (-0.3085160, 0.3773541, -0.3085160, 0.3773541, -0.6858702, 0.6858701)
4: (-0.3952212, 0.4911242, -0.3952212, 0.4911242, -0.8863454, 0.8863454)
5: (-0.4354007, 0.5080089, -0.4354007, 0.5080089, -0.9434096, 0.9434096)
6: (-0.4014111, 0.4879104, -0.4014111, 0.4879104, -0.8893216, 0.8893216)
7: (-0.3190002, 0.8904487, -0.3190002, 0.8904487, -1.2094488, 1.2094488)
8: (-0.2782317, 0.7783343, -0.2782317, 0.7783343, -1.0565660, 1.0565660)
9: (-0.4433337, 0.5317172, -0.4433337, 0.5317172, -0.9750509, 0.9750510)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.60 + 6.18 = 8.78 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -1.8702468, upper bound: 1.8702468

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8644097, upper bound: 1.8641396
time: 5.11 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8637191, upper bound: 1.8637191
time: 3.40 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 8.77 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 8.77
Output dim: 1, lower bound: -1.8644097, upper bound: 1.8641396
NS_A2, status: Status.UNKNOWN, split count: 1, time: 8.77
Output dim: 1, lower bound: -1.8637191, upper bound: 1.8637191

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.3086789, 0.3845411, -0.3877775, 0.4621432, -0.7708222, 0.7723186
1: -0.3978511, 2.0840638, -0.5284067, 2.0988278, -2.4966788, 2.6124706
2: -0.3141212, 0.5062522, -0.3809817, 0.5539122, -0.8680333, 0.8872339
3: -0.2634890, 0.3113021, -0.3085160, 0.3773541, -0.6408432, 0.6198182
4: -0.3396746, 0.3952364, -0.3952212, 0.4911242, -0.8307988, 0.7904575
5: -0.3675907, 0.4242743, -0.4354007, 0.5080089, -0.8755996, 0.8596749
6: -0.3227596, 0.3972220, -0.4014111, 0.4879104, -0.8106701, 0.7986331
7: -0.2511618, 0.8166906, -0.3190002, 0.8904487, -1.1416104, 1.1356907
8: -0.2138748, 0.6786944, -0.2782317, 0.7783343, -0.9922090, 0.9569260
9: -0.3666869, 0.4395524, -0.4433337, 0.5317172, -0.8984041, 0.8828861

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8642649, upper bound: 1.8628641
time: 3.42 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8636804, upper bound: 1.8636839
time: 4.49 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.3600613, 0.4353982, -0.3779684, 0.4530229, -0.8130842, 0.8133665
1: -0.4991664, 2.0987844, -0.5132859, 2.0969214, -2.5960879, 2.6120703
2: -0.3610470, 0.5420573, -0.3733377, 0.5473928, -0.9084395, 0.9153949
3: -0.2990898, 0.3568479, -0.3027672, 0.3698272, -0.6689169, 0.6596152
4: -0.3783862, 0.4572270, -0.3877778, 0.4803051, -0.8586913, 0.8450048
5: -0.4189027, 0.4821023, -0.4277219, 0.4979005, -0.9168030, 0.9098243
6: -0.3731854, 0.4597680, -0.3915942, 0.4774922, -0.8506776, 0.8513621
7: -0.2936454, 0.8727159, -0.3107929, 0.8808785, -1.1745239, 1.1835089
8: -0.2565502, 0.7643448, -0.2704100, 0.7667919, -1.0233421, 1.0347548
9: -0.4198536, 0.4921616, -0.4336246, 0.5211008, -0.9409544, 0.9257861

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8579459, upper bound: 1.8453724
time: 3.60 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8448687, upper bound: 1.8448687
time: 3.58 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 9.31 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 9.31
Output dim: 1, lower bound: -1.8642649, upper bound: 1.8628641
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 9.31
Output dim: 1, lower bound: -1.8636804, upper bound: 1.8636839
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 9.31
Output dim: 1, lower bound: -1.8579459, upper bound: 1.8453724
NS_A2_A2, status: Status.VERIFIED, split count: 2, time: 9.31
Output dim: 1, lower bound: -1.8448687, upper bound: 1.8448687

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -0.2345495, 0.2987274, -0.3798329, 0.4540738, -0.6886233, 0.6785604
1: -0.2514833, 2.0442259, -0.5159626, 2.0942264, -2.3457098, 2.5601885
2: -0.2386668, 0.4500791, -0.3743581, 0.5483944, -0.7870611, 0.8244373
3: -0.2072299, 0.2497623, -0.3035169, 0.3707925, -0.5780225, 0.5532792
4: -0.2806634, 0.3030124, -0.3890208, 0.4819493, -0.7626127, 0.6920332
5: -0.2832995, 0.3479608, -0.4284273, 0.4993131, -0.7826126, 0.7763882
6: -0.2439782, 0.3079137, -0.3935231, 0.4786451, -0.7226232, 0.7014368
7: -0.2231200, 0.7107163, -0.3126340, 0.8815927, -1.1047127, 1.0233504
8: -0.1619445, 0.5447927, -0.2718595, 0.7669309, -0.9288754, 0.8166522
9: -0.2917673, 0.3640208, -0.4354405, 0.5249805, -0.8167478, 0.7994614

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 207

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_A1_B1

### Relational analysis result of NS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8535132, upper bound: 1.8371436
time: 3.35 seconds

## Relational analysis of NS_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8596861, upper bound: 1.8583611
time: 4.70 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -0.2863440, 0.3599369, -0.3877775, 0.4621432, -0.7484872, 0.7477144
1: -0.3600385, 2.0699120, -0.5284067, 2.0988278, -2.4588664, 2.5983186
2: -0.2933874, 0.4908197, -0.3809817, 0.5539122, -0.8472993, 0.8718014
3: -0.2486206, 0.2922820, -0.3085160, 0.3773541, -0.6259747, 0.6007981
4: -0.3237067, 0.3699315, -0.3952212, 0.4911242, -0.8148309, 0.7651527
5: -0.3450019, 0.4011392, -0.4354007, 0.5080089, -0.8530108, 0.8365399
6: -0.3010215, 0.3695923, -0.4014111, 0.4879104, -0.7889320, 0.7710034
7: -0.2381915, 0.7898034, -0.3190002, 0.8904487, -1.1286402, 1.1088036
8: -0.1973222, 0.6440070, -0.2782317, 0.7783343, -0.9756565, 0.9222385
9: -0.3457508, 0.4190580, -0.4433337, 0.5317172, -0.8774681, 0.8623917

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 90

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8628514, upper bound: 1.8636839
time: 4.20 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8628514, upper bound: 1.8636839
time: 4.60 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -0.3331268, 0.4088108, -0.3779684, 0.4530229, -0.7861496, 0.7867792
1: -0.4323189, 2.0893643, -0.5132859, 2.0969214, -2.5292401, 2.6026502
2: -0.3327410, 0.5141346, -0.3733377, 0.5473928, -0.8801337, 0.8874723
3: -0.2721563, 0.3310874, -0.3027672, 0.3698272, -0.6419835, 0.6338546
4: -0.3494902, 0.4229948, -0.3877778, 0.4803051, -0.8297952, 0.8107726
5: -0.3884487, 0.4459377, -0.4277219, 0.4979005, -0.8863491, 0.8736596
6: -0.3454409, 0.4268511, -0.3915942, 0.4774922, -0.8229331, 0.8184452
7: -0.2668676, 0.8218553, -0.3107929, 0.8808785, -1.1477461, 1.1326482
8: -0.2315861, 0.7199242, -0.2704100, 0.7667919, -0.9983780, 0.9903342
9: -0.3878319, 0.4655659, -0.4336246, 0.5211008, -0.9089327, 0.8991905

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8566855, upper bound: 1.8452446
time: 3.19 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8574788, upper bound: 1.8452403
time: 2.91 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 8.30 seconds
NS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 8.30
Output dim: 1, lower bound: -1.8535132, upper bound: 1.8371436
NS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 8.30
Output dim: 1, lower bound: -1.8596861, upper bound: 1.8583611
NS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 8.30
Output dim: 1, lower bound: -1.8628514, upper bound: 1.8636839
NS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 8.30
Output dim: 1, lower bound: -1.8628514, upper bound: 1.8636839
NS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 8.30
Output dim: 1, lower bound: -1.8566855, upper bound: 1.8452446
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 8.30
Output dim: 1, lower bound: -1.8574788, upper bound: 1.8452403

## BFS NS instance: NS_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.2298043, 0.2829679, -0.2643852, 0.3427441, -0.5725484, 0.5473531
1: -0.2150313, 2.0408254, -0.2710628, 2.0955586, -2.3105898, 2.3118882
2: -0.2218800, 0.4388362, -0.2659940, 0.4669740, -0.6888540, 0.7048302
3: -0.1943512, 0.2372375, -0.2153580, 0.2723556, -0.4667068, 0.4525956
4: -0.2678274, 0.2835899, -0.2988285, 0.3469647, -0.6147921, 0.5824184
5: -0.2634209, 0.3323546, -0.3098004, 0.3689702, -0.6323911, 0.6421550
6: -0.2267151, 0.2920613, -0.2778741, 0.3454761, -0.5721912, 0.5699354
7: -0.2195720, 0.6881295, -0.2349005, 0.7471330, -0.9667050, 0.9230300
8: -0.1530980, 0.5103065, -0.1771571, 0.5681529, -0.7212509, 0.6874636
9: -0.2743347, 0.3455255, -0.3130699, 0.3943698, -0.6687046, 0.6585954

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 169

## Relational analysis of NS_A1_A1_B1_B1

### Relational analysis result of NS_A1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8192061, upper bound: 1.8309705
time: 2.91 seconds

## Relational analysis of NS_A1_A1_B1_B2

### Relational analysis result of NS_A1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8192026, upper bound: 1.8036732
time: 3.93 seconds

## BFS NS instance: NS_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.2345495, 0.2987274, -0.3305845, 0.4078350, -0.6423845, 0.6293119
1: -0.2514833, 2.0442259, -0.4194051, 2.0863266, -2.3378100, 2.4636312
2: -0.2386668, 0.4500791, -0.3303705, 0.5150807, -0.7537475, 0.7804496
3: -0.2072299, 0.2497623, -0.2682309, 0.3304490, -0.5376789, 0.5179932
4: -0.2806634, 0.3030124, -0.3511074, 0.4250616, -0.7057250, 0.6541198
5: -0.2832995, 0.3479608, -0.3812450, 0.4444785, -0.7277781, 0.7292058
6: -0.2439782, 0.3079137, -0.3439345, 0.4224184, -0.6663966, 0.6518482
7: -0.2231200, 0.7107163, -0.2691463, 0.8284375, -1.0515575, 0.9798626
8: -0.1619445, 0.5447927, -0.2324067, 0.6848393, -0.8467838, 0.7771993
9: -0.2917673, 0.3640208, -0.3848193, 0.4716594, -0.7634267, 0.7488402

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 207

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_A1_B2_B1

### Relational analysis result of NS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8534904, upper bound: 1.8512126
time: 4.08 seconds

## Relational analysis of NS_A1_A1_B2_B2

### Relational analysis result of NS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8534956, upper bound: 1.8521726
time: 3.69 seconds

## BFS NS instance: NS_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.2863440, 0.3599369, -0.2940388, 0.3668055, -0.6531495, 0.6539757
1: -0.3600385, 2.0699120, -0.3693448, 2.0573845, -2.4174230, 2.4392567
2: -0.2933874, 0.4908197, -0.2975228, 0.4883900, -0.7817773, 0.7883425
3: -0.2486206, 0.2922820, -0.2462491, 0.2973567, -0.5459773, 0.5385311
4: -0.3237067, 0.3699315, -0.3231714, 0.3797382, -0.7034449, 0.6931030
5: -0.3450019, 0.4011392, -0.3473406, 0.4029028, -0.7479047, 0.7484798
6: -0.3010215, 0.3695923, -0.3091829, 0.3740433, -0.6750648, 0.6787753
7: -0.2381915, 0.7898034, -0.2475722, 0.7789363, -1.0171278, 1.0373757
8: -0.1973222, 0.6440070, -0.2026851, 0.6354150, -0.8327372, 0.8466921
9: -0.3457508, 0.4190580, -0.3508066, 0.4508612, -0.7966121, 0.7698646

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A1_A2_B1_B1

### Relational analysis result of NS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8628514, upper bound: 1.8636839
time: 3.66 seconds

## Relational analysis of NS_A1_A2_B1_B2

### Relational analysis result of NS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8628514, upper bound: 1.8636839
time: 3.73 seconds

## BFS NS instance: NS_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.2863440, 0.3599369, -0.3613373, 0.4354710, -0.7218150, 0.7212743
1: -0.3600385, 2.0699120, -0.4872911, 2.0839281, -2.4439664, 2.5572031
2: -0.2933874, 0.4908197, -0.3590131, 0.5356452, -0.8290324, 0.8498328
3: -0.2486206, 0.2922820, -0.2919193, 0.3558654, -0.6044860, 0.5842013
4: -0.3237067, 0.3699315, -0.3746938, 0.4604800, -0.7841867, 0.7446253
5: -0.3450019, 0.4011392, -0.4123543, 0.4789883, -0.8239900, 0.8134935
6: -0.3010215, 0.3695923, -0.3752787, 0.4570033, -0.7580248, 0.7448709
7: -0.2381915, 0.7898034, -0.2979783, 0.8608332, -1.0990247, 1.0877817
8: -0.1973222, 0.6440070, -0.2571046, 0.7407256, -0.9380478, 0.9011114
9: -0.3457508, 0.4190580, -0.4171436, 0.5095234, -0.8552742, 0.8362016

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 90

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A1_A2_B2_B1

### Relational analysis result of NS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8628514, upper bound: 1.8636839
time: 4.17 seconds

## Relational analysis of NS_A1_A2_B2_B2

### Relational analysis result of NS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8628514, upper bound: 1.8636839
time: 4.46 seconds

## BFS NS instance: NS_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.3257864, 0.4015283, -0.2865311, 0.3587788, -0.6845652, 0.6880593
1: -0.4207010, 2.0850296, -0.3552955, 2.0559170, -2.4766181, 2.4403253
2: -0.3263586, 0.5092986, -0.2905619, 0.4837215, -0.8100801, 0.7998604
3: -0.2675909, 0.3248172, -0.2415398, 0.2913662, -0.5589571, 0.5663571
4: -0.3440025, 0.4140051, -0.3182264, 0.3710832, -0.7150857, 0.7322315
5: -0.3817177, 0.4376813, -0.3401480, 0.3957557, -0.7774733, 0.7778293
6: -0.3384784, 0.4177796, -0.3013783, 0.3655156, -0.7039940, 0.7191578
7: -0.2612394, 0.8138248, -0.2427161, 0.7713549, -1.0325942, 1.0565408
8: -0.2257551, 0.7087903, -0.1971128, 0.6254674, -0.8512225, 0.9059032
9: -0.3807350, 0.4594309, -0.3440424, 0.4405777, -0.8213128, 0.8034732

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A2_A1_B1_B1

### Relational analysis result of NS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8566855, upper bound: 1.8452446
time: 4.35 seconds

## Relational analysis of NS_A2_A1_B1_B2

### Relational analysis result of NS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8566855, upper bound: 1.8452446
time: 4.66 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.3331268, 0.4088108, -0.3522813, 0.4264866, -0.7596134, 0.7610921
1: -0.4323189, 2.0893643, -0.4722395, 2.0823097, -2.5146286, 2.5616038
2: -0.3327410, 0.5141346, -0.3513889, 0.5303780, -0.8631191, 0.8655236
3: -0.2721563, 0.3310874, -0.2867550, 0.3485063, -0.6206626, 0.6178424
4: -0.3494902, 0.4229948, -0.3683323, 0.4496398, -0.7991300, 0.7913271
5: -0.3884487, 0.4459377, -0.4047469, 0.4693423, -0.8577909, 0.8506846
6: -0.3454409, 0.4268511, -0.3661974, 0.4465886, -0.7920295, 0.7930485
7: -0.2668676, 0.8218553, -0.2898344, 0.8525944, -1.1194620, 1.1116897
8: -0.2315861, 0.7199242, -0.2498346, 0.7292005, -0.9607866, 0.9697587
9: -0.3878319, 0.4655659, -0.4085838, 0.4989411, -0.8867730, 0.8741497

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A2_A1_B2_A1

### Relational analysis result of NS_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8574788, upper bound: 1.8444022
time: 3.47 seconds

## Relational analysis of NS_A2_A1_B2_A2

### Relational analysis result of NS_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8574788, upper bound: 1.8452403
time: 2.99 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 8.73 seconds
NS_A1_A1_B1_B1, status: Status.VERIFIED, split count: 4, time: 8.73
Output dim: 1, lower bound: -1.8192061, upper bound: 1.8309705
NS_A1_A1_B1_B2, status: Status.VERIFIED, split count: 4, time: 8.73
Output dim: 1, lower bound: -1.8192026, upper bound: 1.8036732
NS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 8.73
Output dim: 1, lower bound: -1.8534904, upper bound: 1.8512126
NS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 8.73
Output dim: 1, lower bound: -1.8534956, upper bound: 1.8521726
NS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 8.73
Output dim: 1, lower bound: -1.8628514, upper bound: 1.8636839
NS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 8.73
Output dim: 1, lower bound: -1.8628514, upper bound: 1.8636839
NS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 8.73
Output dim: 1, lower bound: -1.8628514, upper bound: 1.8636839
NS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 8.73
Output dim: 1, lower bound: -1.8628514, upper bound: 1.8636839
NS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 8.73
Output dim: 1, lower bound: -1.8566855, upper bound: 1.8452446
NS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 8.73
Output dim: 1, lower bound: -1.8566855, upper bound: 1.8452446
NS_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 8.73
Output dim: 1, lower bound: -1.8574788, upper bound: 1.8444022
NS_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 8.73
Output dim: 1, lower bound: -1.8574788, upper bound: 1.8452403

## BFS NS instance: NS_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.2338002, 0.2960885, -0.2603828, 0.3348743, -0.5686746, 0.5564713
1: -0.2462872, 2.0435536, -0.2871346, 2.0704286, -2.3167157, 2.3306882
2: -0.2359928, 0.4484234, -0.2660630, 0.4685717, -0.7045645, 0.7144865
3: -0.2053933, 0.2477151, -0.2202177, 0.2707455, -0.4761388, 0.4679328
4: -0.2786240, 0.2998246, -0.3011281, 0.3434686, -0.6220926, 0.6009527
5: -0.2802196, 0.3455874, -0.3111365, 0.3704793, -0.6506989, 0.6567239
6: -0.2411660, 0.3052713, -0.2738077, 0.3379792, -0.5791452, 0.5790790
7: -0.2225479, 0.7073341, -0.2329902, 0.7544655, -0.9770133, 0.9403243
8: -0.1605837, 0.5393349, -0.1770245, 0.5691727, -0.7297564, 0.7163594
9: -0.2890159, 0.3611580, -0.3158495, 0.3979073, -0.6869232, 0.6770075

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 169

## Relational analysis of NS_A1_A1_B2_B1_B1

### Relational analysis result of NS_A1_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8209479, upper bound: 1.8445328
time: 5.61 seconds

## Relational analysis of NS_A1_A1_B2_B1_B2

### Relational analysis result of NS_A1_A1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8212050, upper bound: 1.8159709
time: 4.49 seconds

## BFS NS instance: NS_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.2338610, 0.2963310, -0.3146613, 0.3959772, -0.6298382, 0.6109923
1: -0.2467912, 2.0437171, -0.3804992, 2.0949068, -2.3416979, 2.4242163
2: -0.2362548, 0.4485835, -0.3163133, 0.5061395, -0.7423943, 0.7648968
3: -0.2055692, 0.2479045, -0.2563439, 0.3168301, -0.5223993, 0.5042484
4: -0.2788174, 0.3001361, -0.3404524, 0.4086247, -0.6874421, 0.6405885
5: -0.2805142, 0.3458189, -0.3660244, 0.4278408, -0.7083550, 0.7118433
6: -0.2414188, 0.3055241, -0.3284384, 0.4048879, -0.6463068, 0.6339625
7: -0.2226021, 0.7077343, -0.2545208, 0.8192825, -1.0418845, 0.9622551
8: -0.1607124, 0.5398385, -0.2193746, 0.6542954, -0.8150078, 0.7592131
9: -0.2892799, 0.3613614, -0.3676052, 0.4469694, -0.7362493, 0.7289666

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 169

## Relational analysis of NS_A1_A1_B2_B2_A1

### Relational analysis result of NS_A1_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8472338, upper bound: 1.8174895
time: 2.71 seconds

## Relational analysis of NS_A1_A1_B2_B2_A2

### Relational analysis result of NS_A1_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8213172, upper bound: 1.8179016
time: 3.55 seconds

## BFS NS instance: NS_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.2863440, 0.3599369, -0.2345495, 0.2987274, -0.5850714, 0.5944864
1: -0.3600385, 2.0699120, -0.2514833, 2.0442259, -2.4042645, 2.3213954
2: -0.2933874, 0.4908197, -0.2386668, 0.4500791, -0.7434665, 0.7294865
3: -0.2486206, 0.2922820, -0.2072299, 0.2497623, -0.4983829, 0.4995120
4: -0.3237067, 0.3699315, -0.2806634, 0.3030124, -0.6267191, 0.6505949
5: -0.3450019, 0.4011392, -0.2832995, 0.3479608, -0.6929628, 0.6844387
6: -0.3010215, 0.3695923, -0.2439782, 0.3079137, -0.6089352, 0.6135705
7: -0.2381915, 0.7898034, -0.2231200, 0.7107163, -0.9489079, 1.0129235
8: -0.1973222, 0.6440070, -0.1619445, 0.5447927, -0.7421149, 0.8059515
9: -0.3457508, 0.4190580, -0.2917673, 0.3640208, -0.7097716, 0.7108253

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 207

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 169

## Relational analysis of NS_A1_A2_B1_B1_B1

### Relational analysis result of NS_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8297478, upper bound: 1.8577733
time: 5.23 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2

### Relational analysis result of NS_A1_A2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8300322, upper bound: 1.8288496
time: 3.69 seconds

## BFS NS instance: NS_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.2863440, 0.3599369, -0.2788675, 0.3499624, -0.6363064, 0.6388044
1: -0.3600385, 2.0699120, -0.3509177, 2.0593367, -2.4193752, 2.4208298
2: -0.2933874, 0.4908197, -0.2853840, 0.4833812, -0.7767685, 0.7762037
3: -0.2486206, 0.2922820, -0.2423455, 0.2868207, -0.5354412, 0.5346276
4: -0.3237067, 0.3699315, -0.3163882, 0.3595114, -0.6832181, 0.6863197
5: -0.3450019, 0.4011392, -0.3372528, 0.3928973, -0.7378992, 0.7383920
6: -0.3010215, 0.3695923, -0.2934355, 0.3593891, -0.6604106, 0.6630278
7: -0.2381915, 0.7898034, -0.2371885, 0.7703673, -1.0085589, 1.0269920
8: -0.1973222, 0.6440070, -0.1924376, 0.6338642, -0.8311864, 0.8364446
9: -0.3457508, 0.4190580, -0.3402025, 0.4192069, -0.7649577, 0.7592604

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 169

## Relational analysis of NS_A1_A2_B1_B2_A1

### Relational analysis result of NS_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8559960, upper bound: 1.8284840
time: 3.98 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2

### Relational analysis result of NS_A1_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8300322, upper bound: 1.8288496
time: 4.09 seconds

## BFS NS instance: NS_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.2863440, 0.3599369, -0.2863440, 0.3599369, -0.6462809, 0.6462809
1: -0.3600385, 2.0699120, -0.3600385, 2.0699120, -2.4299505, 2.4299505
2: -0.2933874, 0.4908197, -0.2933874, 0.4908197, -0.7842070, 0.7842070
3: -0.2486206, 0.2922820, -0.2486206, 0.2922820, -0.5409026, 0.5409026
4: -0.3237067, 0.3699315, -0.3237067, 0.3699315, -0.6936382, 0.6936382
5: -0.3450019, 0.4011392, -0.3450019, 0.4011392, -0.7461411, 0.7461411
6: -0.3010215, 0.3695923, -0.3010215, 0.3695923, -0.6706138, 0.6706138
7: -0.2381915, 0.7898034, -0.2381915, 0.7898034, -1.0279950, 1.0279950
8: -0.1973222, 0.6440070, -0.1973222, 0.6440070, -0.8413292, 0.8413292
9: -0.3457508, 0.4190580, -0.3457508, 0.4190580, -0.7648088, 0.7648088

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 169

## Relational analysis of NS_A1_A2_B2_B1_A1

### Relational analysis result of NS_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8559960, upper bound: 1.8284333
time: 3.89 seconds

## Relational analysis of NS_A1_A2_B2_B1_A2

### Relational analysis result of NS_A1_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8300322, upper bound: 1.8288300
time: 4.27 seconds

## BFS NS instance: NS_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.2863440, 0.3599369, -0.3347340, 0.4103626, -0.6967067, 0.6946709
1: -0.3600385, 2.0699120, -0.4598646, 2.0842080, -2.4442465, 2.5297766
2: -0.2933874, 0.4908197, -0.3392447, 0.5255044, -0.8188917, 0.8300644
3: -0.2486206, 0.2922820, -0.2835303, 0.3353857, -0.5840062, 0.5758123
4: -0.3237067, 0.3699315, -0.3595104, 0.4263583, -0.7500650, 0.7294420
5: -0.3450019, 0.4011392, -0.3959390, 0.4538332, -0.7988352, 0.7970781
6: -0.3010215, 0.3695923, -0.3493306, 0.4285127, -0.7295341, 0.7189229
7: -0.2381915, 0.7898034, -0.2732013, 0.8451506, -1.0833421, 1.0630047
8: -0.1973222, 0.6440070, -0.2365433, 0.7263331, -0.9236554, 0.8805503
9: -0.3457508, 0.4190580, -0.3955366, 0.4713076, -0.8170584, 0.8145946

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 39

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 169

## Relational analysis of NS_A1_A2_B2_B2_A1

### Relational analysis result of NS_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8559960, upper bound: 1.8284333
time: 5.56 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2

### Relational analysis result of NS_A1_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8300322, upper bound: 1.8288300
time: 3.97 seconds

## BFS NS instance: NS_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.3257864, 0.4015283, -0.2345495, 0.2987274, -0.6245139, 0.6360778
1: -0.4207010, 2.0850296, -0.2514833, 2.0442259, -2.4649270, 2.3365130
2: -0.3263586, 0.5092986, -0.2386668, 0.4500791, -0.7764376, 0.7479653
3: -0.2675909, 0.3248172, -0.2072299, 0.2497623, -0.5173532, 0.5320472
4: -0.3440025, 0.4140051, -0.2806634, 0.3030124, -0.6470150, 0.6946685
5: -0.3817177, 0.4376813, -0.2832995, 0.3479608, -0.7296785, 0.7209808
6: -0.3384784, 0.4177796, -0.2439782, 0.3079137, -0.6463921, 0.6617577
7: -0.2612394, 0.8138248, -0.2231200, 0.7107163, -0.9719557, 1.0369447
8: -0.2257551, 0.7087903, -0.1619445, 0.5447927, -0.7705477, 0.8707348
9: -0.3807350, 0.4594309, -0.2917673, 0.3640208, -0.7447559, 0.7511982

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 207

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A2_A1_B1_B1_A1

### Relational analysis result of NS_A2_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8306975, upper bound: 1.8338710
time: 3.02 seconds

## Relational analysis of NS_A2_A1_B1_B1_A2

### Relational analysis result of NS_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8522257, upper bound: 1.8410783
time: 3.83 seconds

## BFS NS instance: NS_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.3257864, 0.4015283, -0.2788675, 0.3499624, -0.6757488, 0.6803958
1: -0.4207010, 2.0850296, -0.3509177, 2.0593367, -2.4800377, 2.4359474
2: -0.3263586, 0.5092986, -0.2853840, 0.4833812, -0.8097397, 0.7946826
3: -0.2675909, 0.3248172, -0.2423455, 0.2868207, -0.5544115, 0.5671628
4: -0.3440025, 0.4140051, -0.3163882, 0.3595114, -0.7035140, 0.7303933
5: -0.3817177, 0.4376813, -0.3372528, 0.3928973, -0.7746149, 0.7749341
6: -0.3384784, 0.4177796, -0.2934355, 0.3593891, -0.6978675, 0.7112150
7: -0.2612394, 0.8138248, -0.2371885, 0.7703673, -1.0316067, 1.0510132
8: -0.2257551, 0.7087903, -0.1924376, 0.6338642, -0.8596193, 0.9012280
9: -0.3807350, 0.4594309, -0.3402025, 0.4192069, -0.7999419, 0.7996334

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A2_A1_B1_B2_A1

### Relational analysis result of NS_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8566855, upper bound: 1.8445494
time: 4.26 seconds

## Relational analysis of NS_A2_A1_B1_B2_A2

### Relational analysis result of NS_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8566855, upper bound: 1.8452446
time: 3.68 seconds

## BFS NS instance: NS_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.2559653, 0.3234926, -0.3522813, 0.4264866, -0.6824519, 0.6757740
1: -0.2875144, 2.0500207, -0.4722395, 2.0823097, -2.3698242, 2.5222602
2: -0.2589777, 0.4559059, -0.3513889, 0.5303780, -0.7893557, 0.8072948
3: -0.2160270, 0.2655704, -0.2867550, 0.3485063, -0.5645333, 0.5523254
4: -0.2904082, 0.3297734, -0.3683323, 0.4496398, -0.7400481, 0.6981057
5: -0.3069920, 0.3635661, -0.4047469, 0.4693423, -0.7763343, 0.7683130
6: -0.2672317, 0.3319142, -0.3661974, 0.4465886, -0.7138203, 0.6981117
7: -0.2283156, 0.7205522, -0.2898344, 0.8525944, -1.0809100, 1.0103866
8: -0.1723110, 0.5923766, -0.2498346, 0.7292005, -0.9015114, 0.8422112
9: -0.3122654, 0.3926581, -0.4085838, 0.4989411, -0.8112065, 0.8012419

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 156

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 169

## Relational analysis of NS_A2_A1_B2_A1_A1

### Relational analysis result of NS_A2_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8459996, upper bound: 1.8080319
time: 3.40 seconds

## Relational analysis of NS_A2_A1_B2_A1_A2

### Relational analysis result of NS_A2_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8213536, upper bound: 1.8082949
time: 4.61 seconds

## BFS NS instance: NS_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.3091097, 0.3839549, -0.3522813, 0.4264866, -0.7355963, 0.7362362
1: -0.3937458, 2.0748887, -0.4722395, 2.0823097, -2.4760556, 2.5471282
2: -0.3110401, 0.4979089, -0.3513889, 0.5303780, -0.8414181, 0.8492978
3: -0.2568540, 0.3097642, -0.2867550, 0.3485063, -0.6053603, 0.5965192
4: -0.3319300, 0.3928573, -0.3683323, 0.4496398, -0.7815698, 0.7611896
5: -0.3656092, 0.4192871, -0.4047469, 0.4693423, -0.8349515, 0.8240340
6: -0.3218983, 0.3969350, -0.3661974, 0.4465886, -0.7684870, 0.7631325
7: -0.2499000, 0.7945080, -0.2898344, 0.8525944, -1.1024944, 1.0843424
8: -0.2120178, 0.6841972, -0.2498346, 0.7292005, -0.9412183, 0.9340318
9: -0.3647132, 0.4448199, -0.4085838, 0.4989411, -0.8636543, 0.8534037

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A2_A1_B2_A2_B1

### Relational analysis result of NS_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8566855, upper bound: 1.8452403
time: 4.35 seconds

## Relational analysis of NS_A2_A1_B2_A2_B2

### Relational analysis result of NS_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8566855, upper bound: 1.8452403
time: 4.33 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 11.07 seconds
NS_A1_A1_B2_B1_B1, status: Status.VERIFIED, split count: 5, time: 11.07
Output dim: 1, lower bound: -1.8209479, upper bound: 1.8445328
NS_A1_A1_B2_B1_B2, status: Status.VERIFIED, split count: 5, time: 11.07
Output dim: 1, lower bound: -1.8212050, upper bound: 1.8159709
NS_A1_A1_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 11.07
Output dim: 1, lower bound: -1.8472338, upper bound: 1.8174895
NS_A1_A1_B2_B2_A2, status: Status.VERIFIED, split count: 5, time: 11.07
Output dim: 1, lower bound: -1.8213172, upper bound: 1.8179016
NS_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 11.07
Output dim: 1, lower bound: -1.8297478, upper bound: 1.8577733
NS_A1_A2_B1_B1_B2, status: Status.VERIFIED, split count: 5, time: 11.07
Output dim: 1, lower bound: -1.8300322, upper bound: 1.8288496
NS_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.07
Output dim: 1, lower bound: -1.8559960, upper bound: 1.8284840
NS_A1_A2_B1_B2_A2, status: Status.VERIFIED, split count: 5, time: 11.07
Output dim: 1, lower bound: -1.8300322, upper bound: 1.8288496
NS_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.07
Output dim: 1, lower bound: -1.8559960, upper bound: 1.8284333
NS_A1_A2_B2_B1_A2, status: Status.VERIFIED, split count: 5, time: 11.07
Output dim: 1, lower bound: -1.8300322, upper bound: 1.8288300
NS_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.07
Output dim: 1, lower bound: -1.8559960, upper bound: 1.8284333
NS_A1_A2_B2_B2_A2, status: Status.VERIFIED, split count: 5, time: 11.07
Output dim: 1, lower bound: -1.8300322, upper bound: 1.8288300
NS_A2_A1_B1_B1_A1, status: Status.VERIFIED, split count: 5, time: 11.07
Output dim: 1, lower bound: -1.8306975, upper bound: 1.8338710
NS_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.07
Output dim: 1, lower bound: -1.8522257, upper bound: 1.8410783
NS_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.07
Output dim: 1, lower bound: -1.8566855, upper bound: 1.8445494
NS_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.07
Output dim: 1, lower bound: -1.8566855, upper bound: 1.8452446
NS_A2_A1_B2_A1_A1, status: Status.VERIFIED, split count: 5, time: 11.07
Output dim: 1, lower bound: -1.8459996, upper bound: 1.8080319
NS_A2_A1_B2_A1_A2, status: Status.VERIFIED, split count: 5, time: 11.07
Output dim: 1, lower bound: -1.8213536, upper bound: 1.8082949
NS_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 11.07
Output dim: 1, lower bound: -1.8566855, upper bound: 1.8452403
NS_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 11.07
Output dim: 1, lower bound: -1.8566855, upper bound: 1.8452403

## BFS NS instance: NS_A1_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -0.2863440, 0.3599369, -0.2289875, 0.2888660, -0.5752100, 0.5889245
1: -0.3600385, 2.0699120, -0.2296490, 2.0131452, -2.3731837, 2.2995610
2: -0.2933874, 0.4908197, -0.2286664, 0.4427662, -0.7361535, 0.7194861
3: -0.2486206, 0.2922820, -0.2001629, 0.2416678, -0.4902884, 0.4924449
4: -0.3237067, 0.3699315, -0.2726810, 0.2911071, -0.6148137, 0.6426126
5: -0.3450019, 0.4011392, -0.2719472, 0.3386850, -0.6836870, 0.6730864
6: -0.3010215, 0.3695923, -0.2339092, 0.2977244, -0.5987459, 0.6035016
7: -0.2381915, 0.7898034, -0.2175102, 0.6964503, -0.9346418, 1.0073136
8: -0.1973222, 0.6440070, -0.1567580, 0.5266442, -0.7239664, 0.8007650
9: -0.3457508, 0.4190580, -0.2814359, 0.3530824, -0.6988332, 0.7004939

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A1_A2_B1_B1_B1_A1

### Relational analysis result of NS_A1_A2_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8096869, upper bound: 1.8471127
time: 4.26 seconds

## Relational analysis of NS_A1_A2_B1_B1_B1_A2

### Relational analysis result of NS_A1_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8272026, upper bound: 1.8549440
time: 3.11 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.2743394, 0.3454578, -0.2788675, 0.3499624, -0.6243017, 0.6243253
1: -0.3376172, 2.0394130, -0.3509177, 2.0593367, -2.3969538, 2.3903308
2: -0.2816013, 0.4809300, -0.2853840, 0.4833812, -0.7649825, 0.7663140
3: -0.2396128, 0.2819923, -0.2423455, 0.2868207, -0.5264335, 0.5243378
4: -0.3144519, 0.3550074, -0.3163882, 0.3595114, -0.6739634, 0.6713955
5: -0.3313779, 0.3889177, -0.3372528, 0.3928973, -0.7242752, 0.7261705
6: -0.2882534, 0.3543570, -0.2934355, 0.3593891, -0.6476426, 0.6477925
7: -0.2310441, 0.7737141, -0.2371885, 0.7703673, -1.0014114, 1.0109026
8: -0.1884928, 0.6235960, -0.1924376, 0.6338642, -0.8223569, 0.8160336
9: -0.3337461, 0.4058958, -0.3402025, 0.4192069, -0.7529529, 0.7460983

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 169

## Relational analysis of NS_A1_A2_B1_B2_A1_B1

### Relational analysis result of NS_A1_A2_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8297478, upper bound: 1.8284840
time: 4.04 seconds

## Relational analysis of NS_A1_A2_B1_B2_A1_B2

### Relational analysis result of NS_A1_A2_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8297478, upper bound: 1.8284840
time: 3.82 seconds

## BFS NS instance: NS_A1_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.2743394, 0.3454578, -0.2863440, 0.3599369, -0.6342763, 0.6318018
1: -0.3376172, 2.0394130, -0.3600385, 2.0699120, -2.4075291, 2.3994515
2: -0.2816013, 0.4809300, -0.2933874, 0.4908197, -0.7724210, 0.7743173
3: -0.2396128, 0.2819923, -0.2486206, 0.2922820, -0.5318949, 0.5306128
4: -0.3144519, 0.3550074, -0.3237067, 0.3699315, -0.6843835, 0.6787140
5: -0.3313779, 0.3889177, -0.3450019, 0.4011392, -0.7325171, 0.7339196
6: -0.2882534, 0.3543570, -0.3010215, 0.3695923, -0.6578457, 0.6553785
7: -0.2310441, 0.7737141, -0.2381915, 0.7898034, -1.0208476, 1.0119057
8: -0.1884928, 0.6235960, -0.1973222, 0.6440070, -0.8324997, 0.8209182
9: -0.3337461, 0.4058958, -0.3457508, 0.4190580, -0.7528040, 0.7516466

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 169

## Relational analysis of NS_A1_A2_B2_B1_A1_B1

### Relational analysis result of NS_A1_A2_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8315674, upper bound: 1.8323420
time: 3.81 seconds

## Relational analysis of NS_A1_A2_B2_B1_A1_B2

### Relational analysis result of NS_A1_A2_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8315674, upper bound: 1.8323420
time: 3.94 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.2743394, 0.3454578, -0.3347340, 0.4103626, -0.6847020, 0.6801918
1: -0.3376172, 2.0394130, -0.4598646, 2.0842080, -2.4218252, 2.4992776
2: -0.2816013, 0.4809300, -0.3392447, 0.5255044, -0.8071057, 0.8201746
3: -0.2396128, 0.2819923, -0.2835303, 0.3353857, -0.5749985, 0.5655226
4: -0.3144519, 0.3550074, -0.3595104, 0.4263583, -0.7408103, 0.7145178
5: -0.3313779, 0.3889177, -0.3959390, 0.4538332, -0.7852111, 0.7848567
6: -0.2882534, 0.3543570, -0.3493306, 0.4285127, -0.7167661, 0.7036875
7: -0.2310441, 0.7737141, -0.2732013, 0.8451506, -1.0761946, 1.0469155
8: -0.1884928, 0.6235960, -0.2365433, 0.7263331, -0.9148259, 0.8601393
9: -0.3337461, 0.4058958, -0.3955366, 0.4713076, -0.8050537, 0.8014324

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A1_A2_B2_B2_A1_B1

### Relational analysis result of NS_A1_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8533147, upper bound: 1.8247402
time: 3.82 seconds

## Relational analysis of NS_A1_A2_B2_B2_A1_B2

### Relational analysis result of NS_A1_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8533776, upper bound: 1.8246260
time: 3.98 seconds

## BFS NS instance: NS_A2_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.2833614, 0.3572115, -0.2345495, 0.2987274, -0.5820888, 0.5917611
1: -0.3367057, 2.0787497, -0.2514833, 2.0442259, -2.3809316, 2.3302331
2: -0.2860040, 0.4794227, -0.2386668, 0.4500791, -0.7360831, 0.7180895
3: -0.2361018, 0.2879304, -0.2072299, 0.2497623, -0.4858641, 0.4951604
4: -0.3126453, 0.3646084, -0.2806634, 0.3030124, -0.6156577, 0.6452718
5: -0.3361774, 0.3913888, -0.2832995, 0.3479608, -0.6841382, 0.6746883
6: -0.2957938, 0.3667828, -0.2439782, 0.3079137, -0.6037074, 0.6107610
7: -0.2365651, 0.7645527, -0.2231200, 0.7107163, -0.9472814, 0.9876727
8: -0.1922442, 0.6323004, -0.1619445, 0.5447927, -0.7370369, 0.7942449
9: -0.3377668, 0.4166251, -0.2917673, 0.3640208, -0.7017877, 0.7083924

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 207

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 169

## Relational analysis of NS_A2_A1_B1_B1_A2_B1

### Relational analysis result of NS_A2_A1_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8179846, upper bound: 1.8382609
time: 2.63 seconds

## Relational analysis of NS_A2_A1_B1_B1_A2_B2

### Relational analysis result of NS_A2_A1_B1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8180286, upper bound: 1.8122728
time: 5.30 seconds

## BFS NS instance: NS_A2_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.2559653, 0.3234926, -0.2788675, 0.3499624, -0.6059276, 0.6023601
1: -0.2875144, 2.0500207, -0.3509177, 2.0593367, -2.3468511, 2.4009385
2: -0.2589777, 0.4559059, -0.2853840, 0.4833812, -0.7423588, 0.7412899
3: -0.2160270, 0.2655704, -0.2423455, 0.2868207, -0.5028477, 0.5079160
4: -0.2904082, 0.3297734, -0.3163882, 0.3595114, -0.6499196, 0.6461616
5: -0.3069920, 0.3635661, -0.3372528, 0.3928973, -0.6998893, 0.7008189
6: -0.2672317, 0.3319142, -0.2934355, 0.3593891, -0.6266208, 0.6253497
7: -0.2283156, 0.7205522, -0.2371885, 0.7703673, -0.9986830, 0.9577407
8: -0.1723110, 0.5923766, -0.1924376, 0.6338642, -0.8061751, 0.7848142
9: -0.3122654, 0.3926581, -0.3402025, 0.4192069, -0.7314723, 0.7328606

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 169

## Relational analysis of NS_A2_A1_B1_B2_A1_A1

### Relational analysis result of NS_A2_A1_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8459996, upper bound: 1.8080753
time: 4.25 seconds

## Relational analysis of NS_A2_A1_B1_B2_A1_A2

### Relational analysis result of NS_A2_A1_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8213536, upper bound: 1.8083459
time: 2.47 seconds

## BFS NS instance: NS_A2_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.3091097, 0.3839549, -0.2788675, 0.3499624, -0.6590720, 0.6628224
1: -0.3937458, 2.0748887, -0.3509177, 2.0593367, -2.4530826, 2.4258065
2: -0.3110401, 0.4979089, -0.2853840, 0.4833812, -0.7944213, 0.7832929
3: -0.2568540, 0.3097642, -0.2423455, 0.2868207, -0.5436747, 0.5521097
4: -0.3319300, 0.3928573, -0.3163882, 0.3595114, -0.6914415, 0.7092454
5: -0.3656092, 0.4192871, -0.3372528, 0.3928973, -0.7585064, 0.7565399
6: -0.3218983, 0.3969350, -0.2934355, 0.3593891, -0.6812875, 0.6903706
7: -0.2499000, 0.7945080, -0.2371885, 0.7703673, -1.0202672, 1.0316964
8: -0.2120178, 0.6841972, -0.1924376, 0.6338642, -0.8458819, 0.8766348
9: -0.3647132, 0.4448199, -0.3402025, 0.4192069, -0.7839200, 0.7850224

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 169

## Relational analysis of NS_A2_A1_B1_B2_A2_B1

### Relational analysis result of NS_A2_A1_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8213217, upper bound: 1.8391199
time: 4.20 seconds

## Relational analysis of NS_A2_A1_B1_B2_A2_B2

### Relational analysis result of NS_A2_A1_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8213536, upper bound: 1.8091685
time: 3.04 seconds

## BFS NS instance: NS_A2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.3091097, 0.3839549, -0.2863440, 0.3599369, -0.6690466, 0.6702989
1: -0.3937458, 2.0748887, -0.3600385, 2.0699120, -2.4636579, 2.4349272
2: -0.3110401, 0.4979089, -0.2933874, 0.4908197, -0.8018599, 0.7912962
3: -0.2568540, 0.3097642, -0.2486206, 0.2922820, -0.5491360, 0.5583848
4: -0.3319300, 0.3928573, -0.3237067, 0.3699315, -0.7018616, 0.7165639
5: -0.3656092, 0.4192871, -0.3450019, 0.4011392, -0.7667484, 0.7642890
6: -0.3218983, 0.3969350, -0.3010215, 0.3695923, -0.6914907, 0.6979566
7: -0.2499000, 0.7945080, -0.2381915, 0.7898034, -1.0397034, 1.0326995
8: -0.2120178, 0.6841972, -0.1973222, 0.6440070, -0.8560247, 0.8815194
9: -0.3647132, 0.4448199, -0.3457508, 0.4190580, -0.7837712, 0.7905707

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 169

## Relational analysis of NS_A2_A1_B2_A2_B1_B1

### Relational analysis result of NS_A2_A1_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8213217, upper bound: 1.8390913
time: 3.22 seconds

## Relational analysis of NS_A2_A1_B2_A2_B1_B2

### Relational analysis result of NS_A2_A1_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8213536, upper bound: 1.8091685
time: 4.18 seconds

## BFS NS instance: NS_A2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.3091097, 0.3839549, -0.3347340, 0.4103626, -0.7194723, 0.7186890
1: -0.3937458, 2.0748887, -0.4598646, 2.0842080, -2.4779539, 2.5347533
2: -0.3110401, 0.4979089, -0.3392447, 0.5255044, -0.8365445, 0.8371536
3: -0.2568540, 0.3097642, -0.2835303, 0.3353857, -0.5922397, 0.5932945
4: -0.3319300, 0.3928573, -0.3595104, 0.4263583, -0.7582884, 0.7523677
5: -0.3656092, 0.4192871, -0.3959390, 0.4538332, -0.8194424, 0.8152261
6: -0.3218983, 0.3969350, -0.3493306, 0.4285127, -0.7504110, 0.7462656
7: -0.2499000, 0.7945080, -0.2732013, 0.8451506, -1.0950506, 1.0677093
8: -0.2120178, 0.6841972, -0.2365433, 0.7263331, -0.9383509, 0.9207405
9: -0.3647132, 0.4448199, -0.3955366, 0.4713076, -0.8360207, 0.8403565

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 169

## Relational analysis of NS_A2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8459996, upper bound: 1.8088867
time: 3.22 seconds

## Relational analysis of NS_A2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8213536, upper bound: 1.8091685
time: 4.21 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 9.61 seconds
NS_A1_A2_B1_B1_B1_A1, status: Status.VERIFIED, split count: 6, time: 9.61
Output dim: 1, lower bound: -1.8096869, upper bound: 1.8471127
NS_A1_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 9.61
Output dim: 1, lower bound: -1.8272026, upper bound: 1.8549440
NS_A1_A2_B1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 9.61
Output dim: 1, lower bound: -1.8297478, upper bound: 1.8284840
NS_A1_A2_B1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 9.61
Output dim: 1, lower bound: -1.8297478, upper bound: 1.8284840
NS_A1_A2_B2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 9.61
Output dim: 1, lower bound: -1.8315674, upper bound: 1.8323420
NS_A1_A2_B2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 9.61
Output dim: 1, lower bound: -1.8315674, upper bound: 1.8323420
NS_A1_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 9.61
Output dim: 1, lower bound: -1.8533147, upper bound: 1.8247402
NS_A1_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 9.61
Output dim: 1, lower bound: -1.8533776, upper bound: 1.8246260
NS_A2_A1_B1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 9.61
Output dim: 1, lower bound: -1.8179846, upper bound: 1.8382609
NS_A2_A1_B1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 9.61
Output dim: 1, lower bound: -1.8180286, upper bound: 1.8122728
NS_A2_A1_B1_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 9.61
Output dim: 1, lower bound: -1.8459996, upper bound: 1.8080753
NS_A2_A1_B1_B2_A1_A2, status: Status.VERIFIED, split count: 6, time: 9.61
Output dim: 1, lower bound: -1.8213536, upper bound: 1.8083459
NS_A2_A1_B1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 9.61
Output dim: 1, lower bound: -1.8213217, upper bound: 1.8391199
NS_A2_A1_B1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 9.61
Output dim: 1, lower bound: -1.8213536, upper bound: 1.8091685
NS_A2_A1_B2_A2_B1_B1, status: Status.VERIFIED, split count: 6, time: 9.61
Output dim: 1, lower bound: -1.8213217, upper bound: 1.8390913
NS_A2_A1_B2_A2_B1_B2, status: Status.VERIFIED, split count: 6, time: 9.61
Output dim: 1, lower bound: -1.8213536, upper bound: 1.8091685
NS_A2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 9.61
Output dim: 1, lower bound: -1.8459996, upper bound: 1.8088867
NS_A2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 6, time: 9.61
Output dim: 1, lower bound: -1.8213536, upper bound: 1.8091685

## BFS NS instance: NS_A1_A2_B1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.2477276, 0.3157164, -0.2289875, 0.2888660, -0.5365935, 0.5447039
1: -0.2788595, 2.0636199, -0.2296490, 2.0131452, -2.2920046, 2.2932689
2: -0.2557033, 0.4623371, -0.2286664, 0.4427662, -0.6984695, 0.6910034
3: -0.2178093, 0.2616812, -0.2001629, 0.2416678, -0.4594771, 0.4618441
4: -0.2947571, 0.3240920, -0.2726810, 0.2911071, -0.5858642, 0.5967730
5: -0.2999142, 0.3625269, -0.2719472, 0.3386850, -0.6385992, 0.6344742
6: -0.2596668, 0.3245145, -0.2339092, 0.2977244, -0.5573912, 0.5584238
7: -0.2270625, 0.7411070, -0.2175102, 0.6964503, -0.9235128, 0.9586172
8: -0.1696669, 0.5676554, -0.1567580, 0.5266442, -0.6963111, 0.7244134
9: -0.3065497, 0.3770373, -0.2814359, 0.3530824, -0.6596321, 0.6584732

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_A2_B1_B1_B1_A2_B1

### Relational analysis result of NS_A1_A2_B1_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8220181, upper bound: 1.8474740
time: 2.51 seconds

## Relational analysis of NS_A1_A2_B1_B1_B1_A2_B2

### Relational analysis result of NS_A1_A2_B1_B1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8221230, upper bound: 1.8490195
time: 4.31 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.2688738, 0.3396353, -0.2395950, 0.3093442, -0.5782181, 0.5792304
1: -0.3253543, 2.0382371, -0.2670380, 2.0620129, -2.3873672, 2.3052752
2: -0.2764641, 0.4770280, -0.2490477, 0.4581395, -0.7346036, 0.7260758
3: -0.2351254, 0.2776147, -0.2095835, 0.2558053, -0.4909307, 0.4871982
4: -0.3104705, 0.3491581, -0.2892172, 0.3198109, -0.6302814, 0.6383753
5: -0.3251117, 0.3835475, -0.2922370, 0.3541840, -0.6792957, 0.6757845
6: -0.2825113, 0.3477419, -0.2517996, 0.3130451, -0.5955564, 0.5995415
7: -0.2294885, 0.7679956, -0.2282651, 0.7394358, -0.9689243, 0.9962606
8: -0.1845758, 0.6112978, -0.1656549, 0.5357394, -0.7203152, 0.7769527
9: -0.3283108, 0.4006744, -0.3006588, 0.3813494, -0.7096601, 0.7013332

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 169

## Relational analysis of NS_A1_A2_B2_B2_A1_B1_B1

### Relational analysis result of NS_A1_A2_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8533147, upper bound: 1.8247402
time: 2.86 seconds

## Relational analysis of NS_A1_A2_B2_B2_A1_B1_B2

### Relational analysis result of NS_A1_A2_B2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8533147, upper bound: 1.8247402
time: 3.24 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.2743394, 0.3454578, -0.2730891, 0.3482320, -0.6225713, 0.6185470
1: -0.3376172, 2.0394130, -0.3357309, 2.0740514, -2.4116685, 2.3751440
2: -0.2816013, 0.4809300, -0.2820286, 0.4823935, -0.7639948, 0.7629586
3: -0.2396128, 0.2819923, -0.2362630, 0.2824572, -0.5220701, 0.5182552
4: -0.3144519, 0.3550074, -0.3143302, 0.3594315, -0.6738834, 0.6693375
5: -0.3313779, 0.3889177, -0.3305421, 0.3884024, -0.7197803, 0.7194598
6: -0.2882534, 0.3543570, -0.2875071, 0.3525463, -0.6407997, 0.6418641
7: -0.2310441, 0.7737141, -0.2375509, 0.7806599, -1.0117040, 1.0112650
8: -0.1884928, 0.6235960, -0.1893591, 0.6019715, -0.7904643, 0.8129550
9: -0.3337461, 0.4058958, -0.3342731, 0.4172696, -0.7510157, 0.7401689

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 169

## Relational analysis of NS_A1_A2_B2_B2_A1_B2_B1

### Relational analysis result of NS_A1_A2_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8533776, upper bound: 1.8246260
time: 2.27 seconds

## Relational analysis of NS_A1_A2_B2_B2_A1_B2_B2

### Relational analysis result of NS_A1_A2_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8533776, upper bound: 1.8246260
time: 3.80 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 8.49 seconds
NS_A1_A2_B1_B1_B1_A2_B1, status: Status.VERIFIED, split count: 7, time: 8.49
Output dim: 1, lower bound: -1.8220181, upper bound: 1.8474740
NS_A1_A2_B1_B1_B1_A2_B2, status: Status.VERIFIED, split count: 7, time: 8.49
Output dim: 1, lower bound: -1.8221230, upper bound: 1.8490195
NS_A1_A2_B2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 8.49
Output dim: 1, lower bound: -1.8533147, upper bound: 1.8247402
NS_A1_A2_B2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 8.49
Output dim: 1, lower bound: -1.8533147, upper bound: 1.8247402
NS_A1_A2_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 8.49
Output dim: 1, lower bound: -1.8533776, upper bound: 1.8246260
NS_A1_A2_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 8.49
Output dim: 1, lower bound: -1.8533776, upper bound: 1.8246260

## BFS NS instance: NS_A1_A2_B2_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.2688738, 0.3396353, -0.2320115, 0.2977540, -0.5666278, 0.5716468
1: -0.3253543, 2.0382371, -0.2440845, 2.0318348, -2.3571892, 2.2823217
2: -0.2764641, 0.4770280, -0.2377666, 0.4505844, -0.7270485, 0.7147946
3: -0.2351254, 0.2776147, -0.2020024, 0.2473127, -0.4824381, 0.4796170
4: -0.3104705, 0.3491581, -0.2807329, 0.3060212, -0.6164917, 0.6298910
5: -0.3251117, 0.3835475, -0.2800579, 0.3440230, -0.6691347, 0.6636054
6: -0.2825113, 0.3477419, -0.2411912, 0.3012152, -0.5837265, 0.5889331
7: -0.2294885, 0.7679956, -0.2224168, 0.7241230, -0.9536115, 0.9904124
8: -0.1845758, 0.6112978, -0.1595535, 0.5167934, -0.7013691, 0.7708513
9: -0.3283108, 0.4006744, -0.2897970, 0.3688746, -0.6971853, 0.6904714

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A1_A2_B2_B2_A1_B1_B1_A1

### Relational analysis result of NS_A1_A2_B2_B2_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8285864, upper bound: 1.8145097
time: 3.93 seconds

## Relational analysis of NS_A1_A2_B2_B2_A1_B1_B1_A2

### Relational analysis result of NS_A1_A2_B2_B2_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8490236, upper bound: 1.8205349
time: 4.43 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.2688738, 0.3396353, -0.3406301, 0.4185274, -0.6874013, 0.6802654
1: -0.3253543, 2.0382371, -0.4716518, 2.0438056, -2.3691599, 2.5098889
2: -0.2764641, 0.4770280, -0.3576542, 0.5223730, -0.7988371, 0.8346822
3: -0.2351254, 0.2776147, -0.2883637, 0.3322900, -0.5674154, 0.5659783
4: -0.3104705, 0.3491581, -0.3721008, 0.4515375, -0.7620080, 0.7212589
5: -0.3251117, 0.3835475, -0.4110095, 0.4517729, -0.7768846, 0.7945570
6: -0.2825113, 0.3477419, -0.3574242, 0.4264024, -0.7089138, 0.7051661
7: -0.2294885, 0.7679956, -0.2731546, 0.8802658, -1.1097543, 1.0411502
8: -0.1845758, 0.6112978, -0.2302363, 0.7484282, -0.9330039, 0.8415340
9: -0.3283108, 0.4006744, -0.4043155, 0.4857011, -0.8140119, 0.8049899

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 230

## Relational analysis of NS_A1_A2_B2_B2_A1_B1_B2_B1

### Relational analysis result of NS_A1_A2_B2_B2_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8427984, upper bound: 1.8159032
time: 2.75 seconds

## Relational analysis of NS_A1_A2_B2_B2_A1_B1_B2_B2

### Relational analysis result of NS_A1_A2_B2_B2_A1_B1_B2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8427967, upper bound: 1.8149300
time: 3.33 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.2743394, 0.3454578, -0.2614048, 0.3339730, -0.6083124, 0.6068626
1: -0.3376172, 2.0394130, -0.3109456, 2.0432010, -2.3808181, 2.3503585
2: -0.2816013, 0.4809300, -0.2704639, 0.4728895, -0.7544909, 0.7513939
3: -0.2396128, 0.2819923, -0.2275296, 0.2723715, -0.5119843, 0.5095218
4: -0.3144519, 0.3550074, -0.3053113, 0.3448602, -0.6593121, 0.6603186
5: -0.3313779, 0.3889177, -0.3171708, 0.3764806, -0.7078585, 0.7060885
6: -0.2882534, 0.3543570, -0.2750131, 0.3377274, -0.6259809, 0.6293701
7: -0.2310441, 0.7737141, -0.2310973, 0.7648563, -0.9959004, 1.0048114
8: -0.1884928, 0.6235960, -0.1808557, 0.5821545, -0.7706473, 0.8044516
9: -0.3337461, 0.4058958, -0.3226164, 0.4042610, -0.7380071, 0.7285122

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A1_A2_B2_B2_A1_B2_B1_A1

### Relational analysis result of NS_A1_A2_B2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8529344, upper bound: 1.8244801
time: 4.85 seconds

## Relational analysis of NS_A1_A2_B2_B2_A1_B2_B1_A2

### Relational analysis result of NS_A1_A2_B2_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8529344, upper bound: 1.8246260
time: 3.32 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.2743394, 0.3454578, -0.3918756, 0.4850598, -0.7593992, 0.7373334
1: -0.3376172, 2.0394130, -0.5542024, 2.0577230, -2.3953402, 2.5936155
2: -0.2816013, 0.4809300, -0.3916783, 0.5668897, -0.8484910, 0.8726082
3: -0.2396128, 0.2819923, -0.3254203, 0.3723779, -0.6119907, 0.6074126
4: -0.3144519, 0.3550074, -0.4003388, 0.4965857, -0.8110376, 0.7553462
5: -0.3313779, 0.3889177, -0.4609157, 0.5014303, -0.8328081, 0.8498334
6: -0.2882534, 0.3543570, -0.4115282, 0.4927339, -0.7809873, 0.7658852
7: -0.2310441, 0.7737141, -0.3191423, 0.9224475, -1.1534916, 1.0928564
8: -0.1884928, 0.6235960, -0.2765173, 0.8279120, -1.0164047, 0.9001133
9: -0.3337461, 0.4058958, -0.4452705, 0.5255251, -0.8592712, 0.8511663

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 90

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 230

## Relational analysis of NS_A1_A2_B2_B2_A1_B2_B2_B1

### Relational analysis result of NS_A1_A2_B2_B2_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8429027, upper bound: 1.8157682
time: 4.48 seconds

## Relational analysis of NS_A1_A2_B2_B2_A1_B2_B2_B2

### Relational analysis result of NS_A1_A2_B2_B2_A1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8429027, upper bound: 1.8147541
time: 3.94 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 10.57 seconds
NS_A1_A2_B2_B2_A1_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 10.57
Output dim: 1, lower bound: -1.8285864, upper bound: 1.8145097
NS_A1_A2_B2_B2_A1_B1_B1_A2, status: Status.VERIFIED, split count: 8, time: 10.57
Output dim: 1, lower bound: -1.8490236, upper bound: 1.8205349
NS_A1_A2_B2_B2_A1_B1_B2_B1, status: Status.VERIFIED, split count: 8, time: 10.57
Output dim: 1, lower bound: -1.8427984, upper bound: 1.8159032
NS_A1_A2_B2_B2_A1_B1_B2_B2, status: Status.VERIFIED, split count: 8, time: 10.57
Output dim: 1, lower bound: -1.8427967, upper bound: 1.8149300
NS_A1_A2_B2_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 10.57
Output dim: 1, lower bound: -1.8529344, upper bound: 1.8244801
NS_A1_A2_B2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 10.57
Output dim: 1, lower bound: -1.8529344, upper bound: 1.8246260
NS_A1_A2_B2_B2_A1_B2_B2_B1, status: Status.VERIFIED, split count: 8, time: 10.57
Output dim: 1, lower bound: -1.8429027, upper bound: 1.8157682
NS_A1_A2_B2_B2_A1_B2_B2_B2, status: Status.VERIFIED, split count: 8, time: 10.57
Output dim: 1, lower bound: -1.8429027, upper bound: 1.8147541

## BFS NS instance: NS_A1_A2_B2_B2_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.2194133, 0.2603803, -0.2614048, 0.3339730, -0.5533863, 0.5217851
1: -0.1555623, 2.0184536, -0.3109456, 2.0432010, -2.1987634, 2.3293991
2: -0.1996250, 0.4243026, -0.2704639, 0.4728895, -0.6725145, 0.6947665
3: -0.1730586, 0.2175959, -0.2275296, 0.2723715, -0.4454300, 0.4451255
4: -0.2514595, 0.2597812, -0.3053113, 0.3448602, -0.5963197, 0.5650924
5: -0.2336675, 0.3071090, -0.3171708, 0.3764806, -0.6101481, 0.6242799
6: -0.2019998, 0.2631638, -0.2750131, 0.3377274, -0.5397272, 0.5381769
7: -0.2129424, 0.6720287, -0.2310973, 0.7648563, -0.9777987, 0.9031260
8: -0.1380547, 0.4390651, -0.1808557, 0.5821545, -0.7202091, 0.6199208
9: -0.2487434, 0.3218620, -0.3226164, 0.4042610, -0.6530044, 0.6444784

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 207

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_A2_B2_B2_A1_B2_B1_A1_B1

### Relational analysis result of NS_A1_A2_B2_B2_A1_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8474742, upper bound: 1.8323805
time: 3.87 seconds

## Relational analysis of NS_A1_A2_B2_B2_A1_B2_B1_A1_B2

### Relational analysis result of NS_A1_A2_B2_B2_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8536557, upper bound: 1.8544822
time: 3.93 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.2282053, 0.2883920, -0.2614048, 0.3339730, -0.5621783, 0.5497968
1: -0.2189944, 2.0298734, -0.3109456, 2.0432010, -2.2621953, 2.3408189
2: -0.2275582, 0.4433545, -0.2704639, 0.4728895, -0.7004477, 0.7138184
3: -0.1951899, 0.2392124, -0.2275296, 0.2723715, -0.4675614, 0.4667419
4: -0.2728930, 0.2937274, -0.3053113, 0.3448602, -0.6177532, 0.5990386
5: -0.2684753, 0.3350564, -0.3171708, 0.3764806, -0.6449559, 0.6522272
6: -0.2312279, 0.2925467, -0.2750131, 0.3377274, -0.5689553, 0.5675598
7: -0.2193916, 0.7095237, -0.2310973, 0.7648563, -0.9842479, 0.9406210
8: -0.1531723, 0.5015634, -0.1808557, 0.5821545, -0.7353267, 0.6824191
9: -0.2780488, 0.3533044, -0.3226164, 0.4042610, -0.6823097, 0.6759207

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_A2_B2_B2_A1_B2_B1_A2_B1

### Relational analysis result of NS_A1_A2_B2_B2_A1_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8474742, upper bound: 1.8325280
time: 4.35 seconds

## Relational analysis of NS_A1_A2_B2_B2_A1_B2_B1_A2_B2

### Relational analysis result of NS_A1_A2_B2_B2_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.8536557, upper bound: 1.8547582
time: 5.69 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 12.19 seconds
NS_A1_A2_B2_B2_A1_B2_B1_A1_B1, status: Status.VERIFIED, split count: 9, time: 12.19
Output dim: 1, lower bound: -1.8474742, upper bound: 1.8323805
NS_A1_A2_B2_B2_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 12.19
Output dim: 1, lower bound: -1.8536557, upper bound: 1.8544822
NS_A1_A2_B2_B2_A1_B2_B1_A2_B1, status: Status.VERIFIED, split count: 9, time: 12.19
Output dim: 1, lower bound: -1.8474742, upper bound: 1.8325280
NS_A1_A2_B2_B2_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 12.19
Output dim: 1, lower bound: -1.8536557, upper bound: 1.8547582

## BFS NS instance: NS_A1_A2_B2_B2_A1_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.2194133, 0.2603803, -0.2318245, 0.2938738, -0.5132872, 0.4922047
1: -0.1555623, 2.0184536, -0.2343427, 2.0372081, -2.1927705, 2.2527962
2: -0.1996250, 0.4243026, -0.2336419, 0.4481591, -0.6477841, 0.6579445
3: -0.1730586, 0.2175959, -0.1993651, 0.2444750, -0.4175336, 0.4169610
4: -0.2514595, 0.2597812, -0.2777346, 0.3007545, -0.5522140, 0.5375159
5: -0.2336675, 0.3071090, -0.2750616, 0.3405618, -0.5742294, 0.5821706
6: -0.2019998, 0.2631638, -0.2372145, 0.2982118, -0.5002116, 0.5003783
7: -0.2129424, 0.6720287, -0.2222420, 0.7172873, -0.9302297, 0.8942707
8: -0.1380547, 0.4390651, -0.1575508, 0.5106108, -0.6486655, 0.5966159
9: -0.2487434, 0.3218620, -0.2854424, 0.3635009, -0.6122444, 0.6073043

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_A2_B2_B2_A1_B2_B1_A1_B2_B1

### Relational analysis result of NS_A1_A2_B2_B2_A1_B2_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8479139, upper bound: 1.8477027
time: 3.51 seconds

## Relational analysis of NS_A1_A2_B2_B2_A1_B2_B1_A1_B2_B2

### Relational analysis result of NS_A1_A2_B2_B2_A1_B2_B1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8479139, upper bound: 1.8486519
time: 4.78 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A1_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.2282053, 0.2883920, -0.2318245, 0.2938738, -0.5220791, 0.5202165
1: -0.2189944, 2.0298734, -0.2343427, 2.0372081, -2.2562025, 2.2642159
2: -0.2275582, 0.4433545, -0.2336419, 0.4481591, -0.6757173, 0.6769964
3: -0.1951899, 0.2392124, -0.1993651, 0.2444750, -0.4396649, 0.4385775
4: -0.2728930, 0.2937274, -0.2777346, 0.3007545, -0.5736475, 0.5714620
5: -0.2684753, 0.3350564, -0.2750616, 0.3405618, -0.6090372, 0.6101180
6: -0.2312279, 0.2925467, -0.2372145, 0.2982118, -0.5294397, 0.5297612
7: -0.2193916, 0.7095237, -0.2222420, 0.7172873, -0.9366789, 0.9317657
8: -0.1531723, 0.5015634, -0.1575508, 0.5106108, -0.6637831, 0.6591142
9: -0.2780488, 0.3533044, -0.2854424, 0.3635009, -0.6415497, 0.6387467

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_A2_B2_B2_A1_B2_B1_A2_B2_B1

### Relational analysis result of NS_A1_A2_B2_B2_A1_B2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8482259, upper bound: 1.8478799
time: 4.05 seconds

## Relational analysis of NS_A1_A2_B2_B2_A1_B2_B1_A2_B2_B2

### Relational analysis result of NS_A1_A2_B2_B2_A1_B2_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8482259, upper bound: 1.8488860
time: 4.94 seconds

## Summary of splitting at layer (split count: 9)
- Time for NS candidates: 11.18 seconds
NS_A1_A2_B2_B2_A1_B2_B1_A1_B2_B1, status: Status.VERIFIED, split count: 10, time: 11.18
Output dim: 1, lower bound: -1.8479139, upper bound: 1.8477027
NS_A1_A2_B2_B2_A1_B2_B1_A1_B2_B2, status: Status.VERIFIED, split count: 10, time: 11.18
Output dim: 1, lower bound: -1.8479139, upper bound: 1.8486519
NS_A1_A2_B2_B2_A1_B2_B1_A2_B2_B1, status: Status.VERIFIED, split count: 10, time: 11.18
Output dim: 1, lower bound: -1.8482259, upper bound: 1.8478799
NS_A1_A2_B2_B2_A1_B2_B1_A2_B2_B2, status: Status.VERIFIED, split count: 10, time: 11.18
Output dim: 1, lower bound: -1.8482259, upper bound: 1.8488860

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 8.78 + 422.30 = 431.08 seconds
