## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.542703008


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.7778639, 0.6806120, -0.7778639, 0.6806120, -1.4584759, 1.4584759)
1: (-0.6040506, 0.8626654, -0.6040506, 0.8626654, -1.4667161, 1.4667161)
2: (-0.5841001, 0.8418483, -0.5841001, 0.8418483, -1.4259484, 1.4259484)
3: (-0.6007968, 0.6236554, -0.6007968, 0.6236554, -1.2244523, 1.2244523)
4: (-0.7511072, 0.7287243, -0.7511072, 0.7287243, -1.4798315, 1.4798315)
5: (-0.5852865, 1.1952124, -0.5852865, 1.1952124, -1.7804989, 1.7804989)
6: (-0.4886691, 0.6989943, -0.4886691, 0.6989943, -1.1876633, 1.1876633)
7: (-0.6124615, 0.7924250, -0.6124615, 0.7924250, -1.4048865, 1.4048865)
8: (-0.6664105, 0.8338409, -0.6664105, 0.8338409, -1.5002514, 1.5002514)
9: (-0.6932293, 0.7872544, -0.6932293, 0.7872544, -1.4804838, 1.4804838)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.84 + 3.06 = 4.90 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -1.6069825, upper bound: 1.6069825

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6066799, upper bound: 1.6023961
time: 2.09 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6023961, upper bound: 1.6023961
time: 1.65 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 3.93 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 3.93
Output dim: 5, lower bound: -1.6066799, upper bound: 1.6023961
NS_A2, status: Status.UNKNOWN, split count: 1, time: 3.93
Output dim: 5, lower bound: -1.6023961, upper bound: 1.6023961

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.6882363, 0.6364546, -0.7778639, 0.6806120, -1.3688483, 1.4143186
1: -0.5506119, 0.8005905, -0.6040506, 0.8626654, -1.4132773, 1.4046412
2: -0.5095878, 0.7837211, -0.5841001, 0.8418483, -1.3514361, 1.3678212
3: -0.5005569, 0.5906971, -0.6007968, 0.6236554, -1.1242123, 1.1914939
4: -0.6751141, 0.6620120, -0.7511072, 0.7287243, -1.4038384, 1.4131192
5: -0.4788068, 1.1766840, -0.5852865, 1.1952124, -1.6740191, 1.7619705
6: -0.4297588, 0.6428789, -0.4886691, 0.6989943, -1.1287531, 1.1315480
7: -0.5517590, 0.7234537, -0.6124615, 0.7924250, -1.3441839, 1.3359152
8: -0.5831960, 0.7906954, -0.6664105, 0.8338409, -1.4170370, 1.4571059
9: -0.6288399, 0.7287579, -0.6932293, 0.7872544, -1.4160943, 1.4219873

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6023961, upper bound: 1.6023961
time: 1.74 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6023961, upper bound: 1.6023961
time: 1.75 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -1.0619879, 0.8189312, -0.7629215, 0.6732261, -1.7352140, 1.5818527
1: -0.7763894, 1.0458699, -0.5950864, 0.8517139, -1.6281033, 1.6409564
2: -0.8216664, 1.0350451, -0.5716135, 0.8317087, -1.6533751, 1.6066587
3: -0.9358491, 0.7244021, -0.5833046, 0.6181713, -1.5540204, 1.3077067
4: -1.0026883, 0.9399933, -0.7379612, 0.7176109, -1.7202992, 1.6779544
5: -0.8877734, 1.1884665, -0.5678178, 1.1933068, -2.0810802, 1.7562844
6: -0.6769806, 0.8909385, -0.4787343, 0.6890033, -1.3659840, 1.3696728
7: -0.8049077, 1.0157026, -0.6022778, 0.7806687, -1.5855763, 1.6179804
8: -0.9336596, 0.9811420, -0.6524577, 0.8260036, -1.7596631, 1.6335998
9: -0.9115970, 0.9781733, -0.6817931, 0.7771913, -1.6887884, 1.6599663

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6023961, upper bound: 1.6023961
time: 1.84 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6023961, upper bound: 1.6023961
time: 1.76 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.68 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.68
Output dim: 5, lower bound: -1.6023961, upper bound: 1.6023961
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.68
Output dim: 5, lower bound: -1.6023961, upper bound: 1.6023961
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.68
Output dim: 5, lower bound: -1.6023961, upper bound: 1.6023961
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 5.68
Output dim: 5, lower bound: -1.6023961, upper bound: 1.6023961

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.6882363, 0.6364546, -0.6882363, 0.6364546, -1.3246909, 1.3246909
1: -0.5506119, 0.8005905, -0.5506119, 0.8005905, -1.3512024, 1.3512024
2: -0.5095878, 0.7837211, -0.5095878, 0.7837211, -1.2933090, 1.2933090
3: -0.5005569, 0.5906971, -0.5005569, 0.5906971, -1.0912540, 1.0912540
4: -0.6751141, 0.6620120, -0.6751141, 0.6620120, -1.3371260, 1.3371260
5: -0.4788068, 1.1766840, -0.4788068, 1.1766840, -1.6554909, 1.6554909
6: -0.4297588, 0.6428789, -0.4297588, 0.6428789, -1.0726378, 1.0726378
7: -0.5517590, 0.7234537, -0.5517590, 0.7234537, -1.2752128, 1.2752128
8: -0.5831960, 0.7906954, -0.5831960, 0.7906954, -1.3738915, 1.3738915
9: -0.6288399, 0.7287579, -0.6288399, 0.7287579, -1.3575978, 1.3575978

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5895121, upper bound: 1.5352767
time: 1.97 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5428148, upper bound: 1.5352767
time: 1.56 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.6882363, 0.6364546, -1.0619879, 0.8189312, -1.5071676, 1.6984425
1: -0.5506119, 0.8005905, -0.7763894, 1.0458699, -1.5964818, 1.5769799
2: -0.5095878, 0.7837211, -0.8216664, 1.0350451, -1.5446329, 1.6053876
3: -0.5005569, 0.5906971, -0.9358491, 0.7244021, -1.2249590, 1.5265461
4: -0.6751141, 0.6620120, -1.0026883, 0.9399933, -1.6151073, 1.6647003
5: -0.4788068, 1.1766840, -0.8877734, 1.1884665, -1.6672733, 2.0644574
6: -0.4297588, 0.6428789, -0.6769806, 0.8909385, -1.3206973, 1.3198595
7: -0.5517590, 0.7234537, -0.8049077, 1.0157026, -1.5674616, 1.5283613
8: -0.5831960, 0.7906954, -0.9336596, 0.9811420, -1.5643381, 1.7243550
9: -0.6288399, 0.7287579, -0.9115970, 0.9781733, -1.6070132, 1.6403549

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5895121, upper bound: 1.5352767
time: 1.97 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5428148, upper bound: 1.5352767
time: 1.50 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -1.0619879, 0.8189312, -0.6882363, 0.6364546, -1.6984425, 1.5071676
1: -0.7763894, 1.0458699, -0.5506119, 0.8005905, -1.5769799, 1.5964818
2: -0.8216664, 1.0350451, -0.5095878, 0.7837211, -1.6053876, 1.5446329
3: -0.9358491, 0.7244021, -0.5005569, 0.5906971, -1.5265461, 1.2249590
4: -1.0026883, 0.9399933, -0.6751141, 0.6620120, -1.6647003, 1.6151073
5: -0.8877734, 1.1884665, -0.4788068, 1.1766840, -2.0644574, 1.6672733
6: -0.6769806, 0.8909385, -0.4297588, 0.6428789, -1.3198595, 1.3206973
7: -0.8049077, 1.0157026, -0.5517590, 0.7234537, -1.5283613, 1.5674616
8: -0.9336596, 0.9811420, -0.5831960, 0.7906954, -1.7243550, 1.5643381
9: -0.9115970, 0.9781733, -0.6288399, 0.7287579, -1.6403549, 1.6070132

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5352767, upper bound: 1.5848828
time: 1.76 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5352767, upper bound: 1.5352767
time: 1.54 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -1.0619879, 0.8189312, -1.0619879, 0.8189312, -1.8809191, 1.8809191
1: -0.7763894, 1.0458699, -0.7763894, 1.0458699, -1.8222593, 1.8222593
2: -0.8216664, 1.0350451, -0.8216664, 1.0350451, -1.8567116, 1.8567116
3: -0.9358491, 0.7244021, -0.9358491, 0.7244021, -1.6602511, 1.6602511
4: -1.0026883, 0.9399933, -1.0026883, 0.9399933, -1.9426816, 1.9426816
5: -0.8877734, 1.1884665, -0.8877734, 1.1884665, -2.0762401, 2.0762401
6: -0.6769806, 0.8909385, -0.6769806, 0.8909385, -1.5679191, 1.5679191
7: -0.8049077, 1.0157026, -0.8049077, 1.0157026, -1.8206103, 1.8206103
8: -0.9336596, 0.9811420, -0.9336596, 0.9811420, -1.9148016, 1.9148016
9: -0.9115970, 0.9781733, -0.9115970, 0.9781733, -1.8897703, 1.8897703

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5837636, upper bound: 1.5352767
time: 2.15 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5352767, upper bound: 1.5352767
time: 1.75 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 5.67 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.67
Output dim: 5, lower bound: -1.5895121, upper bound: 1.5352767
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.67
Output dim: 5, lower bound: -1.5428148, upper bound: 1.5352767
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.67
Output dim: 5, lower bound: -1.5895121, upper bound: 1.5352767
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.67
Output dim: 5, lower bound: -1.5428148, upper bound: 1.5352767
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 5.67
Output dim: 5, lower bound: -1.5352767, upper bound: 1.5848828
NS_A2_B1_B2, status: Status.VERIFIED, split count: 3, time: 5.67
Output dim: 5, lower bound: -1.5352767, upper bound: 1.5352767
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.67
Output dim: 5, lower bound: -1.5837636, upper bound: 1.5352767
NS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 5.67
Output dim: 5, lower bound: -1.5352767, upper bound: 1.5352767

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.6553158, 0.6195064, -0.6882363, 0.6364546, -1.2917705, 1.3077426
1: -0.5319067, 0.7554154, -0.5506119, 0.8005905, -1.3324972, 1.3060273
2: -0.4823819, 0.7617587, -0.5095878, 0.7837211, -1.2661030, 1.2713466
3: -0.4634900, 0.5777656, -0.5005569, 0.5906971, -1.0541871, 1.0783224
4: -0.6474417, 0.6361953, -0.6751141, 0.6620120, -1.3094537, 1.3113093
5: -0.4201757, 1.1714723, -0.4788068, 1.1766840, -1.5968597, 1.6502790
6: -0.4081064, 0.6232394, -0.4297588, 0.6428789, -1.0509853, 1.0529982
7: -0.5285623, 0.6983957, -0.5517590, 0.7234537, -1.2520161, 1.2501547
8: -0.5539551, 0.7737538, -0.5831960, 0.7906954, -1.3446505, 1.3569498
9: -0.6065691, 0.7064880, -0.6288399, 0.7287579, -1.3353270, 1.3353279

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5753261, upper bound: 1.5412288
time: 1.86 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5701932, upper bound: 1.5046753
time: 1.91 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.9186115, 0.7475885, -0.6692773, 0.6266999, -1.5453115, 1.4168658
1: -0.6872724, 0.8280116, -0.5398749, 0.7728164, -1.4600887, 1.3678865
2: -0.6918992, 0.9369161, -0.4938508, 0.7711684, -1.4630675, 1.4307669
3: -0.7616947, 0.6611317, -0.4792872, 0.5831771, -1.3448718, 1.1404189
4: -0.8684975, 0.8358751, -0.6592773, 0.6470934, -1.5155909, 1.4951525
5: -0.6333476, 1.1627884, -0.4432620, 1.1735656, -1.8069133, 1.6060505
6: -0.5712624, 0.7853311, -0.4173097, 0.6316651, -1.2029275, 1.2026408
7: -0.6988440, 0.8966272, -0.5382973, 0.7091052, -1.4079492, 1.4349245
8: -0.7936546, 0.8970044, -0.5661137, 0.7809359, -1.5745904, 1.4631181
9: -0.7847473, 0.8750842, -0.6157933, 0.7159690, -1.5007163, 1.4908775

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5046753, upper bound: 1.5331935
time: 1.43 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5046753, upper bound: 1.5046753
time: 1.45 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.6553158, 0.6195064, -1.0619879, 0.8189312, -1.4742470, 1.6814942
1: -0.5319067, 0.7554154, -0.7763894, 1.0458699, -1.5777767, 1.5318048
2: -0.4823819, 0.7617587, -0.8216664, 1.0350451, -1.5174271, 1.5834252
3: -0.4634900, 0.5777656, -0.9358491, 0.7244021, -1.1878922, 1.5136147
4: -0.6474417, 0.6361953, -1.0026883, 0.9399933, -1.5874350, 1.6388836
5: -0.4201757, 1.1714723, -0.8877734, 1.1884665, -1.6086422, 2.0592456
6: -0.4081064, 0.6232394, -0.6769806, 0.8909385, -1.2990448, 1.3002200
7: -0.5285623, 0.6983957, -0.8049077, 1.0157026, -1.5442649, 1.5033033
8: -0.5539551, 0.7737538, -0.9336596, 0.9811420, -1.5350971, 1.7074133
9: -0.6065691, 0.7064880, -0.9115970, 0.9781733, -1.5847423, 1.6180849

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_A1_A1

### Relational analysis result of NS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5757316, upper bound: 1.5006464
time: 1.57 seconds

## Relational analysis of NS_A1_B2_A1_A2

### Relational analysis result of NS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5688276, upper bound: 1.5006464
time: 1.78 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.9186115, 0.7475885, -1.0418146, 0.8086598, -1.7272713, 1.7894030
1: -0.6872724, 0.8280116, -0.7644976, 1.0170965, -1.7043689, 1.5925093
2: -0.6918992, 0.9369161, -0.8043690, 1.0213823, -1.7132815, 1.7412851
3: -0.7616947, 0.6611317, -0.9122488, 0.7161483, -1.4778429, 1.5733805
4: -0.8684975, 0.8358751, -0.9849292, 0.9246237, -1.7931212, 1.8208044
5: -0.6333476, 1.1627884, -0.8528008, 1.1849663, -1.8183140, 2.0155892
6: -0.5712624, 0.7853311, -0.6631462, 0.8776724, -1.4489348, 1.4484773
7: -0.6988440, 0.8966272, -0.7904333, 0.9997826, -1.6986265, 1.6870605
8: -0.7936546, 0.8970044, -0.9148670, 0.9700768, -1.7637315, 1.8118714
9: -0.7847473, 0.8750842, -0.8960054, 0.9642000, -1.7489474, 1.7710896

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5046753, upper bound: 1.5275035
time: 1.48 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5046753, upper bound: 1.5006464
time: 1.40 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -1.0619879, 0.8189312, -0.6553158, 0.6195064, -1.6814942, 1.4742470
1: -0.7763894, 1.0458699, -0.5319067, 0.7554154, -1.5318048, 1.5777767
2: -0.8216664, 1.0350451, -0.4823819, 0.7617587, -1.5834252, 1.5174271
3: -0.9358491, 0.7244021, -0.4634900, 0.5777656, -1.5136147, 1.1878922
4: -1.0026883, 0.9399933, -0.6474417, 0.6361953, -1.6388836, 1.5874350
5: -0.8877734, 1.1884665, -0.4201757, 1.1714723, -2.0592456, 1.6086422
6: -0.6769806, 0.8909385, -0.4081064, 0.6232394, -1.3002200, 1.2990448
7: -0.8049077, 1.0157026, -0.5285623, 0.6983957, -1.5033033, 1.5442649
8: -0.9336596, 0.9811420, -0.5539551, 0.7737538, -1.7074133, 1.5350971
9: -0.9115970, 0.9781733, -0.6065691, 0.7064880, -1.6180849, 1.5847423

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_B1_B1_B1

### Relational analysis result of NS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5006464, upper bound: 1.5757316
time: 1.57 seconds

## Relational analysis of NS_A2_B1_B1_B2

### Relational analysis result of NS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5006464, upper bound: 1.5688276
time: 1.81 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1.0264742, 0.8009382, -1.0619879, 0.8189312, -1.8454055, 1.8629261
1: -0.7553862, 0.9988184, -0.7763894, 1.0458699, -1.8012562, 1.7752078
2: -0.7913131, 1.0109805, -0.8216664, 1.0350451, -1.8263582, 1.8326468
3: -0.8942685, 0.7101072, -0.9358491, 0.7244021, -1.6186707, 1.6459563
4: -0.9714069, 0.9130243, -1.0026883, 0.9399933, -1.9114002, 1.9157126
5: -0.8291278, 1.1826258, -0.8877734, 1.1884665, -2.0175943, 2.0703993
6: -0.6527300, 0.8675123, -0.6769806, 0.8909385, -1.5436685, 1.5444930
7: -0.7796060, 0.9876889, -0.8049077, 1.0157026, -1.7953086, 1.7925966
8: -0.9005474, 0.9617878, -0.9336596, 0.9811420, -1.8816894, 1.8954474
9: -0.8841767, 0.9536622, -0.9115970, 0.9781733, -1.8623500, 1.8652592

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5654860, upper bound: 1.5345388
time: 1.65 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5607533, upper bound: 1.5006464
time: 1.70 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 5.29 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.29
Output dim: 5, lower bound: -1.5753261, upper bound: 1.5412288
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.29
Output dim: 5, lower bound: -1.5701932, upper bound: 1.5046753
NS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 5.29
Output dim: 5, lower bound: -1.5046753, upper bound: 1.5331935
NS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 5.29
Output dim: 5, lower bound: -1.5046753, upper bound: 1.5046753
NS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 5.29
Output dim: 5, lower bound: -1.5757316, upper bound: 1.5006464
NS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 5.29
Output dim: 5, lower bound: -1.5688276, upper bound: 1.5006464
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 5.29
Output dim: 5, lower bound: -1.5046753, upper bound: 1.5275035
NS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 5.29
Output dim: 5, lower bound: -1.5046753, upper bound: 1.5006464
NS_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 5.29
Output dim: 5, lower bound: -1.5006464, upper bound: 1.5757316
NS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 5.29
Output dim: 5, lower bound: -1.5006464, upper bound: 1.5688276
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.29
Output dim: 5, lower bound: -1.5654860, upper bound: 1.5345388
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.29
Output dim: 5, lower bound: -1.5607533, upper bound: 1.5006464

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.6553158, 0.6195064, -0.6337503, 0.6084861, -1.2638018, 1.2532567
1: -0.5319067, 0.7554154, -0.5190299, 0.7488049, -1.2807117, 1.2744453
2: -0.4823819, 0.7617587, -0.4657184, 0.7468294, -1.2292113, 1.2274772
3: -0.4634900, 0.5777656, -0.4402518, 0.5709973, -1.0344872, 1.0180174
4: -0.6474417, 0.6361953, -0.6303590, 0.6194844, -1.2669261, 1.2665544
5: -0.4201757, 1.1714723, -0.4021270, 1.1689751, -1.5891508, 1.5735993
6: -0.4081064, 0.6232394, -0.3941028, 0.6106063, -1.0187126, 1.0173422
7: -0.5285623, 0.6983957, -0.5140792, 0.6816821, -1.2102444, 1.2124748
8: -0.5539551, 0.7737538, -0.5348539, 0.7633204, -1.3172755, 1.3086076
9: -0.6065691, 0.7064880, -0.5923197, 0.6924674, -1.2990365, 1.2988076

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5687144, upper bound: 1.5231830
time: 2.00 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5687144, upper bound: 1.5366049
time: 1.70 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.6427402, 0.6129229, -0.9714550, 0.7755836, -1.4183239, 1.5843779
1: -0.5245499, 0.7430671, -0.7169251, 0.9300792, -1.4546292, 1.4599922
2: -0.4723951, 0.7530687, -0.7369878, 0.9719032, -1.4442983, 1.4900565
3: -0.4498440, 0.5732920, -0.8214043, 0.6838393, -1.1336833, 1.3946964
4: -0.6374068, 0.6262825, -0.9129306, 0.8782474, -1.5156541, 1.5392131
5: -0.4021720, 1.1694989, -0.7507657, 1.1708025, -1.5729744, 1.9202646
6: -0.3997239, 0.6159372, -0.6069249, 0.8163485, -1.2160724, 1.2228621
7: -0.5197313, 0.6886275, -0.7379839, 0.9368770, -1.4566083, 1.4266114
8: -0.5428215, 0.7673848, -0.8414077, 0.9253340, -1.4681555, 1.6087925
9: -0.5981230, 0.6980876, -0.8216796, 0.9116837, -1.5098066, 1.5197673

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5635445, upper bound: 1.4991882
time: 1.95 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5635445, upper bound: 1.5046753
time: 2.10 seconds

## BFS NS instance: NS_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.6017754, 0.5910770, -1.0619879, 0.8189312, -1.4207066, 1.6530650
1: -0.5006989, 0.7040693, -0.7763894, 1.0458699, -1.5465689, 1.4804586
2: -0.4399196, 0.7246044, -0.8216664, 1.0350451, -1.4749647, 1.5462708
3: -0.4064578, 0.5587549, -0.9358491, 0.7244021, -1.1308599, 1.4946039
4: -0.6050895, 0.5938340, -1.0026883, 0.9399933, -1.5450828, 1.5965223
5: -0.3442929, 1.1638973, -0.8877734, 1.1884665, -1.5327594, 2.0516706
6: -0.3719376, 0.5924852, -0.6769806, 0.8909385, -1.2628760, 1.2694658
7: -0.4910856, 0.6563345, -0.8049077, 1.0157026, -1.5067883, 1.4612422
8: -0.5068572, 0.7465459, -0.9336596, 0.9811420, -1.4879992, 1.6802055
9: -0.5710402, 0.6704406, -0.9115970, 0.9781733, -1.5492134, 1.5820376

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_A1_A1_B1

### Relational analysis result of NS_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5688276, upper bound: 1.5006464
time: 2.10 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2

### Relational analysis result of NS_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5688276, upper bound: 1.5006464
time: 1.98 seconds

## BFS NS instance: NS_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.9315765, 0.7560664, -1.0479733, 0.8119298, -1.7435063, 1.8040397
1: -0.6937639, 0.8854553, -0.7680348, 1.0320082, -1.7257721, 1.6534901
2: -0.7029330, 0.9485440, -0.8098318, 1.0255426, -1.7284756, 1.7583759
3: -0.7776566, 0.6702975, -0.9194372, 0.7190377, -1.4966943, 1.5897347
4: -0.8823184, 0.8469309, -0.9903481, 0.9294721, -1.8117905, 1.8372790
5: -0.6828346, 1.1650977, -0.8683431, 1.1863770, -1.8692117, 2.0334408
6: -0.5836831, 0.7941089, -0.6675487, 0.8816245, -1.4653075, 1.4616576
7: -0.7100095, 0.9092127, -0.7951586, 1.0046657, -1.7146752, 1.7043713
8: -0.7961713, 0.9074162, -0.9205818, 0.9736586, -1.7698299, 1.8279980
9: -0.7868631, 0.8880752, -0.9008282, 0.9686235, -1.7554867, 1.7889035

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_A2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5525086, upper bound: 1.5006464
time: 2.15 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5561453, upper bound: 1.4978764
time: 1.87 seconds

## BFS NS instance: NS_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -1.0619879, 0.8189312, -0.6017754, 0.5910770, -1.6530650, 1.4207066
1: -0.7763894, 1.0458699, -0.5006989, 0.7040693, -1.4804586, 1.5465689
2: -0.8216664, 1.0350451, -0.4399196, 0.7246044, -1.5462708, 1.4749647
3: -0.9358491, 0.7244021, -0.4064578, 0.5587549, -1.4946039, 1.1308599
4: -1.0026883, 0.9399933, -0.6050895, 0.5938340, -1.5965223, 1.5450828
5: -0.8877734, 1.1884665, -0.3442929, 1.1638973, -2.0516706, 1.5327594
6: -0.6769806, 0.8909385, -0.3719376, 0.5924852, -1.2694658, 1.2628760
7: -0.8049077, 1.0157026, -0.4910856, 0.6563345, -1.4612422, 1.5067883
8: -0.9336596, 0.9811420, -0.5068572, 0.7465459, -1.6802055, 1.4879992
9: -0.9115970, 0.9781733, -0.5710402, 0.6704406, -1.5820376, 1.5492134

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_B1_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5006464, upper bound: 1.5688276
time: 1.71 seconds

## Relational analysis of NS_A2_B1_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5006464, upper bound: 1.5688276
time: 1.71 seconds

## BFS NS instance: NS_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -1.0479733, 0.8119298, -0.9315765, 0.7560664, -1.8040397, 1.7435063
1: -0.7680348, 1.0320082, -0.6937639, 0.8854553, -1.6534901, 1.7257721
2: -0.8098318, 1.0255426, -0.7029330, 0.9485440, -1.7583759, 1.7284756
3: -0.9194372, 0.7190377, -0.7776566, 0.6702975, -1.5897347, 1.4966943
4: -0.9903481, 0.9294721, -0.8823184, 0.8469309, -1.8372790, 1.8117905
5: -0.8683431, 1.1863770, -0.6828346, 1.1650977, -2.0334408, 1.8692117
6: -0.6675487, 0.8816245, -0.5836831, 0.7941089, -1.4616576, 1.4653075
7: -0.7951586, 1.0046657, -0.7100095, 0.9092127, -1.7043713, 1.7146752
8: -0.9205818, 0.9736586, -0.7961713, 0.9074162, -1.8279980, 1.7698299
9: -0.9008282, 0.9686235, -0.7868631, 0.8880752, -1.7889035, 1.7554867

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_B1_B2_A1

### Relational analysis result of NS_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5006464, upper bound: 1.5525086
time: 1.84 seconds

## Relational analysis of NS_A2_B1_B1_B2_A2

### Relational analysis result of NS_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4978764, upper bound: 1.5561453
time: 1.63 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1.0264742, 0.8009382, -1.0021820, 0.7891003, -1.8155746, 1.8031203
1: -0.7553862, 0.9988184, -0.7406971, 0.9879798, -1.7433660, 1.7395155
2: -0.7913131, 1.0109805, -0.7711954, 0.9944850, -1.7857981, 1.7821759
3: -0.8942685, 0.7101072, -0.8657864, 0.7016091, -1.5958776, 1.5758936
4: -0.9714069, 0.9130243, -0.9500130, 0.8951169, -1.8665239, 1.8630373
5: -0.8291278, 1.1826258, -0.8058947, 1.1800755, -2.0092034, 1.9885205
6: -0.6527300, 0.8675123, -0.6367680, 0.8511494, -1.5038793, 1.5042803
7: -0.7796060, 0.9876889, -0.7633727, 0.9686069, -1.7482129, 1.7510617
8: -0.9005474, 0.9617878, -0.8778203, 0.9492729, -1.8498203, 1.8396081
9: -0.8841767, 0.9536622, -0.8656385, 0.9374646, -1.8216413, 1.8193007

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5607533, upper bound: 1.5006464
time: 1.97 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5607533, upper bound: 1.5006464
time: 1.72 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1.0123142, 0.7938637, -1.3567265, 0.9618111, -1.9741253, 2.1505902
1: -0.7469429, 0.9848180, -0.9555715, 1.2033789, -1.9503219, 1.9403895
2: -0.7793526, 1.0013775, -1.0668855, 1.2355019, -2.0148544, 2.0682631
3: -0.8776824, 0.7046877, -1.2832209, 0.8269311, -1.7046134, 1.9879086
4: -0.9589357, 0.9023919, -1.2634817, 1.1581748, -2.1171105, 2.1658735
5: -0.8094906, 1.1805568, -1.1776241, 1.1826038, -1.9920944, 2.3581810
6: -0.6431984, 0.8580997, -0.8712379, 1.0904816, -1.7336800, 1.7293376
7: -0.7697535, 0.9765347, -1.0026691, 1.2472235, -2.0169771, 1.9792038
8: -0.8873308, 0.9542259, -1.2108443, 1.1327718, -2.0201025, 2.1650701
9: -0.8732941, 0.9440113, -1.1376344, 1.1751842, -2.0484784, 2.0816457

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5565576, upper bound: 1.4978764
time: 1.96 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5504444, upper bound: 1.4978764
time: 1.55 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.57 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.57
Output dim: 5, lower bound: -1.5687144, upper bound: 1.5231830
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.57
Output dim: 5, lower bound: -1.5687144, upper bound: 1.5366049
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.57
Output dim: 5, lower bound: -1.5635445, upper bound: 1.4991882
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.57
Output dim: 5, lower bound: -1.5635445, upper bound: 1.5046753
NS_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.57
Output dim: 5, lower bound: -1.5688276, upper bound: 1.5006464
NS_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.57
Output dim: 5, lower bound: -1.5688276, upper bound: 1.5006464
NS_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.57
Output dim: 5, lower bound: -1.5525086, upper bound: 1.5006464
NS_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.57
Output dim: 5, lower bound: -1.5561453, upper bound: 1.4978764
NS_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.57
Output dim: 5, lower bound: -1.5006464, upper bound: 1.5688276
NS_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.57
Output dim: 5, lower bound: -1.5006464, upper bound: 1.5688276
NS_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.57
Output dim: 5, lower bound: -1.5006464, upper bound: 1.5525086
NS_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.57
Output dim: 5, lower bound: -1.4978764, upper bound: 1.5561453
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.57
Output dim: 5, lower bound: -1.5607533, upper bound: 1.5006464
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.57
Output dim: 5, lower bound: -1.5607533, upper bound: 1.5006464
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.57
Output dim: 5, lower bound: -1.5565576, upper bound: 1.4978764
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.57
Output dim: 5, lower bound: -1.5504444, upper bound: 1.4978764

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.5633982, 0.5689398, -0.6171026, 0.5995962, -1.1629944, 1.1860424
1: -0.4788066, 0.6904866, -0.5092233, 0.7357920, -1.2145987, 1.1997099
2: -0.4115879, 0.6961117, -0.4526328, 0.7354299, -1.1470177, 1.1487446
3: -0.3670223, 0.5440686, -0.4228440, 0.5651449, -0.9321672, 0.9669126
4: -0.5752214, 0.5645321, -0.6173927, 0.6063303, -1.1815517, 1.1819248
5: -0.3074960, 1.1395450, -0.3807664, 1.1634811, -1.4709771, 1.5203114
6: -0.3463388, 0.5704345, -0.3829734, 0.6012567, -0.9475955, 0.9534079
7: -0.4658466, 0.6242265, -0.5024189, 0.6689196, -1.1347661, 1.1266453
8: -0.4743436, 0.7261795, -0.5204343, 0.7548758, -1.2292194, 1.2466137
9: -0.5443203, 0.6443796, -0.5815639, 0.6814047, -1.2257249, 1.2259436

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5521960, upper bound: 1.5190737
time: 2.21 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5558063, upper bound: 1.5183453
time: 1.73 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.6148476, 0.5982409, -0.6337503, 0.6084861, -1.2233336, 1.2319913
1: -0.5080075, 0.7250603, -0.5190299, 0.7488049, -1.2568123, 1.2440901
2: -0.4505857, 0.7338771, -0.4657184, 0.7468294, -1.1974151, 1.1995956
3: -0.4204007, 0.5638877, -0.4402518, 0.5709973, -0.9913980, 1.0041395
4: -0.6155965, 0.6043096, -0.6303590, 0.6194844, -1.2350808, 1.2346686
5: -0.3702411, 1.1634848, -0.4021270, 1.1689751, -1.5392163, 1.5656118
6: -0.3812008, 0.6000586, -0.3941028, 0.6106063, -0.9918071, 0.9941614
7: -0.5004390, 0.6668987, -0.5140792, 0.6816821, -1.1821210, 1.1809778
8: -0.5184724, 0.7534574, -0.5348539, 0.7633204, -1.2817929, 1.2883112
9: -0.5799613, 0.6796618, -0.5923197, 0.6924674, -1.2724288, 1.2719815

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5687144, upper bound: 1.5366049
time: 1.81 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5687144, upper bound: 1.5366049
time: 1.77 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.5548620, 0.5633116, -0.9527297, 0.7660553, -1.3209174, 1.5160413
1: -0.4737587, 0.6800896, -0.7062024, 0.9157274, -1.3894861, 1.3862920
2: -0.4047802, 0.6888762, -0.7218725, 0.9595096, -1.3642899, 1.4107487
3: -0.3582624, 0.5401324, -0.8004873, 0.6770553, -1.0353178, 1.3406197
4: -0.5682003, 0.5577450, -0.8973897, 0.8638864, -1.4320867, 1.4551346
5: -0.2934766, 1.1376083, -0.7267386, 1.1652608, -1.4587374, 1.8643469
6: -0.3412620, 0.5651462, -0.5950376, 0.8051454, -1.1464075, 1.1601839
7: -0.4590116, 0.6167913, -0.7253720, 0.9227059, -1.3817174, 1.3421633
8: -0.4673302, 0.7207871, -0.8245822, 0.9160954, -1.3834255, 1.5453694
9: -0.5378525, 0.6379646, -0.8089782, 0.8994461, -1.4372987, 1.4469428

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5503113, upper bound: 1.4991882
time: 1.60 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5503113, upper bound: 1.4991882
time: 1.88 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.6030326, 0.5917985, -0.9714550, 0.7755836, -1.3786162, 1.5632535
1: -0.5013898, 0.7128651, -0.7169251, 0.9300792, -1.4314690, 1.4297903
2: -0.4412048, 0.7255199, -0.7369878, 0.9719032, -1.4131080, 1.4625077
3: -0.4079365, 0.5595412, -0.8214043, 0.6838393, -1.0917759, 1.3809456
4: -0.6062395, 0.5950349, -0.9129306, 0.8782474, -1.4844868, 1.5079656
5: -0.3525497, 1.1615305, -0.7507657, 1.1708025, -1.5233521, 1.9122962
6: -0.3730458, 0.5932099, -0.6069249, 0.8163485, -1.1893944, 1.2001348
7: -0.4923488, 0.6575755, -0.7379839, 0.9368770, -1.4292258, 1.3955594
8: -0.5080939, 0.7474019, -0.8414077, 0.9253340, -1.4334278, 1.5888096
9: -0.5720584, 0.6715397, -0.8216796, 0.9116837, -1.4837421, 1.4932194

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_A2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5593486, upper bound: 1.5018314
time: 2.02 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5505460, upper bound: 1.5018314
time: 1.74 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.6017754, 0.5910770, -1.0021820, 0.7891003, -1.3908758, 1.5932591
1: -0.5006989, 0.7040693, -0.7406971, 0.9879798, -1.4886787, 1.4447664
2: -0.4399196, 0.7246044, -0.7711954, 0.9944850, -1.4344046, 1.4957998
3: -0.4064578, 0.5587549, -0.8657864, 0.7016091, -1.1080669, 1.4245412
4: -0.6050895, 0.5938340, -0.9500130, 0.8951169, -1.5002065, 1.5438471
5: -0.3442929, 1.1638973, -0.8058947, 1.1800755, -1.5243685, 1.9697920
6: -0.3719376, 0.5924852, -0.6367680, 0.8511494, -1.2230870, 1.2292532
7: -0.4910856, 0.6563345, -0.7633727, 0.9686069, -1.4596926, 1.4197073
8: -0.5068572, 0.7465459, -0.8778203, 0.9492729, -1.4561300, 1.6243662
9: -0.5710402, 0.6704406, -0.8656385, 0.9374646, -1.5085047, 1.5360790

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_A1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5692598, upper bound: 1.4986285
time: 1.58 seconds

## Relational analysis of NS_A1_B2_A1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5692598, upper bound: 1.5006464
time: 2.08 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.6017754, 0.5910770, -1.3567265, 0.9618111, -1.5635865, 1.9478035
1: -0.5006989, 0.7040693, -0.9555715, 1.2033789, -1.7040778, 1.6596408
2: -0.4399196, 0.7246044, -1.0668855, 1.2355019, -1.6754215, 1.7914898
3: -0.4064578, 0.5587549, -1.2832209, 0.8269311, -1.2333889, 1.8419757
4: -0.6050895, 0.5938340, -1.2634817, 1.1581748, -1.7632643, 1.8573158
5: -0.3442929, 1.1638973, -1.1776241, 1.1826038, -1.5268967, 2.3415213
6: -0.3719376, 0.5924852, -0.8712379, 1.0904816, -1.4624193, 1.4637231
7: -0.4910856, 0.6563345, -1.0026691, 1.2472235, -1.7383091, 1.6590036
8: -0.5068572, 0.7465459, -1.2108443, 1.1327718, -1.6396290, 1.9573902
9: -0.5710402, 0.6704406, -1.1376344, 1.1751842, -1.7462244, 1.8080750

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_A1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5692598, upper bound: 1.4986285
time: 1.57 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5692598, upper bound: 1.5006464
time: 1.79 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.9286402, 0.7545813, -0.9604194, 0.7684886, -1.6971288, 1.7150006
1: -0.6920697, 0.8831980, -0.7156307, 0.9614974, -1.6535671, 1.5988287
2: -0.7005666, 0.9465630, -0.7364969, 0.9661483, -1.6667149, 1.6830599
3: -0.7743375, 0.6692406, -0.8170086, 0.6864815, -1.4608190, 1.4862492
4: -0.8798335, 0.8446745, -0.9133524, 0.8641971, -1.7440306, 1.7580268
5: -0.6792139, 1.1646707, -0.7603054, 1.1733613, -1.8525752, 1.9249761
6: -0.5817899, 0.7923188, -0.6091513, 0.8232130, -1.4050028, 1.4014701
7: -0.7080339, 0.9069661, -0.7351689, 0.9357533, -1.6437871, 1.6421349
8: -0.7936136, 0.9059546, -0.8388800, 0.9275126, -1.7211263, 1.7448347
9: -0.7849431, 0.8861207, -0.8337696, 0.9095037, -1.6944468, 1.7198904

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5525086, upper bound: 1.4978764
time: 1.70 seconds

## Relational analysis of NS_A1_B2_A1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5525086, upper bound: 1.4978764
time: 1.73 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.9256229, 0.7530423, -1.2353903, 0.9028609, -1.8284838, 1.9884326
1: -0.6903357, 0.8802221, -0.8819233, 1.1373546, -1.8276904, 1.7621454
2: -0.6981087, 0.9445271, -0.9660013, 1.1530033, -1.8511121, 1.9105284
3: -0.7709200, 0.6681153, -1.1404033, 0.7845557, -1.5554757, 1.8085185
4: -0.8772749, 0.8423353, -1.1562454, 1.0683935, -1.9456683, 1.9985807
5: -0.6749483, 1.1642908, -1.0559487, 1.1813469, -1.8562951, 2.2202396
6: -0.5798209, 0.7904862, -0.7912719, 1.0084571, -1.5882781, 1.5817581
7: -0.7059647, 0.9046549, -0.9212539, 1.1518955, -1.8578603, 1.8259088
8: -0.7909827, 0.9044267, -1.0968750, 1.0702790, -1.8612617, 2.0013018
9: -0.7829581, 0.8840894, -1.0446663, 1.0940982, -1.8770564, 1.9287556

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_A1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5495113, upper bound: 1.4883344
time: 1.71 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5495113, upper bound: 1.4978764
time: 1.91 seconds

## BFS NS instance: NS_A2_B1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -1.0021820, 0.7891003, -0.6017754, 0.5910770, -1.5932591, 1.3908758
1: -0.7406971, 0.9879798, -0.5006989, 0.7040693, -1.4447664, 1.4886787
2: -0.7711954, 0.9944850, -0.4399196, 0.7246044, -1.4957998, 1.4344046
3: -0.8657864, 0.7016091, -0.4064578, 0.5587549, -1.4245412, 1.1080669
4: -0.9500130, 0.8951169, -0.6050895, 0.5938340, -1.5438471, 1.5002065
5: -0.8058947, 1.1800755, -0.3442929, 1.1638973, -1.9697920, 1.5243685
6: -0.6367680, 0.8511494, -0.3719376, 0.5924852, -1.2292532, 1.2230870
7: -0.7633727, 0.9686069, -0.4910856, 0.6563345, -1.4197073, 1.4596926
8: -0.8778203, 0.9492729, -0.5068572, 0.7465459, -1.6243662, 1.4561300
9: -0.8656385, 0.9374646, -0.5710402, 0.6704406, -1.5360790, 1.5085047

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A2_B1_B1_B1_A1_B1

### Relational analysis result of NS_A2_B1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4986285, upper bound: 1.5692598
time: 1.70 seconds

## Relational analysis of NS_A2_B1_B1_B1_A1_B2

### Relational analysis result of NS_A2_B1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5006464, upper bound: 1.5692598
time: 1.75 seconds

## BFS NS instance: NS_A2_B1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -1.3567265, 0.9618111, -0.6017754, 0.5910770, -1.9478035, 1.5635865
1: -0.9555715, 1.2033789, -0.5006989, 0.7040693, -1.6596408, 1.7040778
2: -1.0668855, 1.2355019, -0.4399196, 0.7246044, -1.7914898, 1.6754215
3: -1.2832209, 0.8269311, -0.4064578, 0.5587549, -1.8419757, 1.2333889
4: -1.2634817, 1.1581748, -0.6050895, 0.5938340, -1.8573158, 1.7632643
5: -1.1776241, 1.1826038, -0.3442929, 1.1638973, -2.3415213, 1.5268967
6: -0.8712379, 1.0904816, -0.3719376, 0.5924852, -1.4637231, 1.4624193
7: -1.0026691, 1.2472235, -0.4910856, 0.6563345, -1.6590036, 1.7383091
8: -1.2108443, 1.1327718, -0.5068572, 0.7465459, -1.9573902, 1.6396290
9: -1.1376344, 1.1751842, -0.5710402, 0.6704406, -1.8080750, 1.7462244

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A2_B1_B1_B1_A2_B1

### Relational analysis result of NS_A2_B1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4986285, upper bound: 1.5692598
time: 1.75 seconds

## Relational analysis of NS_A2_B1_B1_B1_A2_B2

### Relational analysis result of NS_A2_B1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5006464, upper bound: 1.5692598
time: 1.74 seconds

## BFS NS instance: NS_A2_B1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.9604194, 0.7684886, -0.9286402, 0.7545813, -1.7150006, 1.6971288
1: -0.7156307, 0.9614974, -0.6920697, 0.8831980, -1.5988287, 1.6535671
2: -0.7364969, 0.9661483, -0.7005666, 0.9465630, -1.6830599, 1.6667149
3: -0.8170086, 0.6864815, -0.7743375, 0.6692406, -1.4862492, 1.4608190
4: -0.9133524, 0.8641971, -0.8798335, 0.8446745, -1.7580268, 1.7440306
5: -0.7603054, 1.1733613, -0.6792139, 1.1646707, -1.9249761, 1.8525752
6: -0.6091513, 0.8232130, -0.5817899, 0.7923188, -1.4014701, 1.4050028
7: -0.7351689, 0.9357533, -0.7080339, 0.9069661, -1.6421349, 1.6437871
8: -0.8388800, 0.9275126, -0.7936136, 0.9059546, -1.7448347, 1.7211263
9: -0.8337696, 0.9095037, -0.7849431, 0.8861207, -1.7198904, 1.6944468

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_B1_B2_A1_B1

### Relational analysis result of NS_A2_B1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4978764, upper bound: 1.5525086
time: 1.75 seconds

## Relational analysis of NS_A2_B1_B1_B2_A1_B2

### Relational analysis result of NS_A2_B1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4978764, upper bound: 1.5525086
time: 1.68 seconds

## BFS NS instance: NS_A2_B1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -1.2353903, 0.9028609, -0.9256229, 0.7530423, -1.9884326, 1.8284838
1: -0.8819233, 1.1373546, -0.6903357, 0.8802221, -1.7621454, 1.8276904
2: -0.9660013, 1.1530033, -0.6981087, 0.9445271, -1.9105284, 1.8511121
3: -1.1404033, 0.7845557, -0.7709200, 0.6681153, -1.8085185, 1.5554757
4: -1.1562454, 1.0683935, -0.8772749, 0.8423353, -1.9985807, 1.9456683
5: -1.0559487, 1.1813469, -0.6749483, 1.1642908, -2.2202396, 1.8562951
6: -0.7912719, 1.0084571, -0.5798209, 0.7904862, -1.5817581, 1.5882781
7: -0.9212539, 1.1518955, -0.7059647, 0.9046549, -1.8259088, 1.8578603
8: -1.0968750, 1.0702790, -0.7909827, 0.9044267, -2.0013018, 1.8612617
9: -1.0446663, 1.0940982, -0.7829581, 0.8840894, -1.9287556, 1.8770564

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A2_B1_B1_B2_A2_B1

### Relational analysis result of NS_A2_B1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4883344, upper bound: 1.5495113
time: 1.55 seconds

## Relational analysis of NS_A2_B1_B1_B2_A2_B2

### Relational analysis result of NS_A2_B1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4978764, upper bound: 1.5495113
time: 1.65 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.9651505, 0.7703574, -1.0021820, 0.7891003, -1.7542508, 1.7725394
1: -0.7187708, 0.9398615, -0.7406971, 0.9879798, -1.7067506, 1.6805587
2: -0.7395604, 0.9693825, -0.7711954, 0.9944850, -1.7340454, 1.7405779
3: -0.8224011, 0.6867737, -0.8657864, 0.7016091, -1.5240102, 1.5525601
4: -0.9173770, 0.8670150, -0.9500130, 0.8951169, -1.8124939, 1.8170280
5: -0.7455353, 1.1743234, -0.8058947, 1.1800755, -1.9256108, 1.9802182
6: -0.6115028, 0.8266949, -0.6367680, 0.8511494, -1.4626522, 1.4634628
7: -0.7370316, 0.9393928, -0.7633727, 0.9686069, -1.7056385, 1.7027655
8: -0.8432758, 0.9291166, -0.8778203, 0.9492729, -1.7925487, 1.8069369
9: -0.8370489, 0.9119180, -0.8656385, 0.9374646, -1.7745135, 1.7775565

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5525538, upper bound: 1.5327385
time: 1.61 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5561671, upper bound: 1.5324480
time: 1.65 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1.3190053, 0.9427488, -1.0021820, 0.7891003, -2.1081057, 1.9449308
1: -0.9332042, 1.1568534, -0.7406971, 0.9879798, -1.9211839, 1.8975506
2: -1.0347108, 1.2099220, -0.7711954, 0.9944850, -2.0291958, 1.9811174
3: -1.2390045, 0.8119444, -0.8657864, 0.7016091, -1.9406136, 1.6777308
4: -1.2302247, 1.1296053, -0.9500130, 0.8951169, -2.1253417, 2.0796185
5: -1.1193368, 1.1762189, -0.8058947, 1.1800755, -2.2994123, 1.9821136
6: -0.8455557, 1.0655383, -0.6367680, 0.8511494, -1.6967051, 1.7023063
7: -0.9759331, 1.2174655, -0.7633727, 0.9686069, -1.9445400, 1.9808383
8: -1.1756496, 1.1122890, -0.8778203, 0.9492729, -2.1249225, 1.9901092
9: -1.1085353, 1.1491958, -0.8656385, 0.9374646, -2.0460000, 2.0148344

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5525538, upper bound: 1.5327385
time: 2.74 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5561671, upper bound: 1.5324480
time: 1.65 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.9258955, 0.7510032, -1.3535842, 0.9602531, -1.8861486, 2.1045873
1: -0.6951942, 0.9156294, -0.9536896, 1.2008419, -1.8960361, 1.8693190
2: -0.7069680, 0.9427447, -1.0642514, 1.2333690, -1.9403369, 2.0069962
3: -0.7765459, 0.6726069, -1.2795404, 0.8257641, -1.6023099, 1.9521474
4: -0.8829145, 0.8379726, -1.2607163, 1.1558311, -2.0387456, 2.0986888
5: -0.7031187, 1.1678039, -1.1737610, 1.1821527, -1.8852714, 2.3415649
6: -0.5855679, 0.8004208, -0.8691404, 1.0883838, -1.6739516, 1.6695611
7: -0.7105628, 0.9085158, -1.0005150, 1.2447505, -1.9553133, 1.9090308
8: -0.8066685, 0.9086926, -1.2079095, 1.1311158, -1.9377842, 2.1166019
9: -0.8071000, 0.8856572, -1.1352260, 1.1730614, -1.9801614, 2.0208831

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5465795, upper bound: 1.4978764
time: 2.08 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5465795, upper bound: 1.4978764
time: 1.86 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1.2015018, 0.8856845, -1.3505220, 0.9587256, -2.1602273, 2.2362065
1: -0.8618726, 1.0929892, -0.9518602, 1.1978183, -2.0596910, 2.0448494
2: -0.9370291, 1.1300316, -1.0616628, 1.2312921, -2.1683211, 2.1916943
3: -1.1007035, 0.7709315, -1.2759488, 0.8245947, -1.9252982, 2.0468802
4: -1.1263803, 1.0426614, -1.2580160, 1.1535295, -2.2799098, 2.3006773
5: -1.0014402, 1.1752590, -1.1696004, 1.1817656, -2.1832056, 2.3448594
6: -0.7681288, 0.9860938, -0.8670785, 1.0863448, -1.8544736, 1.8531723
7: -0.8971119, 1.1251557, -0.9983830, 1.2423382, -2.1394501, 2.1235387
8: -1.0652705, 1.0518020, -1.2050478, 1.1294817, -2.1947522, 2.2568498
9: -1.0184999, 1.0706968, -1.1328695, 1.1709735, -2.1894734, 2.2035663

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5465795, upper bound: 1.4978764
time: 2.27 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5465795, upper bound: 1.4978764
time: 1.92 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 5.96 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 5, lower bound: -1.5521960, upper bound: 1.5190737
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 5, lower bound: -1.5558063, upper bound: 1.5183453
NS_A1_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 5, lower bound: -1.5687144, upper bound: 1.5366049
NS_A1_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 5, lower bound: -1.5687144, upper bound: 1.5366049
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 5, lower bound: -1.5503113, upper bound: 1.4991882
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 5, lower bound: -1.5503113, upper bound: 1.4991882
NS_A1_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 5, lower bound: -1.5593486, upper bound: 1.5018314
NS_A1_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 5, lower bound: -1.5505460, upper bound: 1.5018314
NS_A1_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 5, lower bound: -1.5692598, upper bound: 1.4986285
NS_A1_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 5, lower bound: -1.5692598, upper bound: 1.5006464
NS_A1_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 5, lower bound: -1.5692598, upper bound: 1.4986285
NS_A1_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 5, lower bound: -1.5692598, upper bound: 1.5006464
NS_A1_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 5, lower bound: -1.5525086, upper bound: 1.4978764
NS_A1_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 5, lower bound: -1.5525086, upper bound: 1.4978764
NS_A1_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 5, lower bound: -1.5495113, upper bound: 1.4883344
NS_A1_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 5, lower bound: -1.5495113, upper bound: 1.4978764
NS_A2_B1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 5, lower bound: -1.4986285, upper bound: 1.5692598
NS_A2_B1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 5, lower bound: -1.5006464, upper bound: 1.5692598
NS_A2_B1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 5, lower bound: -1.4986285, upper bound: 1.5692598
NS_A2_B1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 5, lower bound: -1.5006464, upper bound: 1.5692598
NS_A2_B1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 5, lower bound: -1.4978764, upper bound: 1.5525086
NS_A2_B1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 5, lower bound: -1.4978764, upper bound: 1.5525086
NS_A2_B1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 5, lower bound: -1.4883344, upper bound: 1.5495113
NS_A2_B1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 5, lower bound: -1.4978764, upper bound: 1.5495113
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 5, lower bound: -1.5525538, upper bound: 1.5327385
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 5, lower bound: -1.5561671, upper bound: 1.5324480
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 5, lower bound: -1.5525538, upper bound: 1.5327385
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 5, lower bound: -1.5561671, upper bound: 1.5324480
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 5, lower bound: -1.5465795, upper bound: 1.4978764
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 5, lower bound: -1.5465795, upper bound: 1.4978764
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 5, lower bound: -1.5465795, upper bound: 1.4978764
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 5, lower bound: -1.5465795, upper bound: 1.4978764

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.5615433, 0.5677513, -0.5516181, 0.5615857, -1.1231289, 1.1193694
1: -0.4776802, 0.6888618, -0.4712898, 0.6814693, -1.1591495, 1.1601516
2: -0.4101332, 0.6945543, -0.4020353, 0.6858026, -1.0959357, 1.0965897
3: -0.3651401, 0.5432546, -0.3541295, 0.5396401, -0.9047802, 0.8973840
4: -0.5736591, 0.5631060, -0.5651512, 0.5549644, -1.1286235, 1.1282572
5: -0.3050084, 1.1391411, -0.2953038, 1.1511328, -1.4561412, 1.4344449
6: -0.3452637, 0.5692508, -0.3395725, 0.5626274, -0.9078911, 0.9088234
7: -0.4644209, 0.6226522, -0.4563878, 0.6159503, -1.0803711, 1.0790401
8: -0.4728004, 0.7250535, -0.4642555, 0.7190107, -1.1918111, 1.1893091
9: -0.5429381, 0.6429994, -0.5349725, 0.6354777, -1.1784158, 1.1779718

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5439451, upper bound: 1.5190737
time: 1.65 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5439451, upper bound: 1.5190737
time: 1.75 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.5597346, 0.5665687, -0.7697145, 0.6781471, -1.2378817, 1.3362832
1: -0.4765722, 0.6866383, -0.6017669, 0.8058647, -1.2824368, 1.2884052
2: -0.4086696, 0.6930238, -0.5719693, 0.8404937, -1.2491633, 1.2649931
3: -0.3632936, 0.5424174, -0.5850106, 0.6130869, -0.9763805, 1.1274281
4: -0.5721267, 0.5616771, -0.7378362, 0.7265887, -1.2987154, 1.2995133
5: -0.3020072, 1.1387842, -0.5241472, 1.1579883, -1.4599955, 1.6629314
6: -0.3441997, 0.5680857, -0.4836946, 0.6893014, -1.0335011, 1.0517802
7: -0.4629831, 0.6210673, -0.6069094, 0.7872602, -1.2502434, 1.2279767
8: -0.4712815, 0.7239193, -0.6548724, 0.8293373, -1.3006189, 1.3787918
9: -0.5415809, 0.6416145, -0.6805933, 0.7817035, -1.3232844, 1.3222077

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5457925, upper bound: 1.5183453
time: 1.96 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5457925, upper bound: 1.5183453
time: 1.73 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.5672411, 0.5714372, -0.6337503, 0.6084861, -1.1757271, 1.2051876
1: -0.4807033, 0.6761887, -0.5190299, 0.7488049, -1.2295082, 1.1952186
2: -0.4134003, 0.6989219, -0.4657184, 0.7468294, -1.1602297, 1.1646404
3: -0.3698834, 0.5453974, -0.4402518, 0.5709973, -0.9408807, 0.9856492
4: -0.5775470, 0.5667447, -0.6303590, 0.6194844, -1.1970313, 1.1971037
5: -0.2994074, 1.1559496, -0.4021270, 1.1689751, -1.4683826, 1.5580766
6: -0.3482073, 0.5723276, -0.3941028, 0.6106063, -0.9588136, 0.9664304
7: -0.4675447, 0.6279905, -0.5140792, 0.6816821, -1.1492267, 1.1420696
8: -0.4768460, 0.7280012, -0.5348539, 0.7633204, -1.2401664, 1.2628551
9: -0.5463970, 0.6463131, -0.5923197, 0.6924674, -1.2388644, 1.2386328

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B1_A1_B1_A2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5586556, upper bound: 1.5366048
time: 1.83 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5586556, upper bound: 1.5366049
time: 1.66 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.8724293, 0.7321481, -0.6337503, 0.6084861, -1.4809153, 1.3658984
1: -0.6597273, 0.8400022, -0.5190299, 0.7488049, -1.4085321, 1.3590320
2: -0.6493913, 0.9130070, -0.4657184, 0.7468294, -1.3962207, 1.3787255
3: -0.6967273, 0.6454272, -0.4402518, 0.5709973, -1.2677246, 1.0856791
4: -0.8154522, 0.8091797, -0.6303590, 0.6194844, -1.4349365, 1.4395387
5: -0.6127775, 1.1567936, -0.4021270, 1.1689751, -1.7817526, 1.5589206
6: -0.5527496, 0.7497519, -0.3941028, 0.6106063, -1.1633558, 1.1438547
7: -0.6721511, 0.8690442, -0.5140792, 0.6816821, -1.3538332, 1.3831234
8: -0.7415717, 0.8802458, -0.5348539, 0.7633204, -1.5048921, 1.4150996
9: -0.7481984, 0.8504668, -0.5923197, 0.6924674, -1.4406657, 1.4427866

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B1_A1_B1_A2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5586556, upper bound: 1.5366048
time: 1.74 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5586556, upper bound: 1.5366049
time: 1.77 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.5548620, 0.5633116, -0.8419616, 0.7100949, -1.2649570, 1.4052732
1: -0.4737587, 0.6800896, -0.6424602, 0.8413903, -1.3151489, 1.3225498
2: -0.4047802, 0.6888762, -0.6327407, 0.8861454, -1.2909256, 1.3216169
3: -0.3582624, 0.5401324, -0.6765327, 0.6378064, -0.9960688, 1.2166650
4: -0.5682003, 0.5577450, -0.8053375, 0.7791409, -1.3473413, 1.3630825
5: -0.2934766, 1.1376083, -0.5942552, 1.1386817, -1.4321582, 1.7318635
6: -0.3412620, 0.5651462, -0.5250557, 0.7385710, -1.0798330, 1.0902019
7: -0.4590116, 0.6167913, -0.6513444, 0.8389583, -1.2979698, 1.2681357
8: -0.4673302, 0.7207871, -0.7248328, 0.8619634, -1.3292935, 1.4456198
9: -0.5378525, 0.6379646, -0.7338808, 0.8273777, -1.3652303, 1.3718454

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5456782, upper bound: 1.4927321
time: 1.84 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5376395, upper bound: 1.4929717
time: 1.47 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.5548620, 0.5633116, -0.9265137, 0.7529350, -1.3077970, 1.4898252
1: -0.4737587, 0.6800896, -0.6909959, 0.8966742, -1.3704329, 1.3710855
2: -0.4047802, 0.6888762, -0.7006226, 0.9421192, -1.3468995, 1.3894987
3: -0.3582624, 0.5401324, -0.7709039, 0.6678675, -1.0261300, 1.3110363
4: -0.5682003, 0.5577450, -0.8754370, 0.8437187, -1.4119190, 1.4331820
5: -0.2934766, 1.1376083, -0.6948385, 1.1627233, -1.4561999, 1.8324468
6: -0.3412620, 0.5651462, -0.5784094, 0.7892718, -1.1305338, 1.1435556
7: -0.4590116, 0.6167913, -0.7077425, 0.9029057, -1.3619173, 1.3245337
8: -0.4673302, 0.7207871, -0.8007981, 0.9033054, -1.3706355, 1.5215852
9: -0.5378525, 0.6379646, -0.7910853, 0.8823017, -1.4201542, 1.4290500

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5364309, upper bound: 1.4948836
time: 1.70 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5376395, upper bound: 1.4929717
time: 1.54 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.5435179, 0.5559961, -0.9684985, 0.7740972, -1.3176150, 1.5244945
1: -0.4667313, 0.6630547, -0.7152162, 0.9278353, -1.3945667, 1.3782709
2: -0.3953950, 0.6789523, -0.7345923, 0.9699434, -1.3653384, 1.4135445
3: -0.3456593, 0.5356200, -0.8180768, 0.6827921, -1.0284513, 1.3536968
4: -0.5587332, 0.5483223, -0.9104605, 0.8759729, -1.4347060, 1.4587827
5: -0.2748681, 1.1494682, -0.7470795, 1.1703699, -1.4452380, 1.8965477
6: -0.3347107, 0.5578873, -0.6050481, 0.8145642, -1.1492749, 1.1629354
7: -0.4493440, 0.6087279, -0.7359929, 0.9346429, -1.3839869, 1.3447208
8: -0.4579076, 0.7135138, -0.8387316, 0.9238868, -1.3817943, 1.5522454
9: -0.5286373, 0.6293439, -0.8196642, 0.9097503, -1.4383876, 1.4490081

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B1_A1_B2_A2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5456782, upper bound: 1.5016473
time: 1.99 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5456782, upper bound: 1.5018314
time: 1.83 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.7497785, 0.6680276, -0.9654684, 0.7725629, -1.5223414, 1.6334960
1: -0.5841198, 0.7865181, -0.7134706, 0.9249222, -1.5090420, 1.4999887
2: -0.5511015, 0.8313252, -0.7321126, 0.9679338, -1.5190353, 1.5634377
3: -0.5593077, 0.6087750, -0.8146586, 0.6816828, -1.2409904, 1.4234335
4: -0.7247543, 0.7049389, -0.9079223, 0.8736238, -1.5983782, 1.6128612
5: -0.4981094, 1.1559100, -0.7428042, 1.1699808, -1.6680901, 1.8987142
6: -0.4712533, 0.6799220, -0.6031029, 0.8127411, -1.2839944, 1.2830248
7: -0.5871033, 0.7772533, -0.7339161, 0.9323498, -1.5194530, 1.5111694
8: -0.6400384, 0.8188547, -0.8359854, 0.9223803, -1.5624187, 1.6548402
9: -0.6703289, 0.7684357, -0.8175870, 0.9077461, -1.5780749, 1.5860226

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5470452, upper bound: 1.5018314
time: 1.71 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5470452, upper bound: 1.5018314
time: 2.00 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.5263217, 0.5448294, -0.9834900, 0.7797163, -1.3060380, 1.5283194
1: -0.4573160, 0.6466167, -0.7296106, 0.9725906, -1.4299066, 1.3762273
2: -0.3826107, 0.6646317, -0.7555886, 0.9818274, -1.3644381, 1.4202204
3: -0.3286017, 0.5278014, -0.8440751, 0.6945100, -1.0231117, 1.3718765
4: -0.5462130, 0.5351171, -0.9336795, 0.8812118, -1.4274248, 1.4687966
5: -0.2489608, 1.1318836, -0.7820457, 1.1748499, -1.4238107, 1.9139293
6: -0.3253044, 0.5482060, -0.6243007, 0.8387851, -1.1640894, 1.1725068
7: -0.4359377, 0.5933949, -0.7505552, 0.9538813, -1.3898189, 1.3439500
8: -0.4451309, 0.7025202, -0.8604984, 0.9393537, -1.3844845, 1.5630186
9: -0.5160955, 0.6172245, -0.8513950, 0.9248503, -1.4409459, 1.4686196

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_A1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5721794, upper bound: 1.5299768
time: 2.07 seconds

## Relational analysis of NS_A1_B2_A1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5721794, upper bound: 1.5299768
time: 1.94 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.5672411, 0.5714372, -1.0021820, 0.7891003, -1.3563415, 1.5736191
1: -0.4807033, 0.6761887, -0.7406971, 0.9879798, -1.4686830, 1.4168859
2: -0.4134003, 0.6989219, -0.7711954, 0.9944850, -1.4078853, 1.4701173
3: -0.3698834, 0.5453974, -0.8657864, 0.7016091, -1.0714926, 1.4111838
4: -0.5775470, 0.5667447, -0.9500130, 0.8951169, -1.4726639, 1.5167577
5: -0.2994074, 1.1559496, -0.8058947, 1.1800755, -1.4794829, 1.9618443
6: -0.3482073, 0.5723276, -0.6367680, 0.8511494, -1.1993567, 1.2090956
7: -0.4675447, 0.6279905, -0.7633727, 0.9686069, -1.4361516, 1.3913631
8: -0.4768460, 0.7280012, -0.8778203, 0.9492729, -1.4261189, 1.6058215
9: -0.5463970, 0.6463131, -0.8656385, 0.9374646, -1.4838617, 1.5119516

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_A1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5721794, upper bound: 1.5352767
time: 1.62 seconds

## Relational analysis of NS_A1_B2_A1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5721794, upper bound: 1.5352767
time: 1.86 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.5263217, 0.5448294, -1.3371168, 0.9519856, -1.4783072, 1.8819462
1: -0.4573160, 0.6466167, -0.9439206, 1.1873144, -1.6446304, 1.5905373
2: -0.3826107, 0.6646317, -1.0505031, 1.2222170, -1.6048276, 1.7151349
3: -0.3286017, 0.5278014, -1.2604107, 0.8195143, -1.1481160, 1.7882121
4: -0.5462130, 0.5351171, -1.2463251, 1.1435820, -1.6897950, 1.7814423
5: -0.2489608, 1.1318836, -1.1527361, 1.1773784, -1.4263391, 2.2846198
6: -0.3253044, 0.5482060, -0.8581581, 1.0774893, -1.4027936, 1.4063642
7: -0.4359377, 0.5933949, -0.9892256, 1.2317768, -1.6677146, 1.5826205
8: -0.4451309, 0.7025202, -1.1926479, 1.1223788, -1.5675097, 1.8951681
9: -0.5160955, 0.6172245, -1.1226774, 1.1619502, -1.6780457, 1.7399020

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 218

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_A1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5550893, upper bound: 1.4986285
time: 1.69 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5550893, upper bound: 1.4986285
time: 2.12 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.5672411, 0.5714372, -1.3567265, 0.9618111, -1.5290523, 1.9281638
1: -0.4807033, 0.6761887, -0.9555715, 1.2033789, -1.6840823, 1.6317602
2: -0.4134003, 0.6989219, -1.0668855, 1.2355019, -1.6489022, 1.7658074
3: -0.3698834, 0.5453974, -1.2832209, 0.8269311, -1.1968145, 1.8286183
4: -0.5775470, 0.5667447, -1.2634817, 1.1581748, -1.7357217, 1.8302264
5: -0.2994074, 1.1559496, -1.1776241, 1.1826038, -1.4820113, 2.3335738
6: -0.3482073, 0.5723276, -0.8712379, 1.0904816, -1.4386890, 1.4435655
7: -0.4675447, 0.6279905, -1.0026691, 1.2472235, -1.7147682, 1.6306596
8: -0.4768460, 0.7280012, -1.2108443, 1.1327718, -1.6096179, 1.9388455
9: -0.5463970, 0.6463131, -1.1376344, 1.1751842, -1.7215812, 1.7839475

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_A1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5550893, upper bound: 1.5006464
time: 1.80 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5550893, upper bound: 1.5006464
time: 1.73 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.8431016, 0.7113134, -0.9604194, 0.7684886, -1.6115901, 1.6717329
1: -0.6427077, 0.8172264, -0.7156307, 0.9614974, -1.6042051, 1.5328571
2: -0.6315987, 0.8888518, -0.7364969, 0.9661483, -1.5977470, 1.6253487
3: -0.6776149, 0.6384385, -0.8170086, 0.6864815, -1.3640964, 1.4554471
4: -0.8074291, 0.7789104, -0.9133524, 0.8641971, -1.6716263, 1.6922628
5: -0.5733571, 1.1524251, -0.7603054, 1.1733613, -1.7467184, 1.9127305
6: -0.5266141, 0.7401577, -0.6091513, 0.8232130, -1.3498271, 1.3493090
7: -0.6504493, 0.8415111, -0.7351689, 0.9357533, -1.5862026, 1.5766799
8: -0.7190910, 0.8633411, -0.8388800, 0.9275126, -1.6466036, 1.7022212
9: -0.7289797, 0.8291612, -0.8337696, 0.9095037, -1.6384834, 1.6629307

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_A1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5525086, upper bound: 1.5006464
time: 2.01 seconds

## Relational analysis of NS_A1_B2_A1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5525086, upper bound: 1.5006464
time: 1.73 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.0870236, 0.8328964, -0.9604194, 0.7684886, -1.8555121, 1.7933158
1: -0.7848459, 0.9608730, -0.7156307, 0.9614974, -1.7463434, 1.6765037
2: -0.8271096, 1.0536442, -0.7364969, 0.9661483, -1.7932580, 1.7901411
3: -0.9545004, 0.7223951, -0.8170086, 0.6864815, -1.6409819, 1.5394037
4: -1.0145283, 0.9655470, -0.9133524, 0.8641971, -1.8787254, 1.8788993
5: -0.8336362, 1.1588053, -0.7603054, 1.1733613, -2.0069976, 1.9191107
6: -0.6825227, 0.8902621, -0.6091513, 0.8232130, -1.5057358, 1.4994135
7: -0.8121901, 1.0278324, -0.7351689, 0.9357533, -1.7479434, 1.7630012
8: -0.9324895, 0.9826777, -0.8388800, 0.9275126, -1.8600022, 1.8215578
9: -0.8883433, 0.9903239, -0.8337696, 0.9095037, -1.7978470, 1.8240936

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_A1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5525086, upper bound: 1.5006464
time: 1.64 seconds

## Relational analysis of NS_A1_B2_A1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5525086, upper bound: 1.5006464
time: 2.02 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.8007439, 0.6895599, -1.2177820, 0.8940181, -1.6947620, 1.9073420
1: -0.6186302, 0.7942876, -0.8714827, 1.1231222, -1.7417524, 1.6657703
2: -0.5981474, 0.8603787, -0.9513150, 1.1410803, -1.7392277, 1.8116938
3: -0.6306096, 0.6231076, -1.1199625, 0.7778771, -1.4084866, 1.7430701
4: -0.7721874, 0.7468568, -1.1408675, 1.0553061, -1.8274934, 1.8877243
5: -0.5268149, 1.1322763, -1.0336517, 1.1762013, -1.7030163, 2.1659279
6: -0.4996769, 0.7146878, -0.7795375, 0.9968133, -1.4964902, 1.4942253
7: -0.6225621, 0.8090602, -0.9091976, 1.1380233, -1.7605853, 1.7182577
8: -0.6827840, 0.8423395, -1.0805665, 1.0609418, -1.7437258, 1.9229060
9: -0.7017353, 0.8014120, -1.0312577, 1.0822264, -1.7839618, 1.8326697

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_A1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5495113, upper bound: 1.4883344
time: 1.70 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5495113, upper bound: 1.4883344
time: 1.80 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.8815992, 0.7307225, -1.2353903, 0.9028609, -1.7844601, 1.9661129
1: -0.6649849, 0.8468497, -0.8819233, 1.1373546, -1.8023396, 1.7287730
2: -0.6626682, 0.9148399, -0.9660013, 1.1530033, -1.8156716, 1.8808411
3: -0.7212399, 0.6522114, -1.1404033, 0.7845557, -1.5057956, 1.7926147
4: -0.8400771, 0.8085326, -1.1562454, 1.0683935, -1.9084706, 1.9647779
5: -0.6206321, 1.1562657, -1.0559487, 1.1813469, -1.8019791, 2.2122145
6: -0.5514464, 0.7636939, -0.7912719, 1.0084571, -1.5599035, 1.5549657
7: -0.6763610, 0.8709608, -0.9212539, 1.1518955, -1.8282566, 1.7922148
8: -0.7527007, 0.8824761, -1.0968750, 1.0702790, -1.8229797, 1.9793510
9: -0.7542017, 0.8548005, -1.0446663, 1.0940982, -1.8482999, 1.8994668

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_A2_B2_A2_A1

### Relational analysis result of NS_A1_B2_A1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5453651, upper bound: 1.4978764
time: 1.89 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_A2_A2

### Relational analysis result of NS_A1_B2_A1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5453651, upper bound: 1.4978764
time: 1.71 seconds

## BFS NS instance: NS_A2_B1_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.9834900, 0.7797163, -0.5263217, 0.5448294, -1.5283194, 1.3060380
1: -0.7296106, 0.9725906, -0.4573160, 0.6466167, -1.3762273, 1.4299066
2: -0.7555886, 0.9818274, -0.3826107, 0.6646317, -1.4202204, 1.3644381
3: -0.8440751, 0.6945100, -0.3286017, 0.5278014, -1.3718765, 1.0231117
4: -0.9336795, 0.8812118, -0.5462130, 0.5351171, -1.4687966, 1.4274248
5: -0.7820457, 1.1748499, -0.2489608, 1.1318836, -1.9139293, 1.4238107
6: -0.6243007, 0.8387851, -0.3253044, 0.5482060, -1.1725068, 1.1640894
7: -0.7505552, 0.9538813, -0.4359377, 0.5933949, -1.3439500, 1.3898189
8: -0.8604984, 0.9393537, -0.4451309, 0.7025202, -1.5630186, 1.3844845
9: -0.8513950, 0.9248503, -0.5160955, 0.6172245, -1.4686196, 1.4409459

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A2_B1_B1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5299768, upper bound: 1.5721794
time: 1.91 seconds

## Relational analysis of NS_A2_B1_B1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5299768, upper bound: 1.5817163
time: 1.84 seconds

## BFS NS instance: NS_A2_B1_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1.0021820, 0.7891003, -0.5672411, 0.5714372, -1.5736191, 1.3563415
1: -0.7406971, 0.9879798, -0.4807033, 0.6761887, -1.4168859, 1.4686830
2: -0.7711954, 0.9944850, -0.4134003, 0.6989219, -1.4701173, 1.4078853
3: -0.8657864, 0.7016091, -0.3698834, 0.5453974, -1.4111838, 1.0714926
4: -0.9500130, 0.8951169, -0.5775470, 0.5667447, -1.5167577, 1.4726639
5: -0.8058947, 1.1800755, -0.2994074, 1.1559496, -1.9618443, 1.4794829
6: -0.6367680, 0.8511494, -0.3482073, 0.5723276, -1.2090956, 1.1993567
7: -0.7633727, 0.9686069, -0.4675447, 0.6279905, -1.3913631, 1.4361516
8: -0.8778203, 0.9492729, -0.4768460, 0.7280012, -1.6058215, 1.4261189
9: -0.8656385, 0.9374646, -0.5463970, 0.6463131, -1.5119516, 1.4838617

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A2_B1_B1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5352767, upper bound: 1.5721794
time: 1.71 seconds

## Relational analysis of NS_A2_B1_B1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5352767, upper bound: 1.5817163
time: 1.67 seconds

## BFS NS instance: NS_A2_B1_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.3371168, 0.9519856, -0.5263217, 0.5448294, -1.8819462, 1.4783072
1: -0.9439206, 1.1873144, -0.4573160, 0.6466167, -1.5905373, 1.6446304
2: -1.0505031, 1.2222170, -0.3826107, 0.6646317, -1.7151349, 1.6048276
3: -1.2604107, 0.8195143, -0.3286017, 0.5278014, -1.7882121, 1.1481160
4: -1.2463251, 1.1435820, -0.5462130, 0.5351171, -1.7814423, 1.6897950
5: -1.1527361, 1.1773784, -0.2489608, 1.1318836, -2.2846198, 1.4263391
6: -0.8581581, 1.0774893, -0.3253044, 0.5482060, -1.4063642, 1.4027936
7: -0.9892256, 1.2317768, -0.4359377, 0.5933949, -1.5826205, 1.6677146
8: -1.1926479, 1.1223788, -0.4451309, 0.7025202, -1.8951681, 1.5675097
9: -1.1226774, 1.1619502, -0.5160955, 0.6172245, -1.7399020, 1.6780457

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 218

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A2_B1_B1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4986285, upper bound: 1.5550893
time: 2.20 seconds

## Relational analysis of NS_A2_B1_B1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4986285, upper bound: 1.5692598
time: 2.12 seconds

## BFS NS instance: NS_A2_B1_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.3567265, 0.9618111, -0.5672411, 0.5714372, -1.9281638, 1.5290523
1: -0.9555715, 1.2033789, -0.4807033, 0.6761887, -1.6317602, 1.6840823
2: -1.0668855, 1.2355019, -0.4134003, 0.6989219, -1.7658074, 1.6489022
3: -1.2832209, 0.8269311, -0.3698834, 0.5453974, -1.8286183, 1.1968145
4: -1.2634817, 1.1581748, -0.5775470, 0.5667447, -1.8302264, 1.7357217
5: -1.1776241, 1.1826038, -0.2994074, 1.1559496, -2.3335738, 1.4820113
6: -0.8712379, 1.0904816, -0.3482073, 0.5723276, -1.4435655, 1.4386890
7: -1.0026691, 1.2472235, -0.4675447, 0.6279905, -1.6306596, 1.7147682
8: -1.2108443, 1.1327718, -0.4768460, 0.7280012, -1.9388455, 1.6096179
9: -1.1376344, 1.1751842, -0.5463970, 0.6463131, -1.7839475, 1.7215812

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A2_B1_B1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5006464, upper bound: 1.5550893
time: 1.72 seconds

## Relational analysis of NS_A2_B1_B1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5006464, upper bound: 1.5692598
time: 1.79 seconds

## BFS NS instance: NS_A2_B1_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.9604194, 0.7684886, -0.8431016, 0.7113134, -1.6717329, 1.6115901
1: -0.7156307, 0.9614974, -0.6427077, 0.8172264, -1.5328571, 1.6042051
2: -0.7364969, 0.9661483, -0.6315987, 0.8888518, -1.6253487, 1.5977470
3: -0.8170086, 0.6864815, -0.6776149, 0.6384385, -1.4554471, 1.3640964
4: -0.9133524, 0.8641971, -0.8074291, 0.7789104, -1.6922628, 1.6716263
5: -0.7603054, 1.1733613, -0.5733571, 1.1524251, -1.9127305, 1.7467184
6: -0.6091513, 0.8232130, -0.5266141, 0.7401577, -1.3493090, 1.3498271
7: -0.7351689, 0.9357533, -0.6504493, 0.8415111, -1.5766799, 1.5862026
8: -0.8388800, 0.9275126, -0.7190910, 0.8633411, -1.7022212, 1.6466036
9: -0.8337696, 0.9095037, -0.7289797, 0.8291612, -1.6629307, 1.6384834

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_B1_B1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5006464, upper bound: 1.5525086
time: 1.72 seconds

## Relational analysis of NS_A2_B1_B1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5006464, upper bound: 1.5525086
time: 1.71 seconds

## BFS NS instance: NS_A2_B1_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.9604194, 0.7684886, -1.0870236, 0.8328964, -1.7933158, 1.8555121
1: -0.7156307, 0.9614974, -0.7848459, 0.9608730, -1.6765037, 1.7463434
2: -0.7364969, 0.9661483, -0.8271096, 1.0536442, -1.7901411, 1.7932580
3: -0.8170086, 0.6864815, -0.9545004, 0.7223951, -1.5394037, 1.6409819
4: -0.9133524, 0.8641971, -1.0145283, 0.9655470, -1.8788993, 1.8787254
5: -0.7603054, 1.1733613, -0.8336362, 1.1588053, -1.9191107, 2.0069976
6: -0.6091513, 0.8232130, -0.6825227, 0.8902621, -1.4994135, 1.5057358
7: -0.7351689, 0.9357533, -0.8121901, 1.0278324, -1.7630012, 1.7479434
8: -0.8388800, 0.9275126, -0.9324895, 0.9826777, -1.8215578, 1.8600022
9: -0.8337696, 0.9095037, -0.8883433, 0.9903239, -1.8240936, 1.7978470

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_B1_B1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5006464, upper bound: 1.5525086
time: 1.70 seconds

## Relational analysis of NS_A2_B1_B1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5006464, upper bound: 1.5525086
time: 1.64 seconds

## BFS NS instance: NS_A2_B1_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.2177820, 0.8940181, -0.8007439, 0.6895599, -1.9073420, 1.6947620
1: -0.8714827, 1.1231222, -0.6186302, 0.7942876, -1.6657703, 1.7417524
2: -0.9513150, 1.1410803, -0.5981474, 0.8603787, -1.8116938, 1.7392277
3: -1.1199625, 0.7778771, -0.6306096, 0.6231076, -1.7430701, 1.4084866
4: -1.1408675, 1.0553061, -0.7721874, 0.7468568, -1.8877243, 1.8274934
5: -1.0336517, 1.1762013, -0.5268149, 1.1322763, -2.1659279, 1.7030163
6: -0.7795375, 0.9968133, -0.4996769, 0.7146878, -1.4942253, 1.4964902
7: -0.9091976, 1.1380233, -0.6225621, 0.8090602, -1.7182577, 1.7605853
8: -1.0805665, 1.0609418, -0.6827840, 0.8423395, -1.9229060, 1.7437258
9: -1.0312577, 1.0822264, -0.7017353, 0.8014120, -1.8326697, 1.7839618

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_B1_B1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4883344, upper bound: 1.5495113
time: 1.72 seconds

## Relational analysis of NS_A2_B1_B1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4883344, upper bound: 1.5495113
time: 1.57 seconds

## BFS NS instance: NS_A2_B1_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.2353903, 0.9028609, -0.8815992, 0.7307225, -1.9661129, 1.7844601
1: -0.8819233, 1.1373546, -0.6649849, 0.8468497, -1.7287730, 1.8023396
2: -0.9660013, 1.1530033, -0.6626682, 0.9148399, -1.8808411, 1.8156716
3: -1.1404033, 0.7845557, -0.7212399, 0.6522114, -1.7926147, 1.5057956
4: -1.1562454, 1.0683935, -0.8400771, 0.8085326, -1.9647779, 1.9084706
5: -1.0559487, 1.1813469, -0.6206321, 1.1562657, -2.2122145, 1.8019791
6: -0.7912719, 1.0084571, -0.5514464, 0.7636939, -1.5549657, 1.5599035
7: -0.9212539, 1.1518955, -0.6763610, 0.8709608, -1.7922148, 1.8282566
8: -1.0968750, 1.0702790, -0.7527007, 0.8824761, -1.9793510, 1.8229797
9: -1.0446663, 1.0940982, -0.7542017, 0.8548005, -1.8994668, 1.8482999

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_B1_B2_A2_B2_B1

### Relational analysis result of NS_A2_B1_B1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4978764, upper bound: 1.5495113
time: 1.75 seconds

## Relational analysis of NS_A2_B1_B1_B2_A2_B2_B2

### Relational analysis result of NS_A2_B1_B1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4978764, upper bound: 1.5495113
time: 1.79 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.9621096, 0.7688487, -0.9132454, 0.7449896, -1.7070992, 1.6820941
1: -0.7169495, 0.9373744, -0.6874459, 0.9164211, -1.6333705, 1.6248202
2: -0.7370105, 0.9673195, -0.6966894, 0.9341465, -1.6711571, 1.6640089
3: -0.8188409, 0.6856422, -0.7617079, 0.6685639, -1.4874048, 1.4473500
4: -0.9147004, 0.8647465, -0.8717791, 0.8288034, -1.7435038, 1.7365255
5: -0.7417467, 1.1738902, -0.6961297, 1.1671618, -1.9089085, 1.8700199
6: -0.6094728, 0.8246655, -0.5774449, 0.7917952, -1.4012680, 1.4021103
7: -0.7349455, 0.9369996, -0.7024334, 0.8986073, -1.6335528, 1.6394329
8: -0.8404368, 0.9275126, -0.7948046, 0.9024062, -1.7428430, 1.7223172
9: -0.8347179, 0.9098628, -0.7975066, 0.8774052, -1.7121230, 1.7073693

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5715006, upper bound: 1.5352260
time: 2.16 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5715006, upper bound: 1.5352260
time: 2.07 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.9591036, 0.7673466, -1.1882548, 0.8793857, -1.8384893, 1.9556015
1: -0.7151568, 0.9342970, -0.8537580, 1.0929241, -1.8080809, 1.7880549
2: -0.7344663, 0.9652805, -0.9262465, 1.1210272, -1.8554935, 1.8915271
3: -0.8153164, 0.6844870, -1.0851533, 0.7666801, -1.5819966, 1.7696403
4: -0.9120502, 0.8624848, -1.1147105, 1.0330446, -1.9450948, 1.9771953
5: -0.7374986, 1.1735215, -0.9930698, 1.1749247, -1.9124234, 2.1665912
6: -0.6074456, 0.8226665, -0.7596078, 0.9770604, -1.5845060, 1.5822743
7: -0.7328478, 0.9346304, -0.8885745, 1.1147792, -1.8476269, 1.8232049
8: -0.8376286, 0.9259037, -1.0528387, 1.0452085, -1.8828371, 1.9787424
9: -0.8324047, 0.9078097, -1.0084419, 1.0620419, -1.8944466, 1.9162517

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5748945, upper bound: 1.5352260
time: 2.03 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5748945, upper bound: 1.5352260
time: 1.87 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.3158797, 0.9411994, -0.9132454, 0.7449896, -2.0608692, 1.8544449
1: -0.9313318, 1.1543211, -0.6874459, 0.9164211, -1.8477528, 1.8417671
2: -1.0320901, 1.2078011, -0.6966894, 0.9341465, -1.9662366, 1.9044905
3: -1.2353444, 0.8107843, -0.7617079, 0.6685639, -1.9039083, 1.5724921
4: -1.2274733, 1.1272736, -0.8717791, 0.8288034, -2.0562768, 1.9990526
5: -1.1154729, 1.1757748, -0.6961297, 1.1671618, -2.2826347, 1.8719046
6: -0.8434699, 1.0634512, -0.5774449, 0.7917952, -1.6352651, 1.6408961
7: -0.9737896, 1.2150052, -0.7024334, 0.8986073, -1.8723968, 1.9174385
8: -1.1727306, 1.1106406, -0.7948046, 0.9024062, -2.0751367, 1.9054452
9: -1.1061393, 1.1470826, -0.7975066, 0.8774052, -1.9835445, 1.9445891

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5525538, upper bound: 1.5318694
time: 1.74 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5525538, upper bound: 1.5324480
time: 1.73 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.3128049, 0.9396645, -1.1882548, 0.8793857, -2.1921906, 2.1279192
1: -0.9294958, 1.1512567, -0.8537580, 1.0929241, -2.0224199, 2.0050147
2: -1.0294888, 1.2057146, -0.9262465, 1.1210272, -2.1505160, 2.1319611
3: -1.2317371, 0.8096072, -1.0851533, 0.7666801, -1.9984173, 1.8947606
4: -1.2247617, 1.1249619, -1.1147105, 1.0330446, -2.2578063, 2.2396722
5: -1.1112499, 1.1753829, -0.9930698, 1.1749247, -2.2861748, 2.1684527
6: -0.8413973, 1.0614045, -0.7596078, 0.9770604, -1.8184578, 1.8210123
7: -0.9716465, 1.2125828, -0.8885745, 1.1147792, -2.0864258, 2.1011572
8: -1.1698570, 1.1089978, -1.0528387, 1.0452085, -2.2150655, 2.1618366
9: -1.1037731, 1.1449842, -1.0084419, 1.0620419, -2.1658149, 2.1534262

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5561671, upper bound: 1.5318694
time: 1.84 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5561671, upper bound: 1.5324480
time: 2.07 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.9258955, 0.7510032, -1.2608913, 0.9143122, -1.8402077, 2.0118945
1: -0.6951942, 0.9156294, -0.8981512, 1.1261199, -1.8213141, 1.8137805
2: -0.7069680, 0.9427447, -0.9865626, 1.1704710, -1.8774390, 1.9293073
3: -0.7765459, 0.6726069, -1.1709890, 0.7913750, -1.5679209, 1.8435960
4: -0.8829145, 0.8379726, -1.1791251, 1.0867043, -1.9696188, 2.0170977
5: -0.7031187, 1.1678039, -1.0597894, 1.1690992, -1.8722179, 2.2275934
6: -0.5855679, 0.8004208, -0.8073027, 1.0264828, -1.6120508, 1.6077235
7: -0.7105628, 0.9085158, -0.9369900, 1.1717957, -1.8823586, 1.8455058
8: -0.8066685, 0.9086926, -1.1213423, 1.0822730, -1.8889415, 2.0300350
9: -0.8071000, 0.8856572, -1.0641882, 1.1104386, -1.9175386, 1.9498454

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5562784, upper bound: 1.4978764
time: 1.86 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5562784, upper bound: 1.4978764
time: 1.78 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.9258955, 0.7510032, -1.5433918, 1.0523913, -1.9782867, 2.2943950
1: -0.6951942, 0.9156294, -1.0689799, 1.3093724, -2.0045667, 1.9846092
2: -0.7069680, 0.9427447, -1.2224125, 1.3624423, -2.0694103, 2.1651573
3: -0.7765459, 0.6726069, -1.5032526, 0.8922273, -1.6687732, 2.1758595
4: -0.8829145, 0.8379726, -1.4286811, 1.2965393, -2.1794538, 2.2666538
5: -0.7031187, 1.1678039, -1.3681687, 1.1769347, -1.8800534, 2.5359726
6: -0.5855679, 0.8004208, -0.9944628, 1.2167815, -1.8023493, 1.7948836
7: -0.7105628, 0.9085158, -1.1282617, 1.3938599, -2.1044226, 2.0367775
8: -0.8066685, 0.9086926, -1.3864056, 1.2290050, -2.0356734, 2.2950983
9: -0.8071000, 0.8856572, -1.2808843, 1.3001393, -2.1072392, 2.1665416

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5562784, upper bound: 1.4978764
time: 1.82 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5562784, upper bound: 1.4978764
time: 1.73 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.2015018, 0.8856845, -1.2608913, 0.9143122, -2.1158142, 2.1465759
1: -0.8618726, 1.0929892, -0.8981512, 1.1261199, -1.9879924, 1.9911404
2: -0.9370291, 1.1300316, -0.9865626, 1.1704710, -2.1075001, 2.1165943
3: -1.1007035, 0.7709315, -1.1709890, 0.7913750, -1.8920785, 1.9419205
4: -1.1263803, 1.0426614, -1.1791251, 1.0867043, -2.2130847, 2.2217865
5: -1.0014402, 1.1752590, -1.0597894, 1.1690992, -2.1705394, 2.2350483
6: -0.7681288, 0.9860938, -0.8073027, 1.0264828, -1.7946117, 1.7933965
7: -0.8971119, 1.1251557, -0.9369900, 1.1717957, -2.0689077, 2.0621457
8: -1.0652705, 1.0518020, -1.1213423, 1.0822730, -2.1475434, 2.1731443
9: -1.0184999, 1.0706968, -1.0641882, 1.1104386, -2.1289384, 2.1348851

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5465795, upper bound: 1.4978764
time: 2.07 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5465795, upper bound: 1.4978764
time: 1.98 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.2015018, 0.8856845, -1.5433918, 1.0523913, -2.2538931, 2.4290762
1: -0.8618726, 1.0929892, -1.0689799, 1.3093724, -2.1712451, 2.1619692
2: -0.9370291, 1.1300316, -1.2224125, 1.3624423, -2.2994714, 2.3524442
3: -1.1007035, 0.7709315, -1.5032526, 0.8922273, -1.9929308, 2.2741842
4: -1.1263803, 1.0426614, -1.4286811, 1.2965393, -2.4229198, 2.4713426
5: -1.0014402, 1.1752590, -1.3681687, 1.1769347, -2.1783748, 2.5434277
6: -0.7681288, 0.9860938, -0.9944628, 1.2167815, -1.9849102, 1.9805566
7: -0.8971119, 1.1251557, -1.1282617, 1.3938599, -2.2909718, 2.2534175
8: -1.0652705, 1.0518020, -1.3864056, 1.2290050, -2.2942755, 2.4382076
9: -1.0184999, 1.0706968, -1.2808843, 1.3001393, -2.3186393, 2.3515811

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5465795, upper bound: 1.4978764
time: 2.01 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5465795, upper bound: 1.4978764
time: 1.93 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 5.88 seconds
NS_A1_B1_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5439451, upper bound: 1.5190737
NS_A1_B1_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5439451, upper bound: 1.5190737
NS_A1_B1_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5457925, upper bound: 1.5183453
NS_A1_B1_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5457925, upper bound: 1.5183453
NS_A1_B1_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5586556, upper bound: 1.5366048
NS_A1_B1_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5586556, upper bound: 1.5366049
NS_A1_B1_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5586556, upper bound: 1.5366048
NS_A1_B1_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5586556, upper bound: 1.5366049
NS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5456782, upper bound: 1.4927321
NS_A1_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5376395, upper bound: 1.4929717
NS_A1_B1_A1_B2_A1_B2_B1, status: Status.VERIFIED, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5364309, upper bound: 1.4948836
NS_A1_B1_A1_B2_A1_B2_B2, status: Status.VERIFIED, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5376395, upper bound: 1.4929717
NS_A1_B1_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5456782, upper bound: 1.5016473
NS_A1_B1_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5456782, upper bound: 1.5018314
NS_A1_B1_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5470452, upper bound: 1.5018314
NS_A1_B1_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5470452, upper bound: 1.5018314
NS_A1_B2_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5721794, upper bound: 1.5299768
NS_A1_B2_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5721794, upper bound: 1.5299768
NS_A1_B2_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5721794, upper bound: 1.5352767
NS_A1_B2_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5721794, upper bound: 1.5352767
NS_A1_B2_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5550893, upper bound: 1.4986285
NS_A1_B2_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5550893, upper bound: 1.4986285
NS_A1_B2_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5550893, upper bound: 1.5006464
NS_A1_B2_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5550893, upper bound: 1.5006464
NS_A1_B2_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5525086, upper bound: 1.5006464
NS_A1_B2_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5525086, upper bound: 1.5006464
NS_A1_B2_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5525086, upper bound: 1.5006464
NS_A1_B2_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5525086, upper bound: 1.5006464
NS_A1_B2_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5495113, upper bound: 1.4883344
NS_A1_B2_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5495113, upper bound: 1.4883344
NS_A1_B2_A1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5453651, upper bound: 1.4978764
NS_A1_B2_A1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5453651, upper bound: 1.4978764
NS_A2_B1_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5299768, upper bound: 1.5721794
NS_A2_B1_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5299768, upper bound: 1.5817163
NS_A2_B1_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5352767, upper bound: 1.5721794
NS_A2_B1_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5352767, upper bound: 1.5817163
NS_A2_B1_B1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.4986285, upper bound: 1.5550893
NS_A2_B1_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.4986285, upper bound: 1.5692598
NS_A2_B1_B1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5006464, upper bound: 1.5550893
NS_A2_B1_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5006464, upper bound: 1.5692598
NS_A2_B1_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5006464, upper bound: 1.5525086
NS_A2_B1_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5006464, upper bound: 1.5525086
NS_A2_B1_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5006464, upper bound: 1.5525086
NS_A2_B1_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5006464, upper bound: 1.5525086
NS_A2_B1_B1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.4883344, upper bound: 1.5495113
NS_A2_B1_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.4883344, upper bound: 1.5495113
NS_A2_B1_B1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.4978764, upper bound: 1.5495113
NS_A2_B1_B1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.4978764, upper bound: 1.5495113
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5715006, upper bound: 1.5352260
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5715006, upper bound: 1.5352260
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5748945, upper bound: 1.5352260
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5748945, upper bound: 1.5352260
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5525538, upper bound: 1.5318694
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5525538, upper bound: 1.5324480
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5561671, upper bound: 1.5318694
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5561671, upper bound: 1.5324480
NS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5562784, upper bound: 1.4978764
NS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5562784, upper bound: 1.4978764
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5562784, upper bound: 1.4978764
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5562784, upper bound: 1.4978764
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5465795, upper bound: 1.4978764
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5465795, upper bound: 1.4978764
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5465795, upper bound: 1.4978764
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 5, lower bound: -1.5465795, upper bound: 1.4978764

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.5615433, 0.5677513, -0.4982012, 0.5266388, -1.0881821, 1.0659525
1: -0.4776802, 0.6888618, -0.4393089, 0.6391180, -1.1167982, 1.1281707
2: -0.4101332, 0.6945543, -0.3628579, 0.6372288, -1.0473620, 1.0574123
3: -0.3651401, 0.5432546, -0.2996235, 0.5159649, -0.8811049, 0.8428780
4: -0.5736591, 0.5631060, -0.5259902, 0.5122890, -1.0859480, 1.0890963
5: -0.3050084, 1.1391411, -0.2269045, 1.1246979, -1.4297063, 1.3660455
6: -0.3452637, 0.5692508, -0.3113982, 0.5302579, -0.8755217, 0.8806491
7: -0.4644209, 0.6226522, -0.4154138, 0.5707872, -1.0352080, 1.0380660
8: -0.4728004, 0.7250535, -0.4231856, 0.6839871, -1.1567876, 1.1482391
9: -0.5429381, 0.6429994, -0.4951464, 0.5966613, -1.1395993, 1.1381457

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5046981, upper bound: 1.5086700
time: 1.74 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5045174, upper bound: 1.4680235
time: 1.96 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.5615433, 0.5677513, -0.5356421, 0.5511143, -1.1126575, 1.1033934
1: -0.4776802, 0.6888618, -0.4620029, 0.6683220, -1.1460023, 1.1508647
2: -0.4101332, 0.6945543, -0.3897382, 0.6721959, -1.0823290, 1.0842925
3: -0.3651401, 0.5432546, -0.3375186, 0.5330602, -0.8982003, 0.8807732
4: -0.5736591, 0.5631060, -0.5526587, 0.5424715, -1.1161306, 1.1157647
5: -0.3050084, 1.1391411, -0.2753383, 1.1487031, -1.4537115, 1.4144794
6: -0.3452637, 0.5692508, -0.3307454, 0.5530482, -0.8983119, 0.8999963
7: -0.4644209, 0.6226522, -0.4436026, 0.6033349, -1.0677557, 1.0662549
8: -0.4728004, 0.7250535, -0.4517691, 0.7089653, -1.1817658, 1.1768227
9: -0.5429381, 0.6429994, -0.5227286, 0.6240724, -1.1670105, 1.1657280

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 131

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5439451, upper bound: 1.5190611
time: 1.68 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5439451, upper bound: 1.5190737
time: 1.79 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.5597346, 0.5665687, -0.6698226, 0.6252810, -1.1850156, 1.2363913
1: -0.4765722, 0.6866383, -0.5425203, 0.7405922, -1.2171645, 1.2291586
2: -0.4086696, 0.6930238, -0.4937863, 0.7720163, -1.1806860, 1.1868100
3: -0.3632936, 0.5424174, -0.4803110, 0.5790600, -0.9423536, 1.0227284
4: -0.5721267, 0.5616771, -0.6599021, 0.6478885, -1.2200152, 1.2215792
5: -0.3020072, 1.1387842, -0.4079604, 1.1312654, -1.4332726, 1.5467446
6: -0.3441997, 0.5680857, -0.4172930, 0.6328434, -0.9770432, 0.9853787
7: -0.4629831, 0.6210673, -0.5376159, 0.7094426, -1.1724257, 1.1586832
8: -0.4712815, 0.7239193, -0.5680908, 0.7792792, -1.2505608, 1.2920101
9: -0.5415809, 0.6416145, -0.6161131, 0.7156835, -1.2572644, 1.2577276

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5439451, upper bound: 1.5182364
time: 2.09 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5439451, upper bound: 1.5183453
time: 2.10 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.5597346, 0.5665687, -0.7474035, 0.6664491, -1.2261837, 1.3139722
1: -0.4765722, 0.6866383, -0.5884170, 0.7897847, -1.2663569, 1.2750553
2: -0.4086696, 0.6930238, -0.5543652, 0.8251685, -1.2338381, 1.2473891
3: -0.3632936, 0.5424174, -0.5614092, 0.6055813, -0.9688749, 1.1038266
4: -0.5721267, 0.5616771, -0.7202846, 0.7089003, -1.2810271, 1.2819617
5: -0.3020072, 1.1387842, -0.4974139, 1.1555154, -1.4575226, 1.6361980
6: -0.3441997, 0.5680857, -0.4687940, 0.6765890, -1.0207887, 1.0368798
7: -0.4629831, 0.6210673, -0.5913229, 0.7698932, -1.2328763, 1.2123902
8: -0.4712815, 0.7239193, -0.6353303, 0.8181738, -1.2894553, 1.3592496
9: -0.5415809, 0.6416145, -0.6660843, 0.7668741, -1.3084550, 1.3076987

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5439451, upper bound: 1.5182364
time: 1.96 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5439451, upper bound: 1.5183453
time: 1.77 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.5672411, 0.5714372, -0.5476493, 0.5585920, -1.1258332, 1.1190865
1: -0.4807033, 0.6761887, -0.4694499, 0.6849468, -1.1656501, 1.1456387
2: -0.4134003, 0.6989219, -0.3994535, 0.6826859, -1.0960861, 1.0983754
3: -0.3698834, 0.5453974, -0.3508350, 0.5376303, -0.9075137, 0.8962324
4: -0.5775470, 0.5667447, -0.5625946, 0.5522501, -1.1297971, 1.1293393
5: -0.2994074, 1.1559496, -0.2939460, 1.1369205, -1.4363279, 1.4498956
6: -0.3482073, 0.5723276, -0.3375167, 0.5606839, -0.9088912, 0.9098443
7: -0.4675447, 0.6279905, -0.4536663, 0.6118128, -1.0793575, 1.0816568
8: -0.4768460, 0.7280012, -0.4617271, 0.7164679, -1.1933138, 1.1897284
9: -0.5463970, 0.6463131, -0.5324571, 0.6329035, -1.1793005, 1.1787702

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5680830, upper bound: 1.5412051
time: 1.78 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5654372, upper bound: 1.5420231
time: 2.06 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.5672411, 0.5714372, -0.5947111, 0.5875352, -1.1547763, 1.1661482
1: -0.4807033, 0.6761887, -0.4965092, 0.7185460, -1.1992493, 1.1726979
2: -0.4134003, 0.6989219, -0.4351903, 0.7194800, -1.1328803, 1.1341121
3: -0.3698834, 0.5453974, -0.3992412, 0.5572081, -0.9270915, 0.9446386
4: -0.5775470, 0.5667447, -0.5995938, 0.5889173, -1.1664643, 1.1663386
5: -0.2994074, 1.1559496, -0.3526397, 1.1609485, -1.4603560, 1.5085893
6: -0.3482073, 0.5723276, -0.3676384, 0.5881414, -0.9363487, 0.9399661
7: -0.4675447, 0.6279905, -0.4873933, 0.6518717, -1.1194165, 1.1153839
8: -0.4768460, 0.7280012, -0.5006096, 0.7435498, -1.2203958, 1.2286108
9: -0.5463970, 0.6463131, -0.5665323, 0.6662003, -1.2125974, 1.2128453

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A2_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5633484, upper bound: 1.5423846
time: 2.07 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5654372, upper bound: 1.5420562
time: 1.99 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.8724293, 0.7321481, -0.5476493, 0.5585920, -1.4310212, 1.2797973
1: -0.6597273, 0.8400022, -0.4694499, 0.6849468, -1.3446741, 1.3094521
2: -0.6493913, 0.9130070, -0.3994535, 0.6826859, -1.3320773, 1.3124605
3: -0.6967273, 0.6454272, -0.3508350, 0.5376303, -1.2343576, 0.9962622
4: -0.8154522, 0.8091797, -0.5625946, 0.5522501, -1.3677022, 1.3717743
5: -0.6127775, 1.1567936, -0.2939460, 1.1369205, -1.7496979, 1.4507396
6: -0.5527496, 0.7497519, -0.3375167, 0.5606839, -1.1134335, 1.0872686
7: -0.6721511, 0.8690442, -0.4536663, 0.6118128, -1.2839639, 1.3227105
8: -0.7415717, 0.8802458, -0.4617271, 0.7164679, -1.4580395, 1.3419729
9: -0.7481984, 0.8504668, -0.5324571, 0.6329035, -1.3811018, 1.3829240

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5538759, upper bound: 1.5314934
time: 1.90 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5457925, upper bound: 1.5321150
time: 1.69 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.8724293, 0.7321481, -0.5947111, 0.5875352, -1.4599645, 1.3268591
1: -0.6597273, 0.8400022, -0.4965092, 0.7185460, -1.3782732, 1.3365114
2: -0.6493913, 0.9130070, -0.4351903, 0.7194800, -1.3688713, 1.3481972
3: -0.6967273, 0.6454272, -0.3992412, 0.5572081, -1.2539355, 1.0446684
4: -0.8154522, 0.8091797, -0.5995938, 0.5889173, -1.4043695, 1.4087735
5: -0.6127775, 1.1567936, -0.3526397, 1.1609485, -1.7737260, 1.5094333
6: -0.5527496, 0.7497519, -0.3676384, 0.5881414, -1.1408910, 1.1173904
7: -0.6721511, 0.8690442, -0.4873933, 0.6518717, -1.3240229, 1.3564374
8: -0.7415717, 0.8802458, -0.5006096, 0.7435498, -1.4851215, 1.3808553
9: -0.7481984, 0.8504668, -0.5665323, 0.6662003, -1.4143987, 1.4169991

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A2_A2_B2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5439451, upper bound: 1.5326515
time: 1.98 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_A2_B2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5457925, upper bound: 1.5321564
time: 2.02 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.5043612, 0.5308766, -0.8391411, 0.7086776, -1.2130388, 1.3700178
1: -0.4434332, 0.6343944, -0.6408284, 0.8393061, -1.2827392, 1.2752228
2: -0.3669876, 0.6438618, -0.6304567, 0.8842752, -1.2512629, 1.2743185
3: -0.3059967, 0.5183336, -0.6733575, 0.6368105, -0.9428072, 1.1916912
4: -0.5303541, 0.5172405, -0.8029805, 0.7769719, -1.3073261, 1.3202211
5: -0.2262707, 1.1255345, -0.5907815, 1.1382631, -1.3645338, 1.7163160
6: -0.3141770, 0.5344315, -0.5232665, 0.7368669, -1.0510439, 1.0576980
7: -0.4195183, 0.5752821, -0.6494476, 0.8368269, -1.2563453, 1.2247297
8: -0.4279660, 0.6881204, -0.7222785, 0.8605849, -1.2885509, 1.4103990
9: -0.4996428, 0.6012410, -0.7319579, 0.8255336, -1.3251765, 1.3331988

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5050793, upper bound: 1.4830186
time: 1.58 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5048801, upper bound: 1.4452378
time: 1.76 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.5435179, 0.5559961, -0.8391411, 0.7086776, -1.2521956, 1.3951372
1: -0.4667313, 0.6630547, -0.6408284, 0.8393061, -1.3060374, 1.3038831
2: -0.3953950, 0.6789523, -0.6304567, 0.8842752, -1.2796702, 1.3094089
3: -0.3456593, 0.5356200, -0.6733575, 0.6368105, -0.9824698, 1.2089775
4: -0.5587332, 0.5483223, -0.8029805, 0.7769719, -1.3357050, 1.3513029
5: -0.2748681, 1.1494682, -0.5907815, 1.1382631, -1.4131312, 1.7402496
6: -0.3347107, 0.5578873, -0.5232665, 0.7368669, -1.0715777, 1.0811538
7: -0.4493440, 0.6087279, -0.6494476, 0.8368269, -1.2861710, 1.2581754
8: -0.4579076, 0.7135138, -0.7222785, 0.8605849, -1.3184924, 1.4357923
9: -0.5286373, 0.6293439, -0.7319579, 0.8255336, -1.3541710, 1.3613017

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_A2_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5456782, upper bound: 1.5016473
time: 2.02 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5456782, upper bound: 1.5016473
time: 2.00 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.5435179, 0.5559961, -0.9235446, 0.7514418, -1.2949597, 1.4795407
1: -0.4667313, 0.6630547, -0.6892793, 0.8944167, -1.3611480, 1.3523340
2: -0.3953950, 0.6789523, -0.6982160, 0.9401498, -1.3355448, 1.3771683
3: -0.3456593, 0.5356200, -0.7675613, 0.6668150, -1.0124743, 1.3031812
4: -0.5587332, 0.5483223, -0.8729563, 0.8414344, -1.4001676, 1.4212785
5: -0.2748681, 1.1494682, -0.6911302, 1.1622884, -1.4371566, 1.8405983
6: -0.3347107, 0.5578873, -0.5765235, 0.7874798, -1.1221906, 1.1344109
7: -0.4493440, 0.6087279, -0.7057423, 0.9006615, -1.3500054, 1.3144702
8: -0.4579076, 0.7135138, -0.7981097, 0.9018514, -1.3597590, 1.5116235
9: -0.5286373, 0.6293439, -0.7890604, 0.8803588, -1.4089961, 1.4184042

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_A2_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5456782, upper bound: 1.5018314
time: 1.89 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5456782, upper bound: 1.5018314
time: 1.89 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.7497785, 0.6680276, -0.8813014, 0.7302631, -1.4800416, 1.5493290
1: -0.5841198, 0.7865181, -0.6647936, 0.8618708, -1.4459906, 1.4513117
2: -0.5511015, 0.8313252, -0.6639413, 0.9121257, -1.4632273, 1.4952664
3: -0.5593077, 0.6087750, -0.7199010, 0.6519264, -1.2112341, 1.3286760
4: -0.7247543, 0.7049389, -0.8375885, 0.8089012, -1.5336554, 1.5425274
5: -0.4981094, 1.1559100, -0.6384306, 1.1578932, -1.6560025, 1.7943406
6: -0.4712533, 0.6799220, -0.5496904, 0.7619194, -1.2331727, 1.2296124
7: -0.5871033, 0.7772533, -0.6772745, 0.8687456, -1.4558489, 1.4545277
8: -0.6400384, 0.8188547, -0.7597872, 0.8812040, -1.5212424, 1.5786419
9: -0.6703289, 0.7684357, -0.7602102, 0.8527094, -1.5230384, 1.5286459

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B1_A1_B2_A2_A2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5364309, upper bound: 1.5016473
time: 1.84 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_A2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5364309, upper bound: 1.5018314
time: 1.75 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.7497785, 0.6680276, -1.1288611, 0.8528860, -1.6026645, 1.7968887
1: -0.5841198, 0.7865181, -0.8093686, 1.0043435, -1.5884633, 1.5958867
2: -0.5511015, 0.8313252, -0.8634650, 1.0764867, -1.6275883, 1.6947901
3: -0.5593077, 0.6087750, -0.9998003, 0.7356546, -1.2949623, 1.6085753
4: -0.7247543, 0.7049389, -1.0451580, 0.9984848, -1.7232392, 1.7500970
5: -0.4981094, 1.1559100, -0.9056400, 1.1648138, -1.6629231, 2.0615501
6: -0.4712533, 0.6799220, -0.7054486, 0.9127687, -1.3840220, 1.3853706
7: -0.5871033, 0.7772533, -0.8415911, 1.0554829, -1.6425862, 1.6188443
8: -0.6400384, 0.8188547, -0.9849553, 1.0002134, -1.6402518, 1.8038101
9: -0.6703289, 0.7684357, -0.9289026, 1.0134271, -1.6837561, 1.6973383

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A1_B2_A2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5470452, upper bound: 1.5018314
time: 2.16 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5470452, upper bound: 1.5018314
time: 1.71 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.5263217, 0.5448294, -0.9531180, 0.7638087, -1.2901304, 1.4979473
1: -0.4573160, 0.6466167, -0.7122197, 0.9460655, -1.4033816, 1.3588364
2: -0.3826107, 0.6646317, -0.7305958, 0.9614030, -1.3440137, 1.3952276
3: -0.3286017, 0.5278014, -0.8098066, 0.6820929, -1.0106945, 1.3376080
4: -0.5462130, 0.5351171, -0.9078164, 0.8588266, -1.4050395, 1.4429336
5: -0.2489608, 1.1318836, -0.7386829, 1.1507246, -1.3996854, 1.8705665
6: -0.3253044, 0.5482060, -0.6040805, 0.8193476, -1.1446519, 1.1522865
7: -0.4359377, 0.5933949, -0.7297332, 0.9298794, -1.3658171, 1.3231281
8: -0.4451309, 0.7025202, -0.8331056, 0.9228835, -1.3680143, 1.5356258
9: -0.5160955, 0.6172245, -0.8287107, 0.9044701, -1.4205656, 1.4459352

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5681235, upper bound: 1.5257456
time: 1.98 seconds

## Relational analysis of NS_A1_B2_A1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5651700, upper bound: 1.5261914
time: 1.68 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.5263217, 0.5448294, -0.9561330, 0.7661559, -1.2924776, 1.5009624
1: -0.4573160, 0.6466167, -0.7132269, 0.9508001, -1.4081161, 1.3598435
2: -0.3826107, 0.6646317, -0.7326699, 0.9632682, -1.3458788, 1.3973017
3: -0.3286017, 0.5278014, -0.8120513, 0.6843585, -1.0129602, 1.3398527
4: -0.5462130, 0.5351171, -0.9096080, 0.8608177, -1.4070307, 1.4447252
5: -0.2489608, 1.1318836, -0.7484537, 1.1710870, -1.4200478, 1.8803374
6: -0.3253044, 0.5482060, -0.6060559, 0.8205242, -1.1458285, 1.1542619
7: -0.4359377, 0.5933949, -0.7318144, 0.9323517, -1.3682895, 1.3252093
8: -0.4451309, 0.7025202, -0.8349611, 0.9249382, -1.3700690, 1.5374813
9: -0.5160955, 0.6172245, -0.8304380, 0.9063733, -1.4224687, 1.4476625

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_A1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B2_A1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5628257, upper bound: 1.5262310
time: 2.01 seconds

## Relational analysis of NS_A1_B2_A1_A1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B2_A1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5651700, upper bound: 1.5261914
time: 1.92 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.5672411, 0.5714372, -0.9531180, 0.7638087, -1.3310499, 1.5245552
1: -0.4807033, 0.6761887, -0.7122197, 0.9460655, -1.4267688, 1.3884084
2: -0.4134003, 0.6989219, -0.7305958, 0.9614030, -1.3748033, 1.4295177
3: -0.3698834, 0.5453974, -0.8098066, 0.6820929, -1.0519763, 1.3552040
4: -0.5775470, 0.5667447, -0.9078164, 0.8588266, -1.4363735, 1.4745612
5: -0.2994074, 1.1559496, -0.7386829, 1.1507246, -1.4501321, 1.8946325
6: -0.3482073, 0.5723276, -0.6040805, 0.8193476, -1.1675549, 1.1764081
7: -0.4675447, 0.6279905, -0.7297332, 0.9298794, -1.3974241, 1.3577237
8: -0.4768460, 0.7280012, -0.8331056, 0.9228835, -1.3997295, 1.5611069
9: -0.5463970, 0.6463131, -0.8287107, 0.9044701, -1.4508672, 1.4750237

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5681234, upper bound: 1.5350282
time: 1.62 seconds

## Relational analysis of NS_A1_B2_A1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5651700, upper bound: 1.5352260
time: 1.98 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.5672411, 0.5714372, -0.9561330, 0.7661559, -1.3333970, 1.5275702
1: -0.4807033, 0.6761887, -0.7132269, 0.9508001, -1.4315033, 1.3894155
2: -0.4134003, 0.6989219, -0.7326699, 0.9632682, -1.3766685, 1.4315919
3: -0.3698834, 0.5453974, -0.8120513, 0.6843585, -1.0542419, 1.3574487
4: -0.5775470, 0.5667447, -0.9096080, 0.8608177, -1.4383646, 1.4763527
5: -0.2994074, 1.1559496, -0.7484537, 1.1710870, -1.4704945, 1.9044033
6: -0.3482073, 0.5723276, -0.6060559, 0.8205242, -1.1687315, 1.1783836
7: -0.4675447, 0.6279905, -0.7318144, 0.9323517, -1.3998964, 1.3598049
8: -0.4768460, 0.7280012, -0.8349611, 0.9249382, -1.4017842, 1.5629623
9: -0.5463970, 0.6463131, -0.8304380, 0.9063733, -1.4527702, 1.4767511

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5681234, upper bound: 1.5350282
time: 9.89 seconds

## Relational analysis of NS_A1_B2_A1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5651700, upper bound: 1.5352260
time: 2.21 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.5263217, 0.5448294, -1.2890378, 0.9274813, -1.4538031, 1.8338672
1: -0.4573160, 0.6466167, -0.9157523, 1.1495152, -1.6068311, 1.5623690
2: -0.3826107, 0.6646317, -1.0107118, 1.1897380, -1.5723487, 1.6753435
3: -0.3286017, 0.5278014, -1.2052171, 0.8008965, -1.1294982, 1.7330185
4: -0.5462130, 0.5351171, -1.2047541, 1.1080422, -1.6542553, 1.7398713
5: -0.2489608, 1.1318836, -1.0903399, 1.1533077, -1.4022684, 2.2222235
6: -0.3253044, 0.5482060, -0.8262197, 1.0460510, -1.3713554, 1.3744256
7: -0.4359377, 0.5933949, -0.9564478, 1.1938564, -1.6297941, 1.5498427
8: -0.4451309, 0.7025202, -1.1485672, 1.0967597, -1.5418906, 1.8510873
9: -0.5160955, 0.6172245, -1.0863751, 1.1296828, -1.6457782, 1.7035997

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 218

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5506546, upper bound: 1.4921119
time: 4.33 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5437171, upper bound: 1.4925852
time: 1.88 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.5263217, 0.5448294, -1.3084635, 0.9377898, -1.4641116, 1.8532928
1: -0.4573160, 0.6466167, -0.9267504, 1.1646020, -1.6219180, 1.5733671
2: -0.3826107, 0.6646317, -1.0264972, 1.2027751, -1.5853858, 1.6911290
3: -0.3286017, 0.5278014, -1.2268595, 0.8088975, -1.1374991, 1.7546608
4: -0.5462130, 0.5351171, -1.2211061, 1.1222211, -1.6684341, 1.7562232
5: -0.2489608, 1.1318836, -1.1177658, 1.1734645, -1.4224253, 2.2496495
6: -0.3253044, 0.5482060, -0.8390513, 1.0583521, -1.3836565, 1.3872573
7: -0.4359377, 0.5933949, -0.9696023, 1.2092265, -1.6451643, 1.5629972
8: -0.4451309, 0.7025202, -1.1658900, 1.1072890, -1.5524199, 1.8684101
9: -0.5160955, 0.6172245, -1.1007218, 1.1425996, -1.6586950, 1.7179463

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 218

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_A1_B2_A1_B2_B1

### Relational analysis result of NS_A1_B2_A1_A1_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5424521, upper bound: 1.4944975
time: 1.94 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2_A1_B2_B2

### Relational analysis result of NS_A1_B2_A1_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5437171, upper bound: 1.4925852
time: 2.03 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.5672411, 0.5714372, -1.2890378, 0.9274813, -1.4947224, 1.8604751
1: -0.4807033, 0.6761887, -0.9157523, 1.1495152, -1.6302185, 1.5919410
2: -0.4134003, 0.6989219, -1.0107118, 1.1897380, -1.6031383, 1.7096337
3: -0.3698834, 0.5453974, -1.2052171, 0.8008965, -1.1707799, 1.7506145
4: -0.5775470, 0.5667447, -1.2047541, 1.1080422, -1.6855892, 1.7714989
5: -0.2994074, 1.1559496, -1.0903399, 1.1533077, -1.4527152, 2.2462895
6: -0.3482073, 0.5723276, -0.8262197, 1.0460510, -1.3942583, 1.3985473
7: -0.4675447, 0.6279905, -0.9564478, 1.1938564, -1.6614010, 1.5844383
8: -0.4768460, 0.7280012, -1.1485672, 1.0967597, -1.5736058, 1.8765684
9: -0.5463970, 0.6463131, -1.0863751, 1.1296828, -1.6760798, 1.7326882

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5506091, upper bound: 1.4976625
time: 1.82 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5437171, upper bound: 1.4976625
time: 2.01 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.5672411, 0.5714372, -1.3084635, 0.9377898, -1.5050309, 1.8799007
1: -0.4807033, 0.6761887, -0.9267504, 1.1646020, -1.6453054, 1.6029391
2: -0.4134003, 0.6989219, -1.0264972, 1.2027751, -1.6161754, 1.7254192
3: -0.3698834, 0.5453974, -1.2268595, 0.8088975, -1.1787809, 1.7722569
4: -0.5775470, 0.5667447, -1.2211061, 1.1222211, -1.6997681, 1.7878509
5: -0.2994074, 1.1559496, -1.1177658, 1.1734645, -1.4728720, 2.2737155
6: -0.3482073, 0.5723276, -0.8390513, 1.0583521, -1.4065595, 1.4113789
7: -0.4675447, 0.6279905, -0.9696023, 1.2092265, -1.6767712, 1.5975928
8: -0.4768460, 0.7280012, -1.1658900, 1.1072890, -1.5841351, 1.8938912
9: -0.5463970, 0.6463131, -1.1007218, 1.1425996, -1.6889966, 1.7470349

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5506091, upper bound: 1.4978764
time: 1.98 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5437171, upper bound: 1.4978764
time: 1.83 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.8431016, 0.7113134, -0.9132454, 0.7449896, -1.5880911, 1.6245589
1: -0.6427077, 0.8172264, -0.6874459, 0.9164211, -1.5591288, 1.5046723
2: -0.6315987, 0.8888518, -0.6966894, 0.9341465, -1.5657452, 1.5855412
3: -0.6776149, 0.6384385, -0.7617079, 0.6685639, -1.3461788, 1.4001464
4: -0.8074291, 0.7789104, -0.8717791, 0.8288034, -1.6362326, 1.6506895
5: -0.5733571, 1.1524251, -0.6961297, 1.1671618, -1.7405189, 1.8485548
6: -0.5266141, 0.7401577, -0.5774449, 0.7917952, -1.3184092, 1.3176026
7: -0.6504493, 0.8415111, -0.7024334, 0.8986073, -1.5490565, 1.5439446
8: -0.7190910, 0.8633411, -0.7948046, 0.9024062, -1.6214972, 1.6581457
9: -0.7289797, 0.8291612, -0.7975066, 0.8774052, -1.6063849, 1.6266677

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_A1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5576512, upper bound: 1.4905858
time: 2.11 seconds

## Relational analysis of NS_A1_B2_A1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5576512, upper bound: 1.5006464
time: 1.82 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.8431016, 0.7113134, -1.2608913, 0.9143122, -1.7574139, 1.9722047
1: -0.6427077, 0.8172264, -0.8981512, 1.1261199, -1.7688276, 1.7153776
2: -0.6315987, 0.8888518, -0.9865626, 1.1704710, -1.8020697, 1.8754144
3: -0.6776149, 0.6384385, -1.1709890, 0.7913750, -1.4689900, 1.8094275
4: -0.8074291, 0.7789104, -1.1791251, 1.0867043, -1.8941333, 1.9580355
5: -0.5733571, 1.1524251, -1.0597894, 1.1690992, -1.7424563, 2.2122145
6: -0.5266141, 0.7401577, -0.8073027, 1.0264828, -1.5530969, 1.5474604
7: -0.6504493, 0.8415111, -0.9369900, 1.1717957, -1.8222450, 1.7785012
8: -0.7190910, 0.8633411, -1.1213423, 1.0822730, -1.8013639, 1.9846834
9: -0.7289797, 0.8291612, -1.0641882, 1.1104386, -1.8394183, 1.8933494

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_A1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5576512, upper bound: 1.4905858
time: 1.99 seconds

## Relational analysis of NS_A1_B2_A1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5576512, upper bound: 1.5006464
time: 2.08 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.0870236, 0.8328964, -0.9132454, 0.7449896, -1.8320132, 1.7461418
1: -0.7848459, 0.9608730, -0.6874459, 0.9164211, -1.7012670, 1.6483190
2: -0.8271096, 1.0536442, -0.6966894, 0.9341465, -1.7612562, 1.7503335
3: -0.9545004, 0.7223951, -0.7617079, 0.6685639, -1.6230643, 1.4841030
4: -1.0145283, 0.9655470, -0.8717791, 0.8288034, -1.8433317, 1.8373260
5: -0.8336362, 1.1588053, -0.6961297, 1.1671618, -2.0007980, 1.8549349
6: -0.6825227, 0.8902621, -0.5774449, 0.7917952, -1.4743179, 1.4677069
7: -0.8121901, 1.0278324, -0.7024334, 0.8986073, -1.7107973, 1.7302659
8: -0.9324895, 0.9826777, -0.7948046, 0.9024062, -1.8348957, 1.7774823
9: -0.8883433, 0.9903239, -0.7975066, 0.8774052, -1.7657485, 1.7878305

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_A1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5453651, upper bound: 1.4902917
time: 2.04 seconds

## Relational analysis of NS_A1_B2_A1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5453651, upper bound: 1.5006464
time: 1.98 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.0870236, 0.8328964, -1.2608913, 0.9143122, -2.0013359, 2.0937877
1: -0.7848459, 0.9608730, -0.8981512, 1.1261199, -1.9109657, 1.8590242
2: -0.8271096, 1.0536442, -0.9865626, 1.1704710, -1.9975805, 2.0402069
3: -0.9545004, 0.7223951, -1.1709890, 0.7913750, -1.7458755, 1.8933842
4: -1.0145283, 0.9655470, -1.1791251, 1.0867043, -2.1012325, 2.1446719
5: -0.8336362, 1.1588053, -1.0597894, 1.1690992, -2.0027354, 2.2185946
6: -0.6825227, 0.8902621, -0.8073027, 1.0264828, -1.7090056, 1.6975648
7: -0.8121901, 1.0278324, -0.9369900, 1.1717957, -1.9839859, 1.9648224
8: -0.9324895, 0.9826777, -1.1213423, 1.0822730, -2.0147624, 2.1040201
9: -0.8883433, 0.9903239, -1.0641882, 1.1104386, -1.9987819, 2.0545120

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_A1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5453651, upper bound: 1.4902917
time: 1.89 seconds

## Relational analysis of NS_A1_B2_A1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5453651, upper bound: 1.5006464
time: 1.79 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.8007439, 0.6895599, -1.1707919, 0.8706188, -1.6713626, 1.8603518
1: -0.6186302, 0.7942876, -0.8433994, 1.0787935, -1.6974237, 1.6376870
2: -0.5981474, 0.8603787, -0.9116770, 1.1092011, -1.7073485, 1.7720557
3: -0.6306096, 0.6231076, -1.0648744, 0.7600607, -1.3906703, 1.6879821
4: -0.7721874, 0.7468568, -1.0994543, 1.0200633, -1.7922506, 1.8463111
5: -0.5268149, 1.1322763, -0.9709433, 1.1697750, -1.6965899, 2.1032195
6: -0.4996769, 0.7146878, -0.7479693, 0.9655082, -1.4651852, 1.4626571
7: -0.6225621, 0.8090602, -0.8766156, 1.1010218, -1.7235839, 1.6856759
8: -0.6827840, 0.8423395, -1.0366592, 1.0359489, -1.7187328, 1.8789988
9: -0.7017353, 0.8014120, -0.9951407, 1.0502661, -1.7520015, 1.7965527

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5453651, upper bound: 1.4881816
time: 2.00 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5453651, upper bound: 1.4883344
time: 2.05 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.8007439, 0.6895599, -1.5236932, 1.0425284, -1.8432723, 2.2132530
1: -0.6186302, 0.7942876, -1.0572674, 1.2932104, -1.9118407, 1.8515551
2: -0.5981474, 0.8603787, -1.2059506, 1.3490940, -1.9472414, 2.0663295
3: -0.6306096, 0.6231076, -1.4803259, 0.8847889, -1.5153985, 2.1034336
4: -0.7721874, 0.7468568, -1.4114362, 1.2818787, -2.0540662, 2.1582930
5: -0.5268149, 1.1322763, -1.3431863, 1.1716940, -1.6985090, 2.4754624
6: -0.4996769, 0.7146878, -0.9813240, 1.2037208, -1.7033978, 1.6960118
7: -0.6225621, 0.8090602, -1.1147566, 1.3783436, -2.0009058, 1.9238167
8: -0.6827840, 0.8423395, -1.3681177, 1.2185670, -1.9013510, 2.2104573
9: -0.7017353, 0.8014120, -1.2658544, 1.2868410, -1.9885764, 2.0672665

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5453651, upper bound: 1.4881816
time: 2.28 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5453651, upper bound: 1.4883344
time: 1.98 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.8009368, 0.6899464, -1.2353903, 0.9028609, -1.7037976, 1.9253366
1: -0.6184196, 0.7859109, -0.8819233, 1.1373546, -1.7557743, 1.6678343
2: -0.5976799, 0.8604170, -0.9660013, 1.1530033, -1.7506833, 1.8264183
3: -0.6300391, 0.6232427, -1.1404033, 0.7845557, -1.4145949, 1.7636460
4: -0.7718078, 0.7465528, -1.1562454, 1.0683935, -1.8402014, 1.9027982
5: -0.5217994, 1.1445112, -1.0559487, 1.1813469, -1.7031463, 2.2004600
6: -0.4994591, 0.7144890, -0.7912719, 1.0084571, -1.5079162, 1.5057609
7: -0.6221327, 0.8092420, -0.9212539, 1.1518955, -1.7740282, 1.7304959
8: -0.6824265, 0.8423425, -1.0968750, 1.0702790, -1.7527056, 1.9392174
9: -0.7014484, 0.8011314, -1.0446663, 1.0940982, -1.7955467, 1.8457978

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_A1_A2_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5453651, upper bound: 1.4978764
time: 1.95 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_A2_A1_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5453651, upper bound: 1.4978764
time: 1.87 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -1.0446327, 0.8114167, -1.2353903, 0.9028609, -1.9474936, 2.0468071
1: -0.7604214, 0.9293060, -0.8819233, 1.1373546, -1.8977760, 1.8112293
2: -0.7930146, 1.0250535, -0.9660013, 1.1530033, -1.9460180, 1.9910548
3: -0.9066652, 0.7071325, -1.1404033, 0.7845557, -1.6912210, 1.8475357
4: -0.9787142, 0.9330187, -1.1562454, 1.0683935, -2.0471077, 2.0892639
5: -0.7819548, 1.1508244, -1.0559487, 1.1813469, -1.9633017, 2.2067733
6: -0.6552263, 0.8644500, -0.7912719, 1.0084571, -1.6636834, 1.6557219
7: -0.7837325, 0.9953882, -0.9212539, 1.1518955, -1.9356281, 1.9166421
8: -0.8956239, 0.9615750, -1.0968750, 1.0702790, -1.9659029, 2.0584500
9: -0.8606642, 0.9621502, -1.0446663, 1.0940982, -1.9547625, 2.0068164

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 29

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_A1_A2_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5453651, upper bound: 1.4978764
time: 2.15 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_A2_A2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5453651, upper bound: 1.4978764
time: 1.68 seconds

## BFS NS instance: NS_A2_B1_B1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.9531180, 0.7638087, -0.5263217, 0.5448294, -1.4979473, 1.2901304
1: -0.7122197, 0.9460655, -0.4573160, 0.6466167, -1.3588364, 1.4033816
2: -0.7305958, 0.9614030, -0.3826107, 0.6646317, -1.3952276, 1.3440137
3: -0.8098066, 0.6820929, -0.3286017, 0.5278014, -1.3376080, 1.0106945
4: -0.9078164, 0.8588266, -0.5462130, 0.5351171, -1.4429336, 1.4050395
5: -0.7386829, 1.1507246, -0.2489608, 1.1318836, -1.8705665, 1.3996854
6: -0.6040805, 0.8193476, -0.3253044, 0.5482060, -1.1522865, 1.1446519
7: -0.7297332, 0.9298794, -0.4359377, 0.5933949, -1.3231281, 1.3658171
8: -0.8331056, 0.9228835, -0.4451309, 0.7025202, -1.5356258, 1.3680143
9: -0.8287107, 0.9044701, -0.5160955, 0.6172245, -1.4459352, 1.4205656

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_B1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_B1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5257456, upper bound: 1.5681235
time: 1.65 seconds

## Relational analysis of NS_A2_B1_B1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_B1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5261914, upper bound: 1.5651700
time: 1.73 seconds

## BFS NS instance: NS_A2_B1_B1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.9561330, 0.7661559, -0.5263217, 0.5448294, -1.5009624, 1.2924776
1: -0.7132269, 0.9508001, -0.4573160, 0.6466167, -1.3598435, 1.4081161
2: -0.7326699, 0.9632682, -0.3826107, 0.6646317, -1.3973017, 1.3458788
3: -0.8120513, 0.6843585, -0.3286017, 0.5278014, -1.3398527, 1.0129602
4: -0.9096080, 0.8608177, -0.5462130, 0.5351171, -1.4447252, 1.4070307
5: -0.7484537, 1.1710870, -0.2489608, 1.1318836, -1.8803374, 1.4200478
6: -0.6060559, 0.8205242, -0.3253044, 0.5482060, -1.1542619, 1.1458285
7: -0.7318144, 0.9323517, -0.4359377, 0.5933949, -1.3252093, 1.3682895
8: -0.8349611, 0.9249382, -0.4451309, 0.7025202, -1.5374813, 1.3700690
9: -0.8304380, 0.9063733, -0.5160955, 0.6172245, -1.4476625, 1.4224687

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_B1_B1_A1_B1_A2_A1

### Relational analysis result of NS_A2_B1_B1_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5262310, upper bound: 1.5705517
time: 1.85 seconds

## Relational analysis of NS_A2_B1_B1_B1_A1_B1_A2_A2

### Relational analysis result of NS_A2_B1_B1_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5261914, upper bound: 1.5750502
time: 1.92 seconds

## BFS NS instance: NS_A2_B1_B1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.9531180, 0.7638087, -0.5672411, 0.5714372, -1.5245552, 1.3310499
1: -0.7122197, 0.9460655, -0.4807033, 0.6761887, -1.3884084, 1.4267688
2: -0.7305958, 0.9614030, -0.4134003, 0.6989219, -1.4295177, 1.3748033
3: -0.8098066, 0.6820929, -0.3698834, 0.5453974, -1.3552040, 1.0519763
4: -0.9078164, 0.8588266, -0.5775470, 0.5667447, -1.4745612, 1.4363735
5: -0.7386829, 1.1507246, -0.2994074, 1.1559496, -1.8946325, 1.4501321
6: -0.6040805, 0.8193476, -0.3482073, 0.5723276, -1.1764081, 1.1675549
7: -0.7297332, 0.9298794, -0.4675447, 0.6279905, -1.3577237, 1.3974241
8: -0.8331056, 0.9228835, -0.4768460, 0.7280012, -1.5611069, 1.3997295
9: -0.8287107, 0.9044701, -0.5463970, 0.6463131, -1.4750237, 1.4508672

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_B1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_B1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5257456, upper bound: 1.5681234
time: 2.01 seconds

## Relational analysis of NS_A2_B1_B1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_B1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5261914, upper bound: 1.5651700
time: 2.14 seconds

## BFS NS instance: NS_A2_B1_B1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.9561330, 0.7661559, -0.5672411, 0.5714372, -1.5275702, 1.3333970
1: -0.7132269, 0.9508001, -0.4807033, 0.6761887, -1.3894155, 1.4315033
2: -0.7326699, 0.9632682, -0.4134003, 0.6989219, -1.4315919, 1.3766685
3: -0.8120513, 0.6843585, -0.3698834, 0.5453974, -1.3574487, 1.0542419
4: -0.9096080, 0.8608177, -0.5775470, 0.5667447, -1.4763527, 1.4383646
5: -0.7484537, 1.1710870, -0.2994074, 1.1559496, -1.9044033, 1.4704945
6: -0.6060559, 0.8205242, -0.3482073, 0.5723276, -1.1783836, 1.1687315
7: -0.7318144, 0.9323517, -0.4675447, 0.6279905, -1.3598049, 1.3998964
8: -0.8349611, 0.9249382, -0.4768460, 0.7280012, -1.5629623, 1.4017842
9: -0.8304380, 0.9063733, -0.5463970, 0.6463131, -1.4767511, 1.4527702

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_B1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_B1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5257456, upper bound: 1.5778394
time: 2.23 seconds

## Relational analysis of NS_A2_B1_B1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_B1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5261914, upper bound: 1.5750502
time: 1.93 seconds

## BFS NS instance: NS_A2_B1_B1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1.2890378, 0.9274813, -0.5263217, 0.5448294, -1.8338672, 1.4538031
1: -0.9157523, 1.1495152, -0.4573160, 0.6466167, -1.5623690, 1.6068311
2: -1.0107118, 1.1897380, -0.3826107, 0.6646317, -1.6753435, 1.5723487
3: -1.2052171, 0.8008965, -0.3286017, 0.5278014, -1.7330185, 1.1294982
4: -1.2047541, 1.1080422, -0.5462130, 0.5351171, -1.7398713, 1.6542553
5: -1.0903399, 1.1533077, -0.2489608, 1.1318836, -2.2222235, 1.4022684
6: -0.8262197, 1.0460510, -0.3253044, 0.5482060, -1.3744256, 1.3713554
7: -0.9564478, 1.1938564, -0.4359377, 0.5933949, -1.5498427, 1.6297941
8: -1.1485672, 1.0967597, -0.4451309, 0.7025202, -1.8510873, 1.5418906
9: -1.0863751, 1.1296828, -0.5160955, 0.6172245, -1.7035997, 1.6457782

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 218

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_B1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_B1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4921119, upper bound: 1.5506546
time: 1.85 seconds

## Relational analysis of NS_A2_B1_B1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_B1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4925852, upper bound: 1.5437171
time: 1.98 seconds

## BFS NS instance: NS_A2_B1_B1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.3084635, 0.9377898, -0.5263217, 0.5448294, -1.8532928, 1.4641116
1: -0.9267504, 1.1646020, -0.4573160, 0.6466167, -1.5733671, 1.6219180
2: -1.0264972, 1.2027751, -0.3826107, 0.6646317, -1.6911290, 1.5853858
3: -1.2268595, 0.8088975, -0.3286017, 0.5278014, -1.7546608, 1.1374991
4: -1.2211061, 1.1222211, -0.5462130, 0.5351171, -1.7562232, 1.6684341
5: -1.1177658, 1.1734645, -0.2489608, 1.1318836, -2.2496495, 1.4224253
6: -0.8390513, 1.0583521, -0.3253044, 0.5482060, -1.3872573, 1.3836565
7: -0.9696023, 1.2092265, -0.4359377, 0.5933949, -1.5629972, 1.6451643
8: -1.1658900, 1.1072890, -0.4451309, 0.7025202, -1.8684101, 1.5524199
9: -1.1007218, 1.1425996, -0.5160955, 0.6172245, -1.7179463, 1.6586950

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 218

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_B1_B1_A2_B1_A2_A1

### Relational analysis result of NS_A2_B1_B1_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4944975, upper bound: 1.5546180
time: 1.97 seconds

## Relational analysis of NS_A2_B1_B1_B1_A2_B1_A2_A2

### Relational analysis result of NS_A2_B1_B1_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4925852, upper bound: 1.5587341
time: 2.03 seconds

## BFS NS instance: NS_A2_B1_B1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1.2890378, 0.9274813, -0.5672411, 0.5714372, -1.8604751, 1.4947224
1: -0.9157523, 1.1495152, -0.4807033, 0.6761887, -1.5919410, 1.6302185
2: -1.0107118, 1.1897380, -0.4134003, 0.6989219, -1.7096337, 1.6031383
3: -1.2052171, 0.8008965, -0.3698834, 0.5453974, -1.7506145, 1.1707799
4: -1.2047541, 1.1080422, -0.5775470, 0.5667447, -1.7714989, 1.6855892
5: -1.0903399, 1.1533077, -0.2994074, 1.1559496, -2.2462895, 1.4527152
6: -0.8262197, 1.0460510, -0.3482073, 0.5723276, -1.3985473, 1.3942583
7: -0.9564478, 1.1938564, -0.4675447, 0.6279905, -1.5844383, 1.6614010
8: -1.1485672, 1.0967597, -0.4768460, 0.7280012, -1.8765684, 1.5736058
9: -1.0863751, 1.1296828, -0.5463970, 0.6463131, -1.7326882, 1.6760798

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_B1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_B1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4921119, upper bound: 1.5506091
time: 1.76 seconds

## Relational analysis of NS_A2_B1_B1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_B1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4925852, upper bound: 1.5437171
time: 2.08 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.90 + 596.80 = 601.70 seconds
