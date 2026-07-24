## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 8)
Time budget: 600 seconds
Split limit: 100
Threshold: 21.8342855583


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-16.9740028, 13.6893234, -16.9740028, 13.6893234, -30.6633148, 30.6633167)
1: (-14.0589867, 11.9158411, -14.0589867, 11.9158411, -25.9748268, 25.9748249)
2: (-17.3404541, 10.5976658, -17.3404541, 10.5976658, -27.9381199, 27.9381199)
3: (-20.5597763, 9.2333050, -20.5597763, 9.2333050, -29.7930794, 29.7930737)
4: (-18.3966045, 14.3898029, -18.3966045, 14.3898029, -32.7863998, 32.7863998)
5: (-15.9375210, 12.1460552, -15.9375210, 12.1460552, -28.0835762, 28.0835762)
6: (-14.7187529, 15.9408302, -14.7187529, 15.9408302, -30.6595840, 30.6595840)
7: (-18.2727642, 10.8036785, -18.2727642, 10.8036785, -29.0764408, 29.0764370)
8: (-19.0032673, 13.1205730, -19.0032673, 13.1205730, -32.1238403, 32.1238403)
9: (-14.8070126, 15.1359177, -14.8070126, 15.1359177, -29.9429302, 29.9429302)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.02 + 5.95 = 7.97 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -21.8561417, upper bound: 21.8561417

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8557715, upper bound: 21.8557122
time: 4.28 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8556421, upper bound: 21.8556421
time: 4.65 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 9.13 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 9.13
Output dim: 7, lower bound: -21.8557715, upper bound: 21.8557122
NS_A2, status: Status.UNKNOWN, split count: 1, time: 9.13
Output dim: 7, lower bound: -21.8556421, upper bound: 21.8556421

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -14.9987030, 12.1140194, -16.5522709, 13.3475876, -28.3462906, 28.6662903
1: -12.3947115, 10.5611153, -13.6968679, 11.6205034, -24.0152149, 24.2579842
2: -15.2852068, 9.4089088, -16.8902435, 10.3181572, -25.6033630, 26.2991505
3: -18.1201172, 8.1803808, -20.0426445, 9.0001316, -27.1202488, 28.2230225
4: -16.2541275, 12.7381372, -17.9408360, 14.0304842, -30.2846107, 30.6789703
5: -14.0756416, 10.7550354, -15.5345955, 11.8379726, -25.9136124, 26.2896271
6: -13.0072546, 14.1124325, -14.3417768, 15.5508289, -28.5580826, 28.4542084
7: -16.1668015, 9.5228405, -17.8195591, 10.4904861, -26.6572876, 27.3423996
8: -16.7991772, 11.6073093, -18.5197334, 12.7888498, -29.5880260, 30.1270409
9: -13.0893850, 13.3706827, -14.4228840, 14.7465830, -27.8359680, 27.7935658

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8545721, upper bound: 21.8546406
time: 7.56 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8544184, upper bound: 21.8543674
time: 6.30 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -16.4103661, 13.2276020, -16.7749653, 13.5261469, -29.9365044, 30.0025616
1: -13.5722818, 11.5181236, -13.8869162, 11.7747812, -25.3470631, 25.4050388
2: -16.7317810, 10.2148952, -17.1248779, 10.4623203, -27.1941013, 27.3397732
3: -19.8714905, 8.9169044, -20.3163967, 9.1210241, -28.9925137, 29.2333012
4: -17.7898293, 13.9090538, -18.1821651, 14.2190895, -32.0089188, 32.0912170
5: -15.3968363, 11.7276077, -15.7461891, 11.9983196, -27.3951530, 27.4737968
6: -14.2128859, 15.4234409, -14.5372677, 15.7576218, -29.9705086, 29.9607086
7: -17.6730232, 10.3609743, -18.0603619, 10.6468077, -28.3198318, 28.4213333
8: -18.3477688, 12.6735630, -18.7720375, 12.9619970, -31.3097649, 31.4456005
9: -14.2837505, 14.6119957, -14.6217003, 14.9501686, -29.2339172, 29.2336960

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8544675, upper bound: 21.8545850
time: 4.99 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8543026, upper bound: 21.8543027
time: 3.23 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 10.31 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 10.31
Output dim: 7, lower bound: -21.8545721, upper bound: 21.8546406
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 10.31
Output dim: 7, lower bound: -21.8544184, upper bound: 21.8543674
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 10.31
Output dim: 7, lower bound: -21.8544675, upper bound: 21.8545850
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 10.31
Output dim: 7, lower bound: -21.8543026, upper bound: 21.8543027

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -14.8005104, 11.9552517, -15.9974174, 12.9026756, -27.7031860, 27.9526691
1: -12.2254848, 10.4238596, -13.2233086, 11.2359409, -23.4614182, 23.6471672
2: -15.0781250, 9.2896576, -16.3090324, 9.9795198, -25.0576420, 25.5986881
3: -17.8733387, 8.0741367, -19.3540993, 8.7014351, -26.5747719, 27.4282341
4: -16.0389404, 12.5716228, -17.3378277, 13.5651569, -29.6040955, 29.9094505
5: -13.8876076, 10.6153498, -15.0089149, 11.4448795, -25.3324871, 25.6242638
6: -12.8350925, 13.9273157, -13.8583164, 15.0327921, -27.8678818, 27.7856255
7: -15.9532852, 9.3951044, -17.2230034, 10.1237917, -26.0770741, 26.6181068
8: -16.5765190, 11.4549713, -17.8949280, 12.3606157, -28.9371300, 29.3498955
9: -12.9174976, 13.1932182, -13.9391527, 14.2456598, -27.1631584, 27.1323700

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8544184, upper bound: 21.8543674
time: 21.08 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8544184, upper bound: 21.8543674
time: 9.06 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -14.7602167, 11.9227657, -18.1242027, 14.6097412, -29.3699570, 30.0469685
1: -12.1902981, 10.3956623, -15.0128555, 12.7045145, -24.8948135, 25.4085159
2: -15.0347776, 9.2641191, -18.5321083, 11.3030624, -26.3378372, 27.7962265
3: -17.8231430, 8.0523052, -21.9959373, 9.8504124, -27.6735516, 30.0482407
4: -15.9952860, 12.5373631, -19.6434784, 15.3405495, -31.3358345, 32.1808395
5: -13.8485851, 10.5863647, -17.0217762, 12.9612331, -26.8098183, 27.6081409
6: -12.7994232, 13.8891859, -15.7044411, 16.9857464, -29.7851696, 29.5936279
7: -15.9097643, 9.3663206, -19.4632263, 11.5631351, -27.4728985, 28.8295479
8: -16.5298405, 11.4237700, -20.2818413, 14.0079098, -30.5377502, 31.7056084
9: -12.8816299, 13.1563263, -15.8083143, 16.1524582, -29.0340881, 28.9646416

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 221

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8537976, upper bound: 21.8538305
time: 7.22 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8544184, upper bound: 21.8543674
time: 6.61 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -16.2105484, 13.0674849, -16.2200699, 13.0807381, -29.2912865, 29.2875557
1: -13.4018650, 11.3796940, -13.4126005, 11.3900681, -24.7919331, 24.7922935
2: -16.5232391, 10.0940514, -16.5432873, 10.1232290, -26.6464691, 26.6373386
3: -19.6232243, 8.8096409, -19.6272469, 8.8220978, -28.4453220, 28.4368839
4: -17.5727806, 13.7414970, -17.5788765, 13.7536755, -31.3264561, 31.3203735
5: -15.2075138, 11.5861511, -15.2199430, 11.6050510, -26.8125648, 26.8060951
6: -14.0391264, 15.2370167, -14.0534611, 15.2391739, -29.2782993, 29.2904778
7: -17.4581871, 10.2309361, -17.4635468, 10.2798338, -27.7380199, 27.6944828
8: -18.1230831, 12.5197697, -18.1468773, 12.5334949, -30.6565781, 30.6666470
9: -14.1101198, 14.4326324, -14.1377335, 14.4488564, -28.5589752, 28.5703659

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8543026, upper bound: 21.8543026
time: 5.40 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8543026, upper bound: 21.8543027
time: 6.67 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -16.1687546, 13.0337620, -18.4858570, 14.8987484, -31.0675030, 31.5196190
1: -13.3655539, 11.3503742, -15.3193197, 12.9537354, -26.3192902, 26.6696930
2: -16.4785156, 10.0673761, -18.9096355, 11.5240650, -28.0025749, 28.9770126
3: -19.5712585, 8.7869253, -22.4428673, 10.0439997, -29.6152573, 31.2297916
4: -17.5274315, 13.7060204, -20.0361786, 15.6445484, -33.1719818, 33.7421951
5: -15.1670656, 11.5558920, -17.3643932, 13.2167492, -28.3838120, 28.9202805
6: -14.0020475, 15.1975422, -16.0183945, 17.3223572, -31.3244057, 31.2159367
7: -17.4129715, 10.2010870, -19.8523521, 11.7979670, -29.2109375, 30.0534401
8: -18.0746670, 12.4872999, -20.6866913, 14.2863903, -32.3610573, 33.1739922
9: -14.0728941, 14.3943329, -16.1255493, 16.4779587, -30.5508461, 30.5198822

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 221

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8536999, upper bound: 21.8537735
time: 3.80 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8543026, upper bound: 21.8543026
time: 7.35 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 13.40 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 13.40
Output dim: 7, lower bound: -21.8544184, upper bound: 21.8543674
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 13.40
Output dim: 7, lower bound: -21.8544184, upper bound: 21.8543674
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 13.40
Output dim: 7, lower bound: -21.8537976, upper bound: 21.8538305
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 13.40
Output dim: 7, lower bound: -21.8544184, upper bound: 21.8543674
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 13.40
Output dim: 7, lower bound: -21.8543026, upper bound: 21.8543026
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 13.40
Output dim: 7, lower bound: -21.8543026, upper bound: 21.8543027
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 13.40
Output dim: 7, lower bound: -21.8536999, upper bound: 21.8537735
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 13.40
Output dim: 7, lower bound: -21.8543026, upper bound: 21.8543026

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -14.4583874, 11.6812420, -15.9974174, 12.9026756, -27.3610611, 27.6786594
1: -11.9334192, 10.1870422, -13.2233086, 11.2359409, -23.1693592, 23.4103508
2: -14.7203655, 9.0828724, -16.3090324, 9.9795198, -24.6998863, 25.3919029
3: -17.4476471, 7.8904905, -19.3540993, 8.7014351, -26.1490822, 27.2445869
4: -15.6676292, 12.2841969, -17.3378277, 13.5651569, -29.2327843, 29.6220245
5: -13.5630569, 10.3740959, -15.0089149, 11.4448795, -25.0079365, 25.3830109
6: -12.5377331, 13.6078987, -13.8583164, 15.0327921, -27.5705261, 27.4662151
7: -15.5848446, 9.1730785, -17.2230034, 10.1237917, -25.7086372, 26.3960800
8: -16.1917763, 11.1918011, -17.8949280, 12.3606157, -28.5523834, 29.0867290
9: -12.6203938, 12.8866940, -13.9391527, 14.2456598, -26.8660545, 26.8258476

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 221

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8545721, upper bound: 21.8546406
time: 6.66 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8545721, upper bound: 21.8546406
time: 7.66 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -16.5880547, 13.3791656, -15.9974174, 12.9026756, -29.4907303, 29.3765793
1: -13.7159586, 11.6504087, -13.2233086, 11.2359409, -24.9518948, 24.8737183
2: -16.9372463, 10.3743496, -16.3090324, 9.9795198, -26.9167671, 26.6833782
3: -20.0923538, 9.0345821, -19.3540993, 8.7014351, -28.7937889, 28.3886795
4: -17.9700851, 14.0579910, -17.3378277, 13.5651569, -31.5352402, 31.3958187
5: -15.5741634, 11.8749561, -15.0089149, 11.4448795, -27.0190430, 26.8838711
6: -14.3741188, 15.5606613, -13.8583164, 15.0327921, -29.4069099, 29.4189777
7: -17.8239708, 10.5697470, -17.2230034, 10.1237917, -27.9477596, 27.7927513
8: -18.5682144, 12.8257856, -17.8949280, 12.3606157, -30.9288292, 30.7207108
9: -14.4674511, 14.7733011, -13.9391527, 14.2456598, -28.7131081, 28.7124538

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 221

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8545721, upper bound: 21.8546406
time: 4.17 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8545721, upper bound: 21.8546406
time: 5.33 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -15.3281040, 12.3818130, -17.5192642, 14.1281042, -29.4562073, 29.9010754
1: -12.6622295, 10.7867804, -14.5015402, 12.2899551, -24.9521828, 25.2883186
2: -15.6461048, 9.6723642, -17.9031467, 10.9383993, -26.5845032, 27.5755119
3: -18.4835243, 8.3734112, -21.2422390, 9.5279093, -28.0114269, 29.6156502
4: -16.5866585, 13.0124798, -18.9874802, 14.8365755, -31.4232254, 31.9999599
5: -14.3817863, 11.0159512, -16.4497375, 12.5371723, -26.9189587, 27.4656887
6: -13.2993565, 14.3824205, -15.1814461, 16.4245777, -29.7239342, 29.5638657
7: -16.4703388, 9.8772907, -18.8182583, 11.1773300, -27.6476688, 28.6955490
8: -17.1981525, 11.8826294, -19.6065540, 13.5461674, -30.7443199, 31.4891834
9: -13.4088917, 13.6783276, -15.2865286, 15.6112852, -29.0201721, 28.9648552

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8537976, upper bound: 21.8538305
time: 7.07 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8537976, upper bound: 21.8538305
time: 4.06 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -14.5093708, 11.7231798, -18.1223984, 14.6083012, -29.1176720, 29.8455772
1: -11.9778728, 10.2235003, -15.0113220, 12.7032709, -24.6811447, 25.2348213
2: -14.7735939, 9.1140079, -18.5302258, 11.3019714, -26.0755596, 27.6442318
3: -17.5107803, 7.9186502, -21.9936829, 9.8494473, -27.3602257, 29.9123325
4: -15.7236328, 12.3277264, -19.6415176, 15.3390446, -31.0626755, 31.9692440
5: -13.6112232, 10.4109945, -17.0200672, 12.9599628, -26.5711861, 27.4310589
6: -12.5826244, 13.6563320, -15.7028770, 16.9840698, -29.5666924, 29.3592052
7: -15.6418076, 9.2072325, -19.4612999, 11.5619736, -27.2037792, 28.6685295
8: -16.2495441, 11.2322674, -20.2798233, 14.0065231, -30.2560673, 31.5120850
9: -12.6654186, 12.9327240, -15.8067474, 16.1508369, -28.8162498, 28.7394714

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8544184, upper bound: 21.8543674
time: 4.65 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8544184, upper bound: 21.8543674
time: 11.13 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -15.8670597, 12.7921581, -16.2200699, 13.0807381, -28.9477978, 29.0122280
1: -13.1089764, 11.1418409, -13.4126005, 11.3900681, -24.4990444, 24.5544415
2: -16.1647415, 9.8859158, -16.5432873, 10.1232290, -26.2879715, 26.4292030
3: -19.1960373, 8.6254482, -19.6272469, 8.8220978, -28.0181351, 28.2526875
4: -17.1998653, 13.4530392, -17.5788765, 13.7536755, -30.9535408, 31.0319138
5: -14.8818531, 11.3431835, -15.2199430, 11.6050510, -26.4869041, 26.5631218
6: -13.7403584, 14.9166317, -14.0534611, 15.2391739, -28.9795284, 28.9700909
7: -17.0883980, 10.0074921, -17.4635468, 10.2798338, -27.3682308, 27.4710388
8: -17.7368755, 12.2552490, -18.1468773, 12.5334949, -30.2703629, 30.4021225
9: -13.8113461, 14.1249504, -14.1377335, 14.4488564, -28.2601929, 28.2626820

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8544675, upper bound: 21.8545850
time: 7.18 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8544675, upper bound: 21.8545850
time: 7.30 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -17.9571381, 14.4636078, -16.2200699, 13.0807381, -31.0378761, 30.6836777
1: -14.8614378, 12.5817528, -13.4126005, 11.3900681, -26.2515068, 25.9943542
2: -18.3357925, 11.1693945, -16.5432873, 10.1232290, -28.4590149, 27.7126808
3: -21.7897339, 9.7488718, -19.6272469, 8.8220978, -30.6118317, 29.3761120
4: -19.4615479, 15.1942978, -17.5788765, 13.7536755, -33.2152252, 32.7731667
5: -16.8538227, 12.8299885, -15.2199430, 11.6050510, -28.4588737, 28.0499306
6: -15.5470066, 16.8303566, -14.0534611, 15.2391739, -30.7861729, 30.8838177
7: -19.2855263, 11.3968906, -17.4635468, 10.2798338, -29.5653610, 28.8604317
8: -20.0755672, 13.8658113, -18.1468773, 12.5334949, -32.6090622, 32.0126877
9: -15.6369848, 15.9795284, -14.1377335, 14.4488564, -30.0858383, 30.1172619

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8544675, upper bound: 21.8545850
time: 4.71 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8544675, upper bound: 21.8545850
time: 5.39 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -16.7469444, 13.5046453, -17.8774700, 14.4143410, -31.1612759, 31.3821144
1: -13.8507977, 11.7497110, -14.8051205, 12.5369511, -26.3877487, 26.5548325
2: -17.0935211, 10.4999886, -18.2773666, 11.1579323, -28.2514496, 28.7773552
3: -20.2435265, 9.1083031, -21.6847343, 9.7198124, -29.9633389, 30.7930298
4: -18.1323414, 14.1910982, -19.3760109, 15.1377983, -33.2701416, 33.5671082
5: -15.7040091, 12.0001020, -16.7892761, 12.7906065, -28.4946156, 28.7893753
6: -14.5128946, 15.6970100, -15.4927311, 16.7577782, -31.2706718, 31.1897411
7: -17.9866734, 10.7363873, -19.2035980, 11.4108772, -29.3975506, 29.9399834
8: -18.7563362, 12.9578209, -20.0078030, 13.8221722, -32.5785065, 32.9656219
9: -14.6224728, 14.9245481, -15.6011343, 15.9338007, -30.5562687, 30.5256824

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8536999, upper bound: 21.8537735
time: 7.07 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8536999, upper bound: 21.8537735
time: 3.99 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -15.9024611, 12.8217602, -18.4840336, 14.8972940, -30.7997551, 31.3057919
1: -13.1403351, 11.1676426, -15.3177738, 12.9524860, -26.0928192, 26.4854164
2: -16.2019711, 9.9082584, -18.9077377, 11.5229645, -27.7249336, 28.8159943
3: -19.2397079, 8.6453571, -22.4406013, 10.0430269, -29.2827339, 31.0859547
4: -17.2389832, 13.4836082, -20.0341949, 15.6430273, -32.8820114, 33.5177994
5: -14.9153099, 11.3693657, -17.3626690, 13.2154684, -28.1307793, 28.7320328
6: -13.7719116, 14.9504814, -16.0168133, 17.3206654, -31.0925770, 30.9672947
7: -17.1285133, 10.0326967, -19.8504066, 11.7967968, -28.9253101, 29.8831024
8: -17.7775478, 12.2839127, -20.6846523, 14.2849960, -32.0625381, 32.9685669
9: -13.8432178, 14.1573114, -16.1239777, 16.4763203, -30.3195362, 30.2812843

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8543026, upper bound: 21.8543027
time: 8.93 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8543026, upper bound: 21.8543027
time: 3.89 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 14.83 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.83
Output dim: 7, lower bound: -21.8545721, upper bound: 21.8546406
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.83
Output dim: 7, lower bound: -21.8545721, upper bound: 21.8546406
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.83
Output dim: 7, lower bound: -21.8545721, upper bound: 21.8546406
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.83
Output dim: 7, lower bound: -21.8545721, upper bound: 21.8546406
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.83
Output dim: 7, lower bound: -21.8537976, upper bound: 21.8538305
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.83
Output dim: 7, lower bound: -21.8537976, upper bound: 21.8538305
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.83
Output dim: 7, lower bound: -21.8544184, upper bound: 21.8543674
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.83
Output dim: 7, lower bound: -21.8544184, upper bound: 21.8543674
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.83
Output dim: 7, lower bound: -21.8544675, upper bound: 21.8545850
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.83
Output dim: 7, lower bound: -21.8544675, upper bound: 21.8545850
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.83
Output dim: 7, lower bound: -21.8544675, upper bound: 21.8545850
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.83
Output dim: 7, lower bound: -21.8544675, upper bound: 21.8545850
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.83
Output dim: 7, lower bound: -21.8536999, upper bound: 21.8537735
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.83
Output dim: 7, lower bound: -21.8536999, upper bound: 21.8537735
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.83
Output dim: 7, lower bound: -21.8543026, upper bound: 21.8543027
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.83
Output dim: 7, lower bound: -21.8543026, upper bound: 21.8543027

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -14.4583874, 11.6812420, -14.4583874, 11.6812420, -26.1396294, 26.1396294
1: -11.9334192, 10.1870422, -11.9334192, 10.1870422, -22.1204605, 22.1204605
2: -14.7203655, 9.0828724, -14.7203655, 9.0828724, -23.8032360, 23.8032360
3: -17.4476471, 7.8904905, -17.4476471, 7.8904905, -25.3381367, 25.3381367
4: -15.6676292, 12.2841969, -15.6676292, 12.2841969, -27.9518242, 27.9518242
5: -13.5630569, 10.3740959, -13.5630569, 10.3740959, -23.9371529, 23.9371529
6: -12.5377331, 13.6078987, -12.5377331, 13.6078987, -26.1456318, 26.1456318
7: -15.5848446, 9.1730785, -15.5848446, 9.1730785, -24.7579231, 24.7579231
8: -16.1917763, 11.1918011, -16.1917763, 11.1918011, -27.3835754, 27.3835773
9: -12.6203938, 12.8866940, -12.6203938, 12.8866940, -25.5070858, 25.5070858

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8548845, upper bound: 21.8549164
time: 7.05 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8557715, upper bound: 21.8557122
time: 3.53 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -14.4583874, 11.6812420, -15.8670597, 12.7921581, -27.2505436, 27.5483017
1: -11.9334192, 10.1870422, -13.1089764, 11.1418409, -23.0752602, 23.2960129
2: -14.7203655, 9.0828724, -16.1647415, 9.8859158, -24.6062813, 25.2476139
3: -17.4476471, 7.8904905, -19.1960373, 8.6254482, -26.0730953, 27.0865288
4: -15.6676292, 12.2841969, -17.1998653, 13.4530392, -29.1206684, 29.4840622
5: -13.5630569, 10.3740959, -14.8818531, 11.3431835, -24.9062405, 25.2559471
6: -12.5377331, 13.6078987, -13.7403584, 14.9166317, -27.4543629, 27.3482571
7: -15.5848446, 9.1730785, -17.0883980, 10.0074921, -25.5923367, 26.2614765
8: -16.1917763, 11.1918011, -17.7368755, 12.2552490, -28.4470158, 28.9286766
9: -12.6203938, 12.8866940, -13.8113461, 14.1249504, -26.7453423, 26.6980381

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 221

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8548845, upper bound: 21.8549164
time: 6.15 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8557715, upper bound: 21.8557122
time: 5.12 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -16.5880547, 13.3791656, -14.4583874, 11.6812420, -28.2692947, 27.8375511
1: -13.7159586, 11.6504087, -11.9334192, 10.1870422, -23.9029999, 23.5838280
2: -16.9372463, 10.3743496, -14.7203655, 9.0828724, -26.0201168, 25.0947151
3: -20.0923538, 9.0345821, -17.4476471, 7.8904905, -27.9828434, 26.4822292
4: -17.9700851, 14.0579910, -15.6676292, 12.2841969, -30.2542820, 29.7256203
5: -15.5741634, 11.8749561, -13.5630569, 10.3740959, -25.9482574, 25.4380131
6: -14.3741188, 15.5606613, -12.5377331, 13.6078987, -27.9820175, 28.0983944
7: -17.8239708, 10.5697470, -15.5848446, 9.1730785, -26.9970493, 26.1545906
8: -18.5682144, 12.8257856, -16.1917763, 11.1918011, -29.7600155, 29.0175610
9: -14.4674511, 14.7733011, -12.6203938, 12.8866940, -27.3541451, 27.3936920

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8539348, upper bound: 21.8540596
time: 4.11 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8545721, upper bound: 21.8546406
time: 5.61 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -16.5880547, 13.3791656, -15.8670597, 12.7921581, -29.3802128, 29.2462196
1: -13.7159586, 11.6504087, -13.1089764, 11.1418409, -24.8577976, 24.7593842
2: -16.9372463, 10.3743496, -16.1647415, 9.8859158, -26.8231621, 26.5390892
3: -20.0923538, 9.0345821, -19.1960373, 8.6254482, -28.7178020, 28.2306194
4: -17.9700851, 14.0579910, -17.1998653, 13.4530392, -31.4231224, 31.2578545
5: -15.5741634, 11.8749561, -14.8818531, 11.3431835, -26.9173450, 26.7568092
6: -14.3741188, 15.5606613, -13.7403584, 14.9166317, -29.2907429, 29.3010197
7: -17.8239708, 10.5697470, -17.0883980, 10.0074921, -27.8314629, 27.6581459
8: -18.5682144, 12.8257856, -17.7368755, 12.2552490, -30.8234615, 30.5626564
9: -14.4674511, 14.7733011, -13.8113461, 14.1249504, -28.5924015, 28.5846481

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8539348, upper bound: 21.8540596
time: 9.74 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8545721, upper bound: 21.8546406
time: 8.48 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -15.3281040, 12.3818130, -16.0294361, 12.9395390, -28.2676430, 28.4112473
1: -12.6622295, 10.7867804, -13.2465878, 11.2717133, -23.9339428, 24.0333672
2: -15.6461048, 9.6723642, -16.3567352, 10.0546837, -25.7007885, 26.0290985
3: -18.4835243, 8.3734112, -19.3952065, 8.7380905, -27.2216148, 27.7686176
4: -16.5866585, 13.0124798, -17.3689480, 13.5940638, -30.1807175, 30.3814278
5: -14.3817863, 11.0159512, -15.0463457, 11.4935265, -25.8753128, 26.0622978
6: -13.2993565, 14.3824205, -13.8976841, 15.0425329, -28.3418884, 28.2801056
7: -16.4703388, 9.8772907, -17.2303982, 10.2344761, -26.7048130, 27.1076889
8: -17.1981525, 11.8826294, -17.9502411, 12.4073830, -29.6055355, 29.8328705
9: -13.4088917, 13.6783276, -13.9986773, 14.2817574, -27.6906433, 27.6770058

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 221

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8530703, upper bound: 21.8530398
time: 6.77 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8529729, upper bound: 21.8529993
time: 6.13 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -15.3281040, 12.3818130, -17.3721962, 14.0017090, -29.3298130, 29.7540016
1: -12.6622295, 10.7867804, -14.3706245, 12.1827641, -24.8449936, 25.1574059
2: -15.6461048, 9.6723642, -17.7361298, 10.8284025, -26.4745026, 27.4084930
3: -18.4835243, 8.3734112, -21.0627556, 9.4405079, -27.9240322, 29.4361668
4: -16.5866585, 13.0124798, -18.8298073, 14.7088470, -31.2955055, 31.8422871
5: -14.3817863, 11.0159512, -16.3047333, 12.4217291, -26.8035126, 27.3206844
6: -13.2993565, 14.3824205, -15.0460949, 16.2910137, -29.5903702, 29.4285164
7: -16.4703388, 9.8772907, -18.6642914, 11.0400629, -27.5103989, 28.5415821
8: -17.1981525, 11.8826294, -19.4271240, 13.4244747, -30.6226273, 31.3097534
9: -13.4088917, 13.6783276, -15.1389923, 15.4678917, -28.8767776, 28.8173199

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 221

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8530703, upper bound: 21.8530398
time: 8.90 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8529729, upper bound: 21.8529993
time: 8.96 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -14.5093708, 11.7231798, -16.6255836, 13.4144230, -27.9237938, 28.3487587
1: -11.9778728, 10.2235003, -13.7509098, 11.6806345, -23.6585045, 23.9744072
2: -14.7735939, 9.1140079, -16.9772129, 10.4152021, -25.1887932, 26.0912189
3: -17.5107803, 7.9186502, -20.1385689, 9.0563469, -26.5671196, 28.0572128
4: -15.7236328, 12.3277264, -18.0153637, 14.0910454, -29.8146782, 30.3430901
5: -13.6112232, 10.4109945, -15.6106358, 11.9117470, -25.5229702, 26.0216274
6: -12.5826244, 13.6563320, -14.4136248, 15.5956898, -28.1783104, 28.0699577
7: -15.6418076, 9.2072325, -17.8660126, 10.6163855, -26.2581863, 27.0732460
8: -16.2495441, 11.2322674, -18.6165581, 12.8628101, -29.1123524, 29.8488216
9: -12.6654186, 12.9327240, -14.5132771, 14.8157949, -27.4812126, 27.4460011

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8544184, upper bound: 21.8543674
time: 25.29 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8544184, upper bound: 21.8543674
time: 6.16 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -14.5093708, 11.7231798, -17.9750900, 14.4818411, -28.9912109, 29.6982689
1: -11.9778728, 10.2235003, -14.8802662, 12.5960054, -24.5738735, 25.1037674
2: -14.7735939, 9.1140079, -18.3629932, 11.1916723, -25.9652634, 27.4770012
3: -17.5107803, 7.9186502, -21.8140316, 9.7618885, -27.2726631, 29.7326794
4: -15.7236328, 12.3277264, -19.4835835, 15.2110901, -30.9347191, 31.8113098
5: -13.6112232, 10.4109945, -16.8748932, 12.8442669, -26.4554882, 27.2858829
6: -12.5826244, 13.6563320, -15.5672865, 16.8503761, -29.4330006, 29.2236176
7: -15.6418076, 9.2072325, -19.3070641, 11.4244089, -27.0662155, 28.5142937
8: -16.2495441, 11.2322674, -20.1000576, 13.8846064, -30.1341515, 31.3323212
9: -12.6654186, 12.9327240, -15.6589203, 16.0070801, -28.6724930, 28.5916424

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8544184, upper bound: 21.8543674
time: 3.84 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8544184, upper bound: 21.8543674
time: 6.68 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -15.8670597, 12.7921581, -14.4583874, 11.6812420, -27.5483017, 27.2505436
1: -13.1089764, 11.1418409, -11.9334192, 10.1870422, -23.2960129, 23.0752602
2: -16.1647415, 9.8859158, -14.7203655, 9.0828724, -25.2476139, 24.6062813
3: -19.1960373, 8.6254482, -17.4476471, 7.8904905, -27.0865288, 26.0730915
4: -17.1998653, 13.4530392, -15.6676292, 12.2841969, -29.4840622, 29.1206684
5: -14.8818531, 11.3431835, -13.5630569, 10.3740959, -25.2559452, 24.9062405
6: -13.7403584, 14.9166317, -12.5377331, 13.6078987, -27.3482571, 27.4543648
7: -17.0883980, 10.0074921, -15.5848446, 9.1730785, -26.2614765, 25.5923367
8: -17.7368755, 12.2552490, -16.1917763, 11.1918011, -28.9286728, 28.4470196
9: -13.8113461, 14.1249504, -12.6203938, 12.8866940, -26.6980362, 26.7453442

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8547647, upper bound: 21.8548679
time: 6.73 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8556421, upper bound: 21.8556421
time: 6.78 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -15.8670597, 12.7921581, -15.8670597, 12.7921581, -28.6592178, 28.6592178
1: -13.1089764, 11.1418409, -13.1089764, 11.1418409, -24.2508144, 24.2508125
2: -16.1647415, 9.8859158, -16.1647415, 9.8859158, -26.0506573, 26.0506573
3: -19.1960373, 8.6254482, -19.1960373, 8.6254482, -27.8214855, 27.8214855
4: -17.1998653, 13.4530392, -17.1998653, 13.4530392, -30.6529045, 30.6529045
5: -14.8818531, 11.3431835, -14.8818531, 11.3431835, -26.2250328, 26.2250347
6: -13.7403584, 14.9166317, -13.7403584, 14.9166317, -28.6569881, 28.6569881
7: -17.0883980, 10.0074921, -17.0883980, 10.0074921, -27.0958900, 27.0958900
8: -17.7368755, 12.2552490, -17.7368755, 12.2552490, -29.9921131, 29.9921150
9: -13.8113461, 14.1249504, -13.8113461, 14.1249504, -27.9362907, 27.9362888

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8547647, upper bound: 21.8548679
time: 3.55 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8556421, upper bound: 21.8556421
time: 9.00 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -17.9571381, 14.4636078, -14.4583874, 11.6812420, -29.6383801, 28.9219933
1: -14.8614378, 12.5817528, -11.9334192, 10.1870422, -25.0484810, 24.5151711
2: -18.3357925, 11.1693945, -14.7203655, 9.0828724, -27.4186649, 25.8897591
3: -21.7897339, 9.7488718, -17.4476471, 7.8904905, -29.6802254, 27.1965179
4: -19.4615479, 15.1942978, -15.6676292, 12.2841969, -31.7457428, 30.8619270
5: -16.8538227, 12.8299885, -13.5630569, 10.3740959, -27.2279186, 26.3930454
6: -15.5470066, 16.8303566, -12.5377331, 13.6078987, -29.1549015, 29.3680897
7: -19.2855263, 11.3968906, -15.5848446, 9.1730785, -28.4586048, 26.9817352
8: -20.0755672, 13.8658113, -16.1917763, 11.1918011, -31.2673683, 30.0575848
9: -15.6369848, 15.9795284, -12.6203938, 12.8866940, -28.5236778, 28.5999165

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8538428, upper bound: 21.8540172
time: 9.59 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8544675, upper bound: 21.8545850
time: 4.36 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -17.9571381, 14.4636078, -15.8670597, 12.7921581, -30.7492962, 30.3306656
1: -14.8614378, 12.5817528, -13.1089764, 11.1418409, -26.0032787, 25.6907234
2: -18.3357925, 11.1693945, -16.1647415, 9.8859158, -28.2217045, 27.3341370
3: -21.7897339, 9.7488718, -19.1960373, 8.6254482, -30.4151821, 28.9449062
4: -19.4615479, 15.1942978, -17.1998653, 13.4530392, -32.9145851, 32.3941612
5: -16.8538227, 12.8299885, -14.8818531, 11.3431835, -28.1970024, 27.7118416
6: -15.5470066, 16.8303566, -13.7403584, 14.9166317, -30.4636250, 30.5707111
7: -19.2855263, 11.3968906, -17.0883980, 10.0074921, -29.2930183, 28.4852867
8: -20.0755672, 13.8658113, -17.7368755, 12.2552490, -32.3308182, 31.6026840
9: -15.6369848, 15.9795284, -13.8113461, 14.1249504, -29.7619324, 29.7908688

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8538428, upper bound: 21.8540172
time: 4.42 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8544675, upper bound: 21.8545850
time: 6.88 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -16.7469444, 13.5046453, -16.0910282, 12.9884386, -29.7353745, 29.5956726
1: -13.8507977, 11.7497110, -13.2980824, 11.3139219, -25.1647186, 25.0477943
2: -17.0935211, 10.4999886, -16.4203854, 10.0888433, -27.1823654, 26.9203739
3: -20.2435265, 9.1083031, -19.4721279, 8.7702751, -29.0138016, 28.5804291
4: -18.1323414, 14.1910982, -17.4365616, 13.6454153, -31.7777538, 31.6276588
5: -15.7040091, 12.0001020, -15.1042557, 11.5358849, -27.2398949, 27.1043587
6: -14.5128946, 15.6970100, -13.9500942, 15.1002808, -29.6131744, 29.6471043
7: -17.9866734, 10.7363873, -17.2964859, 10.2695837, -28.2562561, 28.0328732
8: -18.7563362, 12.9578209, -18.0181599, 12.4539766, -31.2103119, 30.9759808
9: -14.6224728, 14.9245481, -14.0514765, 14.3360577, -28.9585304, 28.9760246

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 122

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8529685, upper bound: 21.8529829
time: 4.53 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8528589, upper bound: 21.8529333
time: 3.85 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -16.7469444, 13.5046453, -17.4869709, 14.0928297, -30.8397636, 30.9916077
1: -13.8507977, 11.7497110, -14.4667377, 12.2614222, -26.1122169, 26.2164497
2: -17.0935211, 10.4999886, -17.8547726, 10.8925934, -27.9861126, 28.3547611
3: -20.2435265, 9.1083031, -21.2056446, 9.5006571, -29.7441826, 30.3139458
4: -18.1323414, 14.1910982, -18.9552860, 14.8045444, -32.9368820, 33.1463852
5: -15.7040091, 12.0001020, -16.4128151, 12.5007029, -28.2047062, 28.4129181
6: -14.5128946, 15.6970100, -15.1439161, 16.3984165, -30.9113121, 30.8409271
7: -17.9866734, 10.7363873, -18.7872467, 11.1059618, -29.0926361, 29.5236340
8: -18.7563362, 12.9578209, -19.5536251, 13.5114202, -32.2677574, 32.5114441
9: -14.6224728, 14.9245481, -15.2373791, 15.5690498, -30.1915226, 30.1619263

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8529685, upper bound: 21.8529829
time: 7.18 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8528589, upper bound: 21.8529333
time: 7.21 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -15.9024611, 12.8217602, -16.6881256, 13.4640808, -29.3665428, 29.5098839
1: -13.1403351, 11.1676426, -13.8031387, 11.7234869, -24.8638191, 24.9707813
2: -16.2019711, 9.9082584, -17.0418472, 10.4498959, -26.6518669, 26.9501038
3: -19.2397079, 8.6453571, -20.2166805, 9.0890045, -28.3287125, 28.8620358
4: -17.2389832, 13.4836082, -18.0840740, 14.1431961, -31.3821793, 31.5676823
5: -14.9153099, 11.3693657, -15.6694393, 11.9547920, -26.8701019, 27.0388050
6: -13.7719116, 14.9504814, -14.4668350, 15.6543140, -29.4262257, 29.4173164
7: -17.1285133, 10.0326967, -17.9331398, 10.6520596, -27.7805729, 27.9658356
8: -17.7775478, 12.2839127, -18.6855011, 12.9101315, -30.6876793, 30.9694138
9: -13.8432178, 14.1573114, -14.5669041, 14.8709459, -28.7141647, 28.7242165

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8543026, upper bound: 21.8543027
time: 26.59 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8543026, upper bound: 21.8543027
time: 5.87 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -15.9024611, 12.8217602, -18.0922356, 14.5748444, -30.4773064, 30.9139881
1: -13.1403351, 11.1676426, -14.9782896, 12.6761503, -25.8164825, 26.1459312
2: -16.2019711, 9.9082584, -18.4839554, 11.2569466, -27.4589176, 28.3922119
3: -19.2397079, 8.6453571, -21.9598923, 9.8231754, -29.0628834, 30.6052475
4: -17.2389832, 13.4836082, -19.6117744, 15.3087044, -32.5476875, 33.0953827
5: -14.9153099, 11.3693657, -16.9851227, 12.9247932, -27.8401031, 28.3544846
6: -13.7719116, 14.9504814, -15.6669693, 16.9600182, -30.7319298, 30.6174469
7: -17.1285133, 10.0326967, -19.4325066, 11.4914894, -28.6200008, 29.4652023
8: -17.7775478, 12.2839127, -20.2290840, 13.9732714, -31.7508202, 32.5129967
9: -13.8432178, 14.1573114, -15.7592821, 16.1103325, -29.9535465, 29.9165936

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 221

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8543026, upper bound: 21.8543026
time: 8.05 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8543026, upper bound: 21.8543026
time: 6.93 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 17.06 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.06
Output dim: 7, lower bound: -21.8548845, upper bound: 21.8549164
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.06
Output dim: 7, lower bound: -21.8557715, upper bound: 21.8557122
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.06
Output dim: 7, lower bound: -21.8548845, upper bound: 21.8549164
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.06
Output dim: 7, lower bound: -21.8557715, upper bound: 21.8557122
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.06
Output dim: 7, lower bound: -21.8539348, upper bound: 21.8540596
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.06
Output dim: 7, lower bound: -21.8545721, upper bound: 21.8546406
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.06
Output dim: 7, lower bound: -21.8539348, upper bound: 21.8540596
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.06
Output dim: 7, lower bound: -21.8545721, upper bound: 21.8546406
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.06
Output dim: 7, lower bound: -21.8530703, upper bound: 21.8530398
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.06
Output dim: 7, lower bound: -21.8529729, upper bound: 21.8529993
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.06
Output dim: 7, lower bound: -21.8530703, upper bound: 21.8530398
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.06
Output dim: 7, lower bound: -21.8529729, upper bound: 21.8529993
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.06
Output dim: 7, lower bound: -21.8544184, upper bound: 21.8543674
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.06
Output dim: 7, lower bound: -21.8544184, upper bound: 21.8543674
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.06
Output dim: 7, lower bound: -21.8544184, upper bound: 21.8543674
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.06
Output dim: 7, lower bound: -21.8544184, upper bound: 21.8543674
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.06
Output dim: 7, lower bound: -21.8547647, upper bound: 21.8548679
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.06
Output dim: 7, lower bound: -21.8556421, upper bound: 21.8556421
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.06
Output dim: 7, lower bound: -21.8547647, upper bound: 21.8548679
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.06
Output dim: 7, lower bound: -21.8556421, upper bound: 21.8556421
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.06
Output dim: 7, lower bound: -21.8538428, upper bound: 21.8540172
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.06
Output dim: 7, lower bound: -21.8544675, upper bound: 21.8545850
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.06
Output dim: 7, lower bound: -21.8538428, upper bound: 21.8540172
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.06
Output dim: 7, lower bound: -21.8544675, upper bound: 21.8545850
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.06
Output dim: 7, lower bound: -21.8529685, upper bound: 21.8529829
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.06
Output dim: 7, lower bound: -21.8528589, upper bound: 21.8529333
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.06
Output dim: 7, lower bound: -21.8529685, upper bound: 21.8529829
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.06
Output dim: 7, lower bound: -21.8528589, upper bound: 21.8529333
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.06
Output dim: 7, lower bound: -21.8543026, upper bound: 21.8543027
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.06
Output dim: 7, lower bound: -21.8543026, upper bound: 21.8543027
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.06
Output dim: 7, lower bound: -21.8543026, upper bound: 21.8543026
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.06
Output dim: 7, lower bound: -21.8543026, upper bound: 21.8543026

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -15.0298700, 12.1432619, -13.8740988, 11.2165451, -26.2464142, 26.0173607
1: -12.4085588, 10.5807991, -11.4391756, 9.7860317, -22.1945896, 22.0199738
2: -15.3358288, 9.4939489, -14.1122618, 8.7336693, -24.0694981, 23.6062107
3: -18.1123867, 8.2137051, -16.7188606, 7.5793171, -25.6917038, 24.9325657
4: -16.2629967, 12.7625074, -15.0340500, 11.7964420, -28.0594387, 27.7965584
5: -14.1000175, 10.8062525, -13.0103998, 9.9657898, -24.0658073, 23.8166504
6: -13.0408821, 14.1044750, -12.0339708, 13.0649929, -26.1058712, 26.1384430
7: -16.1493206, 9.6871614, -14.9604807, 8.8046093, -24.9539280, 24.6476402
8: -16.8645000, 11.6535282, -15.5395660, 10.7469482, -27.6114464, 27.1930904
9: -13.1507282, 13.4121819, -12.1172028, 12.3658543, -25.5165825, 25.5293846

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 52

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8552544, upper bound: 21.8553675
time: 4.32 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8552543, upper bound: 21.8553675
time: 5.21 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -14.2132254, 11.4862251, -14.4566975, 11.6799002, -25.8931255, 25.9429226
1: -11.7257824, 10.0187826, -11.9319916, 10.1858854, -21.9116669, 21.9507732
2: -14.4650726, 8.9358950, -14.7186069, 9.0818615, -23.5469322, 23.6545029
3: -17.1425095, 7.7598362, -17.4455452, 7.8895898, -25.0320988, 25.2053814
4: -15.4021492, 12.0793266, -15.6658010, 12.2827864, -27.6849365, 27.7451286
5: -13.3310509, 10.2026834, -13.5614643, 10.3729143, -23.7039623, 23.7641430
6: -12.3257885, 13.3803368, -12.5362740, 13.6063318, -25.9321194, 25.9166107
7: -15.3229342, 9.0172939, -15.5830460, 9.1720057, -24.4949379, 24.6003399
8: -15.9177513, 11.0046101, -16.1898918, 11.1905155, -27.1082668, 27.1944981
9: -12.4090004, 12.6680832, -12.6189384, 12.8851891, -25.2941895, 25.2870216

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 221

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8553675, upper bound: 21.8552544
time: 5.98 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8553675, upper bound: 21.8561417
time: 7.36 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -15.0298700, 12.1432619, -15.2652340, 12.3139553, -27.3438244, 27.4084969
1: -12.4085588, 10.5807991, -12.6003380, 10.7297153, -23.1382751, 23.1811371
2: -15.3358288, 9.4939489, -15.5400772, 9.5272770, -24.8631058, 25.0340233
3: -18.1123867, 8.2137051, -18.4461288, 8.3056431, -26.4180298, 26.6598339
4: -16.2629967, 12.7625074, -16.5479088, 12.9511890, -29.2141857, 29.3104172
5: -14.1000175, 10.8062525, -14.3135910, 10.9231606, -25.0231781, 25.1198425
6: -13.0408821, 14.1044750, -13.2216969, 14.3583021, -27.3991795, 27.3261681
7: -16.1493206, 9.6871614, -16.4460564, 9.6294060, -25.7787266, 26.1332169
8: -16.8645000, 11.6535282, -17.0669060, 11.7974920, -28.6619911, 28.7204342
9: -13.1507282, 13.4121819, -13.2937679, 13.5896378, -26.7403660, 26.7059498

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8548840, upper bound: 21.8549131
time: 3.58 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8548845, upper bound: 21.8549164
time: 3.66 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -14.2132254, 11.4862251, -15.8652248, 12.7906990, -27.0039253, 27.3514500
1: -11.7257824, 10.0187826, -13.1074247, 11.1405849, -22.8663673, 23.1262074
2: -14.4650726, 8.9358950, -16.1628342, 9.8848228, -24.3498955, 25.0987282
3: -17.1425095, 7.7598362, -19.1937561, 8.6244745, -25.7669792, 26.9535923
4: -15.4021492, 12.0793266, -17.1978798, 13.4515076, -28.8536568, 29.2772045
5: -13.3310509, 10.2026834, -14.8801193, 11.3419018, -24.6729527, 25.0828018
6: -12.3257885, 13.3803368, -13.7387753, 14.9149294, -27.2407188, 27.1191120
7: -15.3229342, 9.0172939, -17.0864391, 10.0063353, -25.3292675, 26.1037331
8: -15.9177513, 11.0046101, -17.7348328, 12.2538490, -28.1715965, 28.7394409
9: -12.4090004, 12.6680832, -13.8097649, 14.1233158, -26.5323162, 26.4778481

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 221

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8550351, upper bound: 21.8548605
time: 4.93 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8550351, upper bound: 21.8557122
time: 8.10 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -17.2313461, 13.8969641, -13.8740988, 11.2165451, -28.4478836, 27.7710629
1: -14.2517891, 12.0926571, -11.4391756, 9.7860317, -24.0378208, 23.5318336
2: -17.6218719, 10.8228149, -14.1122618, 8.7336693, -26.3555412, 24.9350777
3: -20.8484917, 9.3963737, -16.7188606, 7.5793171, -28.4278088, 26.1152344
4: -18.6426220, 14.5931902, -15.0340500, 11.7964420, -30.4390583, 29.6272392
5: -16.1792336, 12.3562775, -13.0103998, 9.9657898, -26.1450214, 25.3666744
6: -14.9375391, 16.1213779, -12.0339708, 13.0649929, -28.0025330, 28.1553459
7: -18.4590435, 11.1237030, -14.9604807, 8.8046093, -27.2636528, 26.0841789
8: -19.3196259, 13.3402243, -15.5395660, 10.7469482, -30.0665665, 28.8797913
9: -15.0595303, 15.3604345, -12.1172028, 12.3658543, -27.4253845, 27.4776344

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 52

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8542519, upper bound: 21.8544624
time: 6.00 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8542519, upper bound: 21.8544624
time: 4.67 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -16.3356724, 13.1783667, -14.4566975, 11.6799002, -28.0155716, 27.6350632
1: -13.5023241, 11.4774141, -11.9319916, 10.1858854, -23.6882057, 23.4094009
2: -16.6747627, 10.2234859, -14.7186069, 9.0818615, -25.7566242, 24.9420929
3: -19.7779045, 8.9002676, -17.4455452, 7.8895898, -27.6674938, 26.3458138
4: -17.6968975, 13.8472404, -15.6658010, 12.2827864, -29.9796829, 29.5130424
5: -15.3354588, 11.6984615, -13.5614643, 10.3729143, -25.7083683, 25.2599258
6: -14.1560831, 15.3264914, -12.5362740, 13.6063318, -27.7624111, 27.8627644
7: -17.5545006, 10.4100132, -15.5830460, 9.1720057, -26.7265015, 25.9930592
8: -18.2863731, 12.6331730, -16.1898918, 11.1905155, -29.4768829, 28.8230572
9: -14.2500648, 14.5484877, -12.6189384, 12.8851891, -27.1352539, 27.1674252

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8542846, upper bound: 21.8543286
time: 12.67 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8542846, upper bound: 21.8550195
time: 5.77 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -17.2313461, 13.8969641, -15.2652340, 12.3139553, -29.5453014, 29.1621971
1: -14.2517891, 12.0926571, -12.6003380, 10.7297153, -24.9815044, 24.6929951
2: -17.6218719, 10.8228149, -15.5400772, 9.5272770, -27.1491470, 26.3628922
3: -20.8484917, 9.3963737, -18.4461288, 8.3056431, -29.1541348, 27.8425026
4: -18.6426220, 14.5931902, -16.5479088, 12.9511890, -31.5938110, 31.1410980
5: -16.1792336, 12.3562775, -14.3135910, 10.9231606, -27.1023941, 26.6698685
6: -14.9375391, 16.1213779, -13.2216969, 14.3583021, -29.2958393, 29.3430748
7: -18.4590435, 11.1237030, -16.4460564, 9.6294060, -28.0884495, 27.5697594
8: -19.3196259, 13.3402243, -17.0669060, 11.7974920, -31.1171112, 30.4071312
9: -15.0595303, 15.3604345, -13.2937679, 13.5896378, -28.6491680, 28.6541977

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 221

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8539344, upper bound: 21.8540596
time: 6.69 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8539348, upper bound: 21.8540596
time: 5.28 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 7.97 + 602.32 = 610.30 seconds
