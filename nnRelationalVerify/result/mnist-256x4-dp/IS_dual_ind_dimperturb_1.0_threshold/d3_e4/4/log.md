## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 10.310653145


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540)
1: (-4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358)
2: (-5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447)
3: (-6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520)
4: (-6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002)
5: (-5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936)
6: (-4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792)
7: (-5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823)
8: (-7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416)
9: (-4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.56 + 5.43 = 6.98 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -10.8533189, upper bound: 10.8533189

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 92

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8517581, upper bound: 10.8514231
time: 3.77 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8518155, upper bound: 10.8518158
time: 7.78 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 11.72 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 11.72
Output dim: 8, lower bound: -10.8517581, upper bound: 10.8514231
IS_A2, status: Status.UNKNOWN, split count: 1, time: 11.72
Output dim: 8, lower bound: -10.8518155, upper bound: 10.8518158

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -2.9945498, 2.4854207, -4.2330718, 3.5171497, -6.5116997, 6.7184916
1: -2.4506655, 2.4260209, -3.6493349, 3.3437438, -5.7944093, 6.0753555
2: -3.2870402, 2.6394088, -4.7990913, 3.5056651, -6.7927041, 7.4385004
3: -3.5422220, 2.3601861, -5.2073903, 3.1388066, -6.6810284, 7.5675764
4: -3.3720222, 2.6054800, -4.9245014, 3.6462295, -7.0182514, 7.5299807
5: -2.8525395, 2.5022659, -4.1511145, 3.4900503, -6.3425889, 6.6533804
6: -2.8254833, 2.7629449, -4.0150690, 3.9169633, -6.7424469, 6.7780137
7: -2.9726825, 2.8755846, -4.2652841, 4.1514683, -7.1241508, 7.1408687
8: -4.6693630, 3.0776982, -6.5759234, 3.5634408, -8.2328033, 9.6536217
9: -2.5680308, 2.8494966, -3.7233696, 4.0015116, -6.5695424, 6.5728664

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=32, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=74, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=15, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=210, inp2_unstable=223, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8503477, upper bound: 10.8499491
time: 2.59 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8502464, upper bound: 10.8499610
time: 3.04 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -3.6680012, 3.0478530, -4.2152534, 3.5034325, -7.1714334, 7.2631063
1: -3.1057496, 2.9268103, -3.6337061, 3.3329256, -6.4386749, 6.5605164
2: -4.1053696, 3.1162860, -4.7789278, 3.4932246, -7.5985942, 7.8952122
3: -4.4386749, 2.7713633, -5.1904926, 3.1330976, -7.5717726, 7.9618549
4: -4.2074580, 3.1692934, -4.9043150, 3.6331391, -7.8405972, 8.0736084
5: -3.5509248, 3.0351212, -4.1341681, 3.4774535, -7.0283756, 7.1692882
6: -3.4747064, 3.3891344, -3.9999735, 3.9023623, -7.3770685, 7.3891077
7: -3.6690035, 3.5624459, -4.2484536, 4.1346374, -7.8036404, 7.8108997
8: -5.7219377, 3.3660746, -6.5484333, 3.5591998, -9.2811375, 9.9145050
9: -3.1934023, 3.4740181, -3.7089088, 3.9868102, -7.1802106, 7.1829252

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=32, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=76, inp2_unstable=74, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=218, inp2_unstable=223, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8518155, upper bound: 10.8518158
time: 2.93 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8518157, upper bound: 10.8518151
time: 3.75 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 8.23 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 8.23
Output dim: 8, lower bound: -10.8503477, upper bound: 10.8499491
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 8.23
Output dim: 8, lower bound: -10.8502464, upper bound: 10.8499610
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 8.23
Output dim: 8, lower bound: -10.8518155, upper bound: 10.8518158
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 8.23
Output dim: 8, lower bound: -10.8518157, upper bound: 10.8518151

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -2.2002499, 1.8494973, -2.1959443, 1.8494157, -4.0496655, 4.0454416
1: -1.7733535, 1.8472070, -1.7841898, 1.8482751, -3.6216285, 3.6313968
2: -2.2915225, 2.0668805, -2.2679930, 2.0180104, -4.3095331, 4.3348732
3: -2.4736643, 1.8531780, -2.5018129, 1.8370330, -4.3106976, 4.3549910
4: -2.4504862, 1.9372973, -2.4706442, 1.9357620, -4.3862481, 4.4079418
5: -2.0246263, 1.9095783, -2.0459325, 1.9088012, -3.9334273, 3.9555109
6: -2.0795374, 2.0387869, -2.0647454, 2.0414233, -4.1209607, 4.1035323
7: -2.1562178, 2.0801592, -2.1630886, 2.0999374, -4.2561550, 4.2432480
8: -3.4075100, 2.8226676, -3.4037845, 2.7859077, -6.1934175, 6.2264519
9: -1.8519459, 2.1501296, -1.8584915, 2.1539917, -4.0059376, 4.0086212

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=73, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=180, inp2_unstable=181, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8424213, upper bound: 10.8434255
time: 2.36 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8486822, upper bound: 10.8485066
time: 16.17 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -2.2280517, 1.8731461, -3.1581256, 2.6235197, -4.8515711, 5.0312719
1: -1.7951037, 1.8628588, -2.6157122, 2.5433664, -4.3384700, 4.4785709
2: -2.3322520, 2.1031368, -3.4782815, 2.7169085, -5.0491605, 5.5814180
3: -2.5081353, 1.8787794, -3.7717891, 2.4278023, -4.9359379, 5.6505685
4: -2.4774613, 1.9590179, -3.5940578, 2.7389915, -5.2164526, 5.5530758
5: -2.0479903, 1.9299963, -3.0376110, 2.6296663, -4.6776567, 4.9676075
6: -2.1109776, 2.0624204, -2.9737740, 2.9179006, -5.0288782, 5.0361943
7: -2.1822233, 2.1005418, -3.1487684, 3.0676262, -5.2498493, 5.2493105
8: -3.4519701, 2.8444986, -4.9294882, 3.1114371, -6.5634069, 7.7739868
9: -1.8751402, 2.1744518, -2.7186060, 3.0086436, -4.8837838, 4.8930578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=77, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=15, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=181, inp2_unstable=210, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8445317, upper bound: 10.8438977
time: 3.41 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8488833, upper bound: 10.8485824
time: 2.90 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -3.3688412, 2.7963219, -3.3286576, 2.7630975, -6.1319389, 6.1249795
1: -2.8201103, 2.7117250, -2.7905605, 2.6969743, -5.5170846, 5.5022855
2: -3.7471359, 2.9100559, -3.7167680, 2.8840227, -6.6311588, 6.6268239
3: -4.0435085, 2.6006446, -4.0307322, 2.6342831, -6.6777916, 6.6313767
4: -3.8418293, 2.9227676, -3.8224938, 2.9042172, -6.7460465, 6.7452612
5: -3.2423074, 2.8038058, -3.2072144, 2.7919540, -6.0342617, 6.0110202
6: -3.1885350, 3.1158218, -3.1588957, 3.0946722, -6.2832069, 6.2747173
7: -3.3627625, 3.2607331, -3.3400431, 3.2417228, -6.6044855, 6.6007762
8: -5.2658396, 3.2297688, -5.1990910, 3.1431789, -8.4090185, 8.4288597
9: -2.9166121, 3.1996279, -2.8903782, 3.1734743, -6.0900865, 6.0900059

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=215, inp2_unstable=213, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8503068, upper bound: 10.8504488
time: 3.98 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8503132, upper bound: 10.8503132
time: 3.70 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -3.5490742, 2.9464583, -3.9284887, 3.2581851, -6.8072596, 6.8749471
1: -2.9921305, 2.8407619, -3.3609047, 3.1240208, -6.1161513, 6.2016668
2: -3.9625013, 3.0341249, -4.4342556, 3.2951584, -7.2576599, 7.4683805
3: -4.2803764, 2.7034256, -4.8113265, 2.9673696, -7.2477460, 7.5147524
4: -4.0620694, 3.0713229, -4.5561419, 3.3961427, -7.4582119, 7.6274648
5: -3.4269443, 2.9435132, -3.8322906, 3.2551899, -6.6821342, 6.7758036
6: -3.3605032, 3.2806571, -3.7256153, 3.6402977, -7.0008011, 7.0062723
7: -3.5474536, 3.4421592, -3.9574881, 3.8445921, -7.3920460, 7.3996472
8: -5.5421901, 3.3089881, -6.1150780, 3.4012477, -8.9434376, 9.4240665
9: -3.0825191, 3.3642206, -3.4407995, 3.7187929, -6.8013120, 6.8050203

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=73, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=216, inp2_unstable=219, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8503068, upper bound: 10.8504489
time: 2.49 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8503127, upper bound: 10.8503136
time: 2.48 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 6.45 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.45
Output dim: 8, lower bound: -10.8424213, upper bound: 10.8434255
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.45
Output dim: 8, lower bound: -10.8486822, upper bound: 10.8485066
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.45
Output dim: 8, lower bound: -10.8445317, upper bound: 10.8438977
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.45
Output dim: 8, lower bound: -10.8488833, upper bound: 10.8485824
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.45
Output dim: 8, lower bound: -10.8503068, upper bound: 10.8504488
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.45
Output dim: 8, lower bound: -10.8503132, upper bound: 10.8503132
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.45
Output dim: 8, lower bound: -10.8503068, upper bound: 10.8504489
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.45
Output dim: 8, lower bound: -10.8503127, upper bound: 10.8503136

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.4043604, 0.6074980, -1.2354671, 1.1876043, -1.5919647, 1.8429651
1: -0.3585974, 0.4519699, -1.0204153, 1.1403959, -1.4989933, 1.4723852
2: -0.4743955, 0.6384621, -1.1290644, 1.3493106, -1.8237062, 1.7675265
3: -0.2517788, 0.6248695, -1.2007763, 1.2570050, -1.5087838, 1.8256458
4: -0.4609523, 0.4615960, -1.3226705, 1.1801713, -1.6411235, 1.7842665
5: -0.4011335, 0.5442472, -1.1476113, 1.1961589, -1.5972924, 1.6918585
6: -0.3823066, 0.4923779, -1.1841390, 1.1812449, -1.5635514, 1.6765169
7: -0.4422995, 0.4368832, -1.2010869, 1.1929245, -1.6352240, 1.6379700
8: -0.2330061, 2.4567294, -1.7685511, 2.5993152, -2.8323212, 4.2252808
9: -0.6012856, 0.6008418, -1.1002686, 1.3244699, -1.9257555, 1.7011104

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=31, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=44, inp2_unstable=140, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8424213, upper bound: 10.8434255
time: 2.03 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8424213, upper bound: 10.8434255
time: 2.05 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1.1180259, 1.1264412, -1.7272825, 1.5123899, -2.6304159, 2.8537238
1: -0.9244901, 1.0602522, -1.4019494, 1.4976541, -2.4221442, 2.4622016
2: -1.0329329, 1.3534248, -1.6805153, 1.7028482, -2.7357812, 3.0339401
3: -1.0377450, 1.1797954, -1.8439873, 1.5485891, -2.5863342, 3.0237827
4: -1.1730082, 1.0821362, -1.9126120, 1.5553848, -2.7283931, 2.9947481
5: -1.0305799, 1.1206187, -1.5983318, 1.5595558, -2.5901356, 2.7189505
6: -1.0965043, 1.0856285, -1.6343329, 1.6135590, -2.7100635, 2.7199614
7: -1.0725073, 1.0902597, -1.6767253, 1.6405287, -2.7130361, 2.7669849
8: -1.5849450, 2.6341076, -2.6162119, 2.6901593, -4.2751045, 5.2503195
9: -1.0289613, 1.2346197, -1.4623364, 1.7395637, -2.7685251, 2.6969562

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=31, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=69, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=127, inp2_unstable=167, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8486822, upper bound: 10.8485066
time: 2.35 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8486822, upper bound: 10.8485065
time: 4.88 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.3856851, 0.5910726, -1.9633067, 1.6912036, -2.0768886, 2.5543792
1: -0.3405835, 0.4391911, -1.5852257, 1.6591485, -1.9997320, 2.0244169
2: -0.4695427, 0.6303051, -1.9753512, 1.8627406, -2.3322835, 2.6056564
3: -0.2402608, 0.5979753, -2.1439354, 1.6410222, -1.8812829, 2.7419107
4: -0.4480854, 0.4362407, -2.2015350, 1.7349483, -2.1830337, 2.6377757
5: -0.3813739, 0.5261358, -1.8145806, 1.7253901, -2.1067638, 2.3407164
6: -0.3630418, 0.4719749, -1.8448713, 1.8254797, -2.1885216, 2.3168461
7: -0.4273925, 0.4155157, -1.9200816, 1.8643911, -2.2917836, 2.3355973
8: -0.2082924, 2.4583650, -3.0216718, 2.7501736, -2.9584661, 5.4800367
9: -0.5943607, 0.5857191, -1.6483594, 1.9395258, -2.5338864, 2.2340784

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=31, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=74, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=10, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=42, inp2_unstable=172, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8445317, upper bound: 10.8438977
time: 2.87 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8445317, upper bound: 10.8438977
time: 2.10 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1.1789749, 1.1638298, -2.6801507, 2.2314696, -3.4104445, 3.8439806
1: -0.9667869, 1.0998735, -2.1717014, 2.1877542, -3.1545410, 3.2715750
2: -1.0958297, 1.4128406, -2.8918145, 2.3735299, -3.4693596, 4.3046551
3: -1.1019998, 1.2109493, -3.1258428, 2.1217892, -3.2237890, 4.3367920
4: -1.2424214, 1.1246908, -3.0224693, 2.3357372, -3.5781586, 4.1471601
5: -1.0822704, 1.1657479, -2.5426488, 2.2626519, -3.3449223, 3.7083967
6: -1.1547188, 1.1350241, -2.5126836, 2.4778674, -3.6325860, 3.6477077
7: -1.1257610, 1.1389929, -2.6603332, 2.5797882, -3.7055492, 3.7993259
8: -1.6918225, 2.6506441, -4.1909132, 2.9516060, -4.6434288, 6.8415575
9: -1.0720564, 1.2819602, -2.2832646, 2.5779598, -3.6500163, 3.5652249

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=31, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=76, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=14, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=132, inp2_unstable=199, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8488833, upper bound: 10.8485824
time: 7.44 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8488825, upper bound: 10.8485825
time: 3.25 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -1.4660666, 1.3306402, -2.4945865, 2.0806525, -3.5467191, 3.8252268
1: -1.1943853, 1.3196143, -2.0271344, 2.0855772, -3.2799625, 3.3467488
2: -1.3775412, 1.5532612, -2.6780229, 2.2756550, -3.6531963, 4.2312841
3: -1.4619946, 1.3992000, -2.9258416, 2.1042838, -3.5662785, 4.3250418
4: -1.5892677, 1.3516893, -2.8155012, 2.1995566, -3.7888243, 4.1671906
5: -1.3517020, 1.3700330, -2.3485756, 2.1487880, -3.5004900, 3.7186086
6: -1.4060488, 1.3793380, -2.3565466, 2.3250558, -3.7311046, 3.7358847
7: -1.4082381, 1.3890175, -2.4774411, 2.3985724, -3.8068104, 3.8664584
8: -2.1806726, 2.7064040, -3.8938568, 2.8483226, -5.0289955, 6.6002607
9: -1.2739632, 1.5111173, -2.1251023, 2.4251099, -3.6990731, 3.6362195

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=31, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=73, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=191, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8457065, upper bound: 10.8463458
time: 2.89 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8489356, upper bound: 10.8488348
time: 3.40 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -2.2922516, 1.9298626, -2.5683627, 2.1405647, -4.4328165, 4.4982252
1: -1.8519363, 1.9171284, -2.0859141, 2.1358392, -3.9877756, 4.0030427
2: -2.4126348, 2.1361301, -2.7798667, 2.3453226, -4.7579575, 4.9159966
3: -2.5911894, 1.8943267, -3.0205579, 2.1531053, -4.7442946, 4.9148846
4: -2.5626817, 2.0129530, -2.8987851, 2.2618291, -4.8245106, 4.9117384
5: -2.1262560, 1.9796728, -2.4200633, 2.2035143, -4.3297701, 4.3997359
6: -2.1675062, 2.1254070, -2.4291143, 2.3918743, -4.5593805, 4.5545216
7: -2.2530234, 2.1735835, -2.5539036, 2.4690247, -4.7220478, 4.7274871
8: -3.5801113, 2.9053893, -4.0128927, 2.8802731, -6.4603844, 6.9182820
9: -1.9425740, 2.2366657, -2.1901360, 2.4907141, -4.4332881, 4.4268017

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=31, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=76, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=183, inp2_unstable=196, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8459213, upper bound: 10.8465889
time: 9.49 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8489859, upper bound: 10.8489859
time: 6.35 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1.6158303, 1.4366711, -3.0788708, 2.5503294, -4.1661596, 4.5155420
1: -1.3121722, 1.4261323, -2.5512300, 2.5004582, -3.8126304, 3.9773622
2: -1.5528413, 1.6479323, -3.3890772, 2.6688333, -4.2216744, 5.0370092
3: -1.6668420, 1.4803703, -3.6905553, 2.4291778, -4.0960197, 5.1709256
4: -1.7682292, 1.4690592, -3.5102134, 2.6858397, -4.4540691, 4.9792728
5: -1.4909390, 1.4780560, -2.9581332, 2.5828500, -4.0737891, 4.4361892
6: -1.5394132, 1.5126630, -2.9026892, 2.8553209, -4.3947344, 4.4153523
7: -1.5587735, 1.5258867, -3.0783267, 2.9921374, -4.5509109, 4.6042132
8: -2.4388802, 2.7437100, -4.8086381, 3.0426402, -5.4815207, 7.5523481
9: -1.3834261, 1.6425891, -2.6507969, 2.9415884, -4.3250146, 4.2933860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=31, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=73, inp2_unstable=73, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=14, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=164, inp2_unstable=211, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8457065, upper bound: 10.8463458
time: 2.66 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8489356, upper bound: 10.8488349
time: 3.12 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -2.4541075, 2.0567029, -3.1128478, 2.5793397, -5.0334473, 5.1695509
1: -1.9851649, 2.0307779, -2.5765743, 2.5216517, -4.5068169, 4.6073523
2: -2.6125124, 2.2412968, -3.4355431, 2.7081873, -5.3206997, 5.6768398
3: -2.8077796, 1.9896080, -3.7323115, 2.4569068, -5.2646866, 5.7219195
4: -2.7491951, 2.1473477, -3.5443130, 2.7126527, -5.4618478, 5.6916609
5: -2.2975371, 2.0971487, -2.9878120, 2.6087832, -4.9063206, 5.0849609
6: -2.3137426, 2.2715616, -2.9393582, 2.8845601, -5.1983027, 5.2109199
7: -2.4204097, 2.3344600, -3.1098335, 3.0186715, -5.4390812, 5.4442935
8: -3.8452392, 2.9574025, -4.8598680, 3.0699134, -6.9151525, 7.8172703
9: -2.0841813, 2.3789923, -2.6825035, 2.9702644, -5.0544457, 5.0614958

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=31, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=76, inp2_unstable=73, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=193, inp2_unstable=211, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8459213, upper bound: 10.8465889
time: 2.27 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8489859, upper bound: 10.8489859
time: 3.68 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 7.53 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.53
Output dim: 8, lower bound: -10.8424213, upper bound: 10.8434255
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.53
Output dim: 8, lower bound: -10.8424213, upper bound: 10.8434255
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.53
Output dim: 8, lower bound: -10.8486822, upper bound: 10.8485066
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.53
Output dim: 8, lower bound: -10.8486822, upper bound: 10.8485065
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.53
Output dim: 8, lower bound: -10.8445317, upper bound: 10.8438977
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.53
Output dim: 8, lower bound: -10.8445317, upper bound: 10.8438977
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.53
Output dim: 8, lower bound: -10.8488833, upper bound: 10.8485824
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.53
Output dim: 8, lower bound: -10.8488825, upper bound: 10.8485825
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.53
Output dim: 8, lower bound: -10.8457065, upper bound: 10.8463458
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.53
Output dim: 8, lower bound: -10.8489356, upper bound: 10.8488348
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.53
Output dim: 8, lower bound: -10.8459213, upper bound: 10.8465889
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.53
Output dim: 8, lower bound: -10.8489859, upper bound: 10.8489859
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.53
Output dim: 8, lower bound: -10.8457065, upper bound: 10.8463458
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.53
Output dim: 8, lower bound: -10.8489356, upper bound: 10.8488349
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.53
Output dim: 8, lower bound: -10.8459213, upper bound: 10.8465889
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.53
Output dim: 8, lower bound: -10.8489859, upper bound: 10.8489859

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.3579350, 0.5515869, -0.6427993, 0.7935848, -1.1515198, 1.1943861
1: -0.3141608, 0.4159026, -0.5836922, 0.7056606, -1.0198214, 0.9995947
2: -0.4479609, 0.5775763, -0.5916769, 0.9196732, -1.3676341, 1.1692531
3: -0.2188401, 0.5660433, -0.4959909, 0.9330486, -1.1518887, 1.0620341
4: -0.4263678, 0.4005649, -0.6691750, 0.7098185, -1.1361864, 1.0697398
5: -0.3559325, 0.4934884, -0.6347641, 0.7563028, -1.1122353, 1.1282525
6: -0.3316242, 0.4413666, -0.6491286, 0.6853328, -1.0169570, 1.0904952
7: -0.4042892, 0.3808291, -0.6589234, 0.6687571, -1.0730462, 1.0397525
8: -0.1215185, 2.4122949, -0.7070176, 2.4705968, -2.5921154, 3.1193125
9: -0.5719382, 0.5488146, -0.7179431, 0.8389457, -1.4108839, 1.2667577

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=68, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=69, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8037597, upper bound: 10.8340135
time: 2.40 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7756909, upper bound: 10.8295545
time: 1.66 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.3842893, 0.5848309, -1.0370927, 1.0584066, -1.4426959, 1.6219236
1: -0.3400367, 0.4364212, -0.8725663, 1.0042590, -1.3442957, 1.3089875
2: -0.4631833, 0.6139858, -0.9323015, 1.2177321, -1.6809154, 1.5462873
3: -0.2370567, 0.6010800, -0.9663225, 1.1552762, -1.3923329, 1.5674025
4: -0.4464871, 0.4362318, -1.0857419, 1.0302024, -1.4766896, 1.5219737
5: -0.3807752, 0.5240986, -0.9678369, 1.0516133, -1.4323885, 1.4919355
6: -0.3596447, 0.4724005, -1.0046282, 1.0131295, -1.3727741, 1.4770287
7: -0.4266691, 0.4143503, -1.0115947, 1.0261889, -1.4528580, 1.4259450
8: -0.1878986, 2.4383802, -1.4165478, 2.5469623, -2.7348609, 3.8549280
9: -0.5893306, 0.5785383, -0.9658356, 1.1627080, -1.7520386, 1.5443740

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=120, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8037597, upper bound: 10.8340135
time: 2.62 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7756909, upper bound: 10.8295545
time: 2.70 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.9146944, 0.9973972, -0.9900287, 1.0262423, -1.9409366, 1.9874259
1: -0.7801512, 0.9220710, -0.8501695, 0.9882913, -1.7684425, 1.7722405
2: -0.8305516, 1.2203692, -0.8944461, 1.2269726, -2.0575242, 2.1148152
3: -0.7997999, 1.0736260, -0.9015329, 1.1724048, -1.9722047, 1.9751589
4: -0.9459305, 0.9256461, -1.0256798, 0.9955359, -1.9414663, 1.9513259
5: -0.8480436, 0.9795129, -0.9234488, 1.0309503, -1.8789940, 1.9029617
6: -0.9067557, 0.9218251, -0.9763470, 0.9809557, -1.8877114, 1.8981720
7: -0.8901040, 0.9158691, -0.9632375, 0.9811059, -1.8712099, 1.8791065
8: -1.2376401, 2.5790095, -1.3488300, 2.5406675, -3.7783077, 3.9278395
9: -0.8940834, 1.0734980, -0.9395475, 1.1308941, -2.0249774, 2.0130455

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=69, inp2_unstable=69, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=108, inp2_unstable=113, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 92

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8478681, upper bound: 10.8479126
time: 3.40 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8477468, upper bound: 10.8475180
time: 3.06 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -1.0372276, 1.0753977, -1.5077183, 1.3510473, -2.3882749, 2.5831161
1: -0.8667643, 1.0060343, -1.2295378, 1.3428172, -2.2095814, 2.2355721
2: -0.9525806, 1.3011525, -1.4149162, 1.5615077, -2.5140882, 2.7160687
3: -0.9445344, 1.1371448, -1.5413941, 1.4286326, -2.3731670, 2.6785388
4: -1.0802345, 1.0207359, -1.6487494, 1.3825060, -2.4627404, 2.6694851
5: -0.9561310, 1.0653993, -1.3922055, 1.4002357, -2.3563666, 2.4576049
6: -1.0227489, 1.0189084, -1.4355237, 1.4157407, -2.4384897, 2.4544320
7: -0.9984612, 1.0226597, -1.4560542, 1.4344676, -2.4329288, 2.4787140
8: -1.4457150, 2.6120191, -2.2360191, 2.6287494, -4.0744643, 4.8480382
9: -0.9747210, 1.1700163, -1.2980397, 1.5450824, -2.5198035, 2.4680560

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=69, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=119, inp2_unstable=159, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8478681, upper bound: 10.8479125
time: 2.77 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8477468, upper bound: 10.8475180
time: 3.20 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.3426904, 0.5365787, -1.2368059, 1.1900363, -1.5327266, 1.7733846
1: -0.2979554, 0.4031311, -1.0171199, 1.1470044, -1.4449598, 1.4202510
2: -0.4447793, 0.5708351, -1.1403388, 1.4009376, -1.8457168, 1.7111739
3: -0.2099150, 0.5406694, -1.1902831, 1.2585814, -1.4684963, 1.7309525
4: -0.4146934, 0.3815612, -1.3248882, 1.1722932, -1.5869865, 1.7064495
5: -0.3401187, 0.4758513, -1.1407405, 1.2053987, -1.5455174, 1.6165918
6: -0.3183238, 0.4213136, -1.1957824, 1.1847206, -1.5030444, 1.6170959
7: -0.3903328, 0.3615949, -1.1934103, 1.1953206, -1.5856534, 1.5550051
8: -0.1010466, 2.4146442, -1.7772872, 2.5895033, -2.6905499, 4.1919317
9: -0.5666758, 0.5350150, -1.1073307, 1.3254876, -1.8921634, 1.6423457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=137, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8094147, upper bound: 10.8362946
time: 2.93 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7817172, upper bound: 10.8317979
time: 3.18 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.3682125, 0.5683104, -1.7362462, 1.5238633, -1.8920758, 2.3045566
1: -0.3232426, 0.4233845, -1.4003532, 1.4933151, -1.8165576, 1.8237377
2: -0.4587233, 0.6063846, -1.6975253, 1.7141773, -2.1729007, 2.3039098
3: -0.2272792, 0.5744941, -1.8310280, 1.5153861, -1.7426653, 2.4055221
4: -0.4339822, 0.4127258, -1.9292423, 1.5541646, -1.9881468, 2.3419681
5: -0.3637002, 0.5057818, -1.6010900, 1.5591528, -1.9228530, 2.1068716
6: -0.3427901, 0.4517881, -1.6368273, 1.6176456, -1.9604357, 2.0886154
7: -0.4117745, 0.3933746, -1.6862175, 1.6482459, -2.0600204, 2.0795922
8: -0.1636088, 2.4401746, -2.6282306, 2.6848936, -2.8485024, 5.0684052
9: -0.5827428, 0.5639079, -1.4698381, 1.7370129, -2.3197556, 2.0337460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=73, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=166, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8094147, upper bound: 10.8362946
time: 2.61 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7817172, upper bound: 10.8317979
time: 3.47 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.9684784, 1.0319247, -1.9023263, 1.6345156, -2.6029940, 2.9342511
1: -0.8171946, 0.9586614, -1.5347983, 1.6278535, -2.4450481, 2.4934597
2: -0.8871958, 1.2770023, -1.9159312, 1.8681345, -2.7553303, 3.1929336
3: -0.8569835, 1.1007984, -2.0699129, 1.6739291, -2.5309126, 3.1707113
4: -1.0025612, 0.9638014, -2.1109936, 1.6911463, -2.6937075, 3.0747950
5: -0.8919826, 1.0204520, -1.7497840, 1.6919507, -2.5839334, 2.7702360
6: -0.9608990, 0.9647748, -1.8021601, 1.7698720, -2.7307711, 2.7669349
7: -0.9337769, 0.9621022, -1.8481581, 1.8033831, -2.7371600, 2.8102603
8: -1.3365197, 2.5948565, -2.9056053, 2.7149687, -4.0514884, 5.5004616
9: -0.9310005, 1.1155852, -1.5978429, 1.8925118, -2.8235123, 2.7134280

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=74, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=112, inp2_unstable=167, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8481978, upper bound: 10.8481799
time: 2.44 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8481303, upper bound: 10.8476902
time: 2.46 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -1.0931306, 1.1099932, -2.4216881, 2.0273552, -3.1204858, 3.5316813
1: -0.9051741, 1.0429332, -1.9645312, 2.0035610, -2.9087353, 3.0074644
2: -1.0116602, 1.3582058, -2.5774493, 2.2062850, -3.2179451, 3.9356551
3: -1.0026276, 1.1655509, -2.7872708, 1.9718709, -2.9744985, 3.9528217
4: -1.1430602, 1.0599892, -2.7309875, 2.1220613, -3.2651215, 3.7909768
5: -1.0041709, 1.1058916, -2.2719655, 2.0781503, -3.0823212, 3.3778572
6: -1.0771387, 1.0645186, -2.2770209, 2.2462416, -3.3233802, 3.3415394
7: -1.0446502, 1.0692073, -2.4002943, 2.3223333, -3.3669834, 3.4695015
8: -1.5467176, 2.6279294, -3.7824099, 2.8569694, -4.4036870, 6.4103394
9: -1.0130436, 1.2137399, -2.0586729, 2.3515041, -3.3645477, 3.2724128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=124, inp2_unstable=188, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8481978, upper bound: 10.8481796
time: 5.23 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8481302, upper bound: 10.8476904
time: 2.76 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.6792385, 0.8392280, -0.4719619, 0.6476282, -1.3268666, 1.3111899
1: -0.6068542, 0.7312909, -0.4322266, 0.5303035, -1.1371577, 1.1635175
2: -0.6275367, 0.9632816, -0.4932367, 0.7079936, -1.3355303, 1.4565182
3: -0.5380987, 0.9323756, -0.3088719, 0.7531818, -1.2912805, 1.2412474
4: -0.7077425, 0.7432685, -0.5083756, 0.5426600, -1.2504025, 1.2516441
5: -0.6659219, 0.7875402, -0.4691519, 0.6092744, -1.2751963, 1.2566922
6: -0.6862788, 0.7175530, -0.4604885, 0.5465732, -1.2328520, 1.1780415
7: -0.6931224, 0.7079447, -0.5029935, 0.5027412, -1.1958635, 1.2109382
8: -0.7972792, 2.5795188, -0.3542492, 2.4153016, -3.2125807, 2.9337680
9: -0.7530746, 0.8765873, -0.6221384, 0.6653582, -1.4184328, 1.4987257

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=68, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=74, inp2_unstable=49, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8407394, upper bound: 10.8239898
time: 1.98 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8404607, upper bound: 10.8229561
time: 2.10 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -1.0345616, 1.0706791, -1.4052455, 1.2792685, -2.3138301, 2.4759245
1: -0.8745953, 1.0141129, -1.1497573, 1.2722820, -2.1468773, 2.1638703
2: -0.9465399, 1.2643640, -1.3115633, 1.5333282, -2.4798679, 2.5759273
3: -0.9537855, 1.1642642, -1.3863803, 1.3818971, -2.3356826, 2.5506444
4: -1.0760713, 1.0300677, -1.5194933, 1.2998794, -2.3759508, 2.5495610
5: -0.9626806, 1.0615180, -1.2914971, 1.3297122, -2.2923927, 2.3530149
6: -1.0212004, 1.0191028, -1.3491627, 1.3248078, -2.3460083, 2.3682656
7: -1.0026731, 1.0198544, -1.3494828, 1.3376110, -2.3402841, 2.3693371
8: -1.4450341, 2.6490076, -2.0605826, 2.6076179, -4.0526519, 4.7095900
9: -0.9741148, 1.1736236, -1.2241430, 1.4573127, -2.4314275, 2.3977666

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=73, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=118, inp2_unstable=149, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8483221, upper bound: 10.8485347
time: 2.29 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8483099, upper bound: 10.8481744
time: 2.85 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -1.2070260, 1.1883464, -0.4581404, 0.6328316, -1.8398576, 1.6464868
1: -0.9903581, 1.1203762, -0.4178665, 0.5154772, -1.5058353, 1.5382427
2: -1.1170475, 1.3852674, -0.4899753, 0.7057483, -1.8227959, 1.8752427
3: -1.1516836, 1.2174633, -0.2964560, 0.7334941, -1.8851776, 1.5139192
4: -1.2838676, 1.1513475, -0.4984052, 0.5259996, -1.8098671, 1.6497527
5: -1.1141745, 1.1821733, -0.4523298, 0.5975058, -1.7116803, 1.6345030
6: -1.1717650, 1.1617544, -0.4431474, 0.5337437, -1.7055087, 1.6049018
7: -1.1624470, 1.1669390, -0.4892989, 0.4894153, -1.6518623, 1.6562378
8: -1.7440765, 2.6920638, -0.3366221, 2.4169292, -4.1610060, 3.0286858
9: -1.0915645, 1.3082197, -0.6159739, 0.6535441, -1.7451086, 1.9241936

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=74, inp2_unstable=68, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=6, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=137, inp2_unstable=46, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8413274, upper bound: 10.8258480
time: 3.92 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8412345, upper bound: 10.8256224
time: 2.18 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -1.8532994, 1.6134348, -1.5194244, 1.3573529, -3.2106524, 3.1328592
1: -1.4939909, 1.5882030, -1.2384129, 1.3503897, -2.8443806, 2.8266158
2: -1.8598261, 1.8402951, -1.4479218, 1.6249522, -3.4847784, 3.2882168
3: -1.9708375, 1.6062328, -1.5398408, 1.4563192, -3.4271567, 3.1460736
4: -2.0474870, 1.6506445, -1.6509537, 1.3879820, -3.4354692, 3.3015981
5: -1.7050924, 1.6493380, -1.3914838, 1.4151889, -3.1202812, 3.0408218
6: -1.7614129, 1.7226501, -1.4534187, 1.4258064, -3.1872191, 3.1760688
7: -1.7928543, 1.7455165, -1.4616591, 1.4379797, -3.2308340, 3.2071757
8: -2.8436840, 2.8095455, -2.2512667, 2.6323557, -5.4760399, 5.0608120
9: -1.5655336, 1.8480453, -1.3077682, 1.5559430, -3.1214767, 3.1558137

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=76, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=9, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=167, inp2_unstable=158, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8483825, upper bound: 10.8487566
time: 2.56 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8483771, upper bound: 10.8483771
time: 2.57 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.7635387, 0.9058533, -0.6758897, 0.8345136, -1.5980523, 1.5817430
1: -0.6710826, 0.8020365, -0.6007124, 0.7196300, -1.3907125, 1.4027489
2: -0.6932665, 1.0360627, -0.6030082, 0.9455654, -1.6388319, 1.6390710
3: -0.6374953, 0.9901159, -0.5366870, 0.9124663, -1.5499617, 1.5268030
4: -0.7888569, 0.8181093, -0.7047047, 0.7383931, -1.5272501, 1.5228140
5: -0.7331417, 0.8598328, -0.6578687, 0.7853696, -1.5185113, 1.5177015
6: -0.7609805, 0.7901736, -0.6631619, 0.7113469, -1.4723274, 1.4533355
7: -0.7712575, 0.7872962, -0.6947887, 0.7113934, -1.4826509, 1.4820849
8: -0.9575533, 2.6085939, -0.7691084, 2.4826326, -3.4401860, 3.3777022
9: -0.8055106, 0.9498602, -0.7360113, 0.8625315, -1.6680422, 1.6858714

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=69, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=89, inp2_unstable=78, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8407394, upper bound: 10.8239898
time: 2.19 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8404607, upper bound: 10.8229561
time: 3.75 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1.1692578, 1.1554668, -1.9133472, 1.6365958, -2.8058536, 3.0688140
1: -0.9723853, 1.1047943, -1.5462160, 1.6303854, -2.6027708, 2.6510103
2: -1.0786647, 1.3524439, -1.9219112, 1.8567576, -2.9354224, 3.2743552
3: -1.1103325, 1.2336708, -2.0869658, 1.6543432, -2.7646756, 3.3206367
4: -1.2343962, 1.1319830, -2.1329875, 1.6984929, -2.9328890, 3.2649705
5: -1.0861676, 1.1575463, -1.7638556, 1.6962368, -2.7824044, 2.9214020
6: -1.1422145, 1.1314278, -1.8026075, 1.7777807, -2.9199953, 2.9340353
7: -1.1290026, 1.1324182, -1.8656248, 1.8165724, -2.9455750, 2.9980431
8: -1.6791956, 2.6813638, -2.9255247, 2.7120457, -4.3912411, 5.6068888
9: -1.0646073, 1.2815963, -1.6030964, 1.9002303, -2.9648376, 2.8846927

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=73, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=134, inp2_unstable=171, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8483221, upper bound: 10.8485341
time: 3.96 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8483099, upper bound: 10.8481744
time: 2.34 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -1.3518485, 1.2789779, -0.6400032, 0.8054264, -2.1572747, 1.9189811
1: -1.0999537, 1.2194175, -0.5688674, 0.6889983, -1.7889520, 1.7882849
2: -1.2611611, 1.4767500, -0.5791275, 0.9205493, -2.1817102, 2.0558774
3: -1.3220115, 1.2919657, -0.4914818, 0.8787404, -2.2007518, 1.7834475
4: -1.4555404, 1.2600650, -0.6659518, 0.7015733, -2.1571136, 1.9260168
5: -1.2441812, 1.2841363, -0.6237407, 0.7517011, -1.9958823, 1.9078770
6: -1.3003306, 1.2826148, -0.6284815, 0.6844633, -1.9847939, 1.9110963
7: -1.2989641, 1.2887585, -0.6580283, 0.6697225, -1.9686866, 1.9467869
8: -1.9934293, 2.7261720, -0.7007135, 2.4838147, -4.4772439, 3.4268856
9: -1.1918334, 1.4212428, -0.7118984, 0.8324041, -2.0242376, 2.1331413

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=74, inp2_unstable=69, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=146, inp2_unstable=71, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8413274, upper bound: 10.8258480
time: 1.98 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8412345, upper bound: 10.8256224
time: 1.93 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -2.0062315, 1.7255309, -2.0007687, 1.7010241, -3.7072556, 3.7262995
1: -1.6156466, 1.6984586, -1.6158538, 1.6915816, -3.3072281, 3.3143125
2: -2.0448954, 1.9389292, -2.0379157, 1.9306110, -3.9755063, 3.9768448
3: -2.1843436, 1.6935439, -2.2058797, 1.7166601, -3.9010038, 3.8994236
4: -2.2300267, 1.7726552, -2.2317688, 1.7679502, -3.9979768, 4.0044241
5: -1.8494927, 1.7609714, -1.8419056, 1.7622378, -3.6117306, 3.6028771
6: -1.9021021, 1.8621922, -1.8868501, 1.8576339, -3.7597361, 3.7490423
7: -1.9531038, 1.8900838, -1.9551194, 1.8965813, -3.8496852, 3.8452032
8: -3.1061475, 2.8515821, -3.0724049, 2.7401683, -5.8463159, 5.9239869
9: -1.6866269, 1.9836403, -1.6734636, 1.9777030, -3.6643300, 3.6571040

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=76, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=171, inp2_unstable=172, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8483825, upper bound: 10.8487559
time: 6.78 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8483771, upper bound: 10.8483771
time: 2.76 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 11.10 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.10
Output dim: 8, lower bound: -10.8037597, upper bound: 10.8340135
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.10
Output dim: 8, lower bound: -10.7756909, upper bound: 10.8295545
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.10
Output dim: 8, lower bound: -10.8037597, upper bound: 10.8340135
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.10
Output dim: 8, lower bound: -10.7756909, upper bound: 10.8295545
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.10
Output dim: 8, lower bound: -10.8478681, upper bound: 10.8479126
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.10
Output dim: 8, lower bound: -10.8477468, upper bound: 10.8475180
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.10
Output dim: 8, lower bound: -10.8478681, upper bound: 10.8479125
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.10
Output dim: 8, lower bound: -10.8477468, upper bound: 10.8475180
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.10
Output dim: 8, lower bound: -10.8094147, upper bound: 10.8362946
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.10
Output dim: 8, lower bound: -10.7817172, upper bound: 10.8317979
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.10
Output dim: 8, lower bound: -10.8094147, upper bound: 10.8362946
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.10
Output dim: 8, lower bound: -10.7817172, upper bound: 10.8317979
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.10
Output dim: 8, lower bound: -10.8481978, upper bound: 10.8481799
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.10
Output dim: 8, lower bound: -10.8481303, upper bound: 10.8476902
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.10
Output dim: 8, lower bound: -10.8481978, upper bound: 10.8481796
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.10
Output dim: 8, lower bound: -10.8481302, upper bound: 10.8476904
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.10
Output dim: 8, lower bound: -10.8407394, upper bound: 10.8239898
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.10
Output dim: 8, lower bound: -10.8404607, upper bound: 10.8229561
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.10
Output dim: 8, lower bound: -10.8483221, upper bound: 10.8485347
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.10
Output dim: 8, lower bound: -10.8483099, upper bound: 10.8481744
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.10
Output dim: 8, lower bound: -10.8413274, upper bound: 10.8258480
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.10
Output dim: 8, lower bound: -10.8412345, upper bound: 10.8256224
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.10
Output dim: 8, lower bound: -10.8483825, upper bound: 10.8487566
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.10
Output dim: 8, lower bound: -10.8483771, upper bound: 10.8483771
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.10
Output dim: 8, lower bound: -10.8407394, upper bound: 10.8239898
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.10
Output dim: 8, lower bound: -10.8404607, upper bound: 10.8229561
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.10
Output dim: 8, lower bound: -10.8483221, upper bound: 10.8485341
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.10
Output dim: 8, lower bound: -10.8483099, upper bound: 10.8481744
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.10
Output dim: 8, lower bound: -10.8413274, upper bound: 10.8258480
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.10
Output dim: 8, lower bound: -10.8412345, upper bound: 10.8256224
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.10
Output dim: 8, lower bound: -10.8483825, upper bound: 10.8487559
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.10
Output dim: 8, lower bound: -10.8483771, upper bound: 10.8483771

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2837172, 0.4566129, -0.5682322, 0.7299621, -1.0136793, 1.0248451
1: -0.2366557, 0.3478408, -0.5222386, 0.6414773, -0.8781331, 0.8700794
2: -0.4080100, 0.4674220, -0.5474883, 0.8410208, -1.2490308, 1.0149103
3: -0.1631940, 0.4603103, -0.4089364, 0.8762745, -1.0394685, 0.8692467
4: -0.3625288, 0.3168460, -0.5996383, 0.6361418, -0.9986706, 0.9164843
5: -0.2873004, 0.3946393, -0.5687888, 0.6923754, -0.9796758, 0.9634281
6: -0.2573539, 0.3556260, -0.5787522, 0.6261016, -0.8834555, 0.9343782
7: -0.3418233, 0.2827336, -0.5882402, 0.5957198, -0.9375432, 0.8709738
8: 0.0547081, 2.3356428, -0.5613427, 2.4380288, -2.3833208, 2.8969855
9: -0.5331606, 0.4568836, -0.6749626, 0.7706154, -1.3037760, 1.1318462

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=61, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5089659, upper bound: 10.4871174
time: 2.50 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6611291, upper bound: 10.7331084
time: 2.67 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.6162384, 0.8692144, -0.5350649, 0.7045001, -1.3207386, 1.4042792
1: -0.5474260, 0.6027564, -0.4962384, 0.6128722, -1.1602982, 1.0989947
2: -0.5596392, 0.8958975, -0.5308120, 0.8063342, -1.3659735, 1.4267095
3: -0.3756389, 0.8745847, -0.3724231, 0.8504226, -1.2260615, 1.2470078
4: -0.6084315, 0.7183238, -0.5728198, 0.6036881, -1.2121196, 1.2911437
5: -0.5698940, 0.7748866, -0.5388077, 0.6663621, -1.2362561, 1.3136944
6: -0.5673255, 0.7321869, -0.5481439, 0.6015072, -1.1688327, 1.2803307
7: -0.6234372, 0.6861230, -0.5575675, 0.5687987, -1.1922359, 1.2436905
8: -0.6795669, 2.4134986, -0.5015911, 2.4205139, -3.1000807, 2.9150898
9: -0.7263808, 0.8062595, -0.6586135, 0.7418987, -1.4682795, 1.4648731

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=113, inp2_unstable=58, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4933578, upper bound: 10.4820897
time: 2.67 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6319641, upper bound: 10.7267906
time: 2.70 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.3007282, 0.4875547, -0.8943683, 0.9703861, -1.2711143, 1.3819230
1: -0.2556671, 0.3679626, -0.7699892, 0.9033329, -1.1590000, 1.1379519
2: -0.4200501, 0.4993166, -0.7933325, 1.1236167, -1.5436668, 1.2926490
3: -0.1788405, 0.4890025, -0.7946840, 1.0818241, -1.2606646, 1.2836864
4: -0.3811160, 0.3381993, -0.9277589, 0.9185477, -1.2996638, 1.2659582
5: -0.3052583, 0.4215920, -0.8368798, 0.9539266, -1.2591850, 1.2584718
6: -0.2779050, 0.3772423, -0.8732637, 0.8968022, -1.1747073, 1.2505060
7: -0.3587787, 0.3074970, -0.8822821, 0.9047309, -1.2635095, 1.1897792
8: 0.0037651, 2.3597918, -1.1654706, 2.5097160, -2.5059509, 3.5252624
9: -0.5450766, 0.4847860, -0.8736245, 1.0480125, -1.5930891, 1.3584105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=69, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=108, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5089659, upper bound: 10.4871174
time: 2.38 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6611291, upper bound: 10.7331084
time: 2.84 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.6558267, 0.9088413, -0.8241374, 0.9272930, -1.5831196, 1.7329787
1: -0.5724534, 0.6294317, -0.7195543, 0.8533888, -1.4258422, 1.3489859
2: -0.5879613, 0.9440218, -0.7301275, 1.0776024, -1.6655637, 1.6741493
3: -0.4285598, 0.9083832, -0.7112947, 1.0444882, -1.4730480, 1.6196778
4: -0.6409537, 0.7519498, -0.8546547, 0.8625112, -1.5034649, 1.6066046
5: -0.6127214, 0.8038550, -0.7763430, 0.9067537, -1.5194751, 1.5801980
6: -0.6300936, 0.7621012, -0.8085641, 0.8411447, -1.4712384, 1.5706654
7: -0.6458867, 0.7335248, -0.8201933, 0.8460959, -1.4919825, 1.5537181
8: -0.7486411, 2.4346743, -1.0442222, 2.4853017, -3.2339430, 3.4788966
9: -0.7502123, 0.8601401, -0.8301407, 0.9922072, -1.7424195, 1.6902808

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=68, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=124, inp2_unstable=100, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4933578, upper bound: 10.4820897
time: 2.10 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6319641, upper bound: 10.7267906
time: 2.11 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.6174310, 0.7739822, -0.8518682, 0.9393665, -1.5567975, 1.6258504
1: -0.5507535, 0.6839973, -0.7527273, 0.8898670, -1.4406205, 1.4367247
2: -0.5866647, 0.9705589, -0.7608837, 1.1361163, -1.7227809, 1.7314427
3: -0.4464211, 0.8758153, -0.7357815, 1.1017866, -1.5482078, 1.6115968
4: -0.6430910, 0.6640586, -0.8762621, 0.8881248, -1.5312157, 1.5403206
5: -0.5958856, 0.7332711, -0.8034533, 0.9337807, -1.5296664, 1.5367244
6: -0.6314876, 0.6748120, -0.8480909, 0.8700425, -1.5015302, 1.5229028
7: -0.6228527, 0.6306599, -0.8407389, 0.8617207, -1.4845734, 1.4713988
8: -0.6953082, 2.4709792, -1.1143382, 2.5043280, -3.1996362, 3.5853174
9: -0.7101520, 0.8180386, -0.8515879, 1.0215318, -1.7316839, 1.6696265

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=66, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=99, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6739700, upper bound: 10.5800244
time: 2.45 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8423197, upper bound: 10.8412798
time: 1.74 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1.9407549, 1.6419548, -0.7950152, 0.9022931, -2.8430481, 2.4369700
1: -1.5002409, 1.6215090, -0.7107618, 0.8463430, -2.3465838, 2.3322706
2: -1.8262285, 1.8804337, -0.7112815, 1.0956557, -2.9218841, 2.5917153
3: -2.0337439, 1.5741997, -0.6674180, 1.0687395, -3.1024833, 2.2416177
4: -2.0956712, 1.6871412, -0.8166280, 0.8421798, -2.9378510, 2.5037692
5: -1.7136470, 1.7011950, -0.7561272, 0.8921697, -2.6058168, 2.4573221
6: -1.8611972, 1.7258128, -0.7980303, 0.8214135, -2.6826108, 2.5238431
7: -1.7944573, 1.8362811, -0.7937586, 0.8138941, -2.6083515, 2.6300397
8: -2.9806154, 2.5952339, -1.0161079, 2.4825883, -5.4632034, 3.6113420
9: -1.6060051, 1.8368833, -0.8179531, 0.9739928, -2.5799980, 2.6548364

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=175, inp2_unstable=93, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6596500, upper bound: 10.5761317
time: 2.52 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8307307, upper bound: 10.8376879
time: 2.96 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.6839392, 0.8341589, -1.3426545, 1.2427678, -1.9267070, 2.1768134
1: -0.6071441, 0.7422991, -1.1036911, 1.2281764, -1.8353205, 1.8459902
2: -0.6304923, 1.0427392, -1.2406734, 1.4586885, -2.0891809, 2.2834125
3: -0.5263523, 0.9272783, -1.3193403, 1.3437486, -1.8701010, 2.2466187
4: -0.7090896, 0.7296945, -1.4503925, 1.2569575, -1.9660470, 2.1800871
5: -0.6555626, 0.7928280, -1.2396553, 1.2816305, -1.9371932, 2.0324831
6: -0.6971338, 0.7312867, -1.2902322, 1.2734959, -1.9706297, 2.0215189
7: -0.6860474, 0.7020286, -1.2967932, 1.2863851, -1.9724324, 1.9988219
8: -0.8324839, 2.5002272, -1.9564599, 2.5866575, -3.4191415, 4.4566870
9: -0.7513066, 0.8814057, -1.1790289, 1.4084811, -2.1597877, 2.0604346

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=69, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=146, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6739700, upper bound: 10.5800244
time: 2.55 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8423197, upper bound: 10.8412640
time: 2.03 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -2.0985000, 1.7295101, -1.2566954, 1.1891844, -3.2876844, 2.9862056
1: -1.6092354, 1.7341914, -1.0395417, 1.1696683, -2.7789037, 2.7737331
2: -1.9973451, 1.9941630, -1.1530864, 1.4047447, -3.4020898, 3.1472495
3: -2.1945701, 1.6789614, -1.2165463, 1.2988788, -3.4934487, 2.8955078
4: -2.3196001, 1.8215735, -1.3474007, 1.1915921, -3.5111923, 3.1689742
5: -1.9149272, 1.7856332, -1.1608363, 1.2209663, -3.1358936, 2.9464695
6: -1.9827391, 1.8685017, -1.2130525, 1.2022126, -3.1849518, 3.0815542
7: -1.9389383, 1.9622091, -1.2151542, 1.2125492, -3.1514874, 3.1773634
8: -3.2658720, 2.6309085, -1.8098272, 2.5595503, -5.8254223, 4.4407358
9: -1.6995120, 2.0145674, -1.1191804, 1.3412135, -3.0407255, 3.1337478

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=68, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=180, inp2_unstable=141, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6596500, upper bound: 10.5761317
time: 2.87 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8307307, upper bound: 10.8376879
time: 1.97 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2762590, 0.4432430, -1.0732609, 1.0903459, -1.3666048, 1.5165038
1: -0.2268272, 0.3355546, -0.8984916, 1.0381279, -1.2649550, 1.2340461
2: -0.4070354, 0.4657289, -0.9798259, 1.2976695, -1.7047050, 1.4455547
3: -0.1565800, 0.4401356, -0.9995790, 1.1761720, -1.3327520, 1.4397146
4: -0.3521411, 0.3059670, -1.1330552, 1.0492313, -1.4013724, 1.4390223
5: -0.2765473, 0.3815849, -0.9924094, 1.0901511, -1.3666984, 1.3739944
6: -0.2479128, 0.3436202, -1.0507615, 1.0516071, -1.2995199, 1.3943816
7: -0.3318422, 0.2711821, -1.0391017, 1.0596232, -1.3914654, 1.3102838
8: 0.0625017, 2.3389528, -1.4987190, 2.5498347, -2.4873331, 3.8376718
9: -0.5320836, 0.4451236, -0.9951489, 1.1974541, -1.7295377, 1.4402726

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=123, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5096823, upper bound: 10.4964377
time: 2.77 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6651469, upper bound: 10.7546932
time: 3.47 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.5957246, 0.8224653, -0.9910657, 1.0389423, -1.6346669, 1.8135310
1: -0.5211629, 0.5753700, -0.8399827, 0.9809563, -1.5021192, 1.4153526
2: -0.5500283, 0.8834287, -0.8962902, 1.2442799, -1.7943082, 1.7797189
3: -0.3653175, 0.8159733, -0.9035264, 1.1316526, -1.4969702, 1.7194996
4: -0.5726804, 0.5873835, -1.0365956, 0.9859313, -1.5586116, 1.6239791
5: -0.5493295, 0.7486562, -0.9157909, 1.0326858, -1.5820153, 1.6644471
6: -0.5095139, 0.7061470, -0.9747267, 0.9849675, -1.4944813, 1.6808736
7: -0.5824388, 0.6305768, -0.9634036, 0.9893062, -1.5717450, 1.5939804
8: -0.5716054, 2.4159245, -1.3575977, 2.5237937, -3.0953991, 3.7735224
9: -0.6750190, 0.7902375, -0.9400241, 1.1309505, -1.8059695, 1.7302617

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=103, inp2_unstable=114, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4920426, upper bound: 10.4909146
time: 2.50 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6357786, upper bound: 10.7484359
time: 2.70 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2924749, 0.4738486, -1.5599929, 1.3977785, -1.6902535, 2.0338414
1: -0.2449671, 0.3554446, -1.2621958, 1.3678271, -1.6127942, 1.6176405
2: -0.4185296, 0.4972199, -1.4828370, 1.6033913, -2.0219209, 1.9800569
3: -0.1713458, 0.4685903, -1.5879194, 1.4206009, -1.5919467, 2.0565095
4: -0.3704672, 0.3265674, -1.7174926, 1.4139450, -1.7844121, 2.0440600
5: -0.2938722, 0.4072210, -1.4350073, 1.4320472, -1.7259195, 1.8422284
6: -0.2677978, 0.3649530, -1.4785318, 1.4604006, -1.7281983, 1.8434848
7: -0.3485916, 0.2943018, -1.5087252, 1.4813133, -1.8299049, 1.8030270
8: 0.0128798, 2.3629889, -2.3273563, 2.6388512, -2.6259713, 4.6903453
9: -0.5432180, 0.4726730, -1.3396621, 1.5827632, -2.1259813, 1.8123350

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=9, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=162, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5096823, upper bound: 10.4964377
time: 2.76 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6651469, upper bound: 10.7546932
time: 1.85 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.6243826, 0.8821505, -1.4624097, 1.3307372, -1.9551198, 2.3445601
1: -0.5537275, 0.6313586, -1.1852744, 1.2993062, -1.8530337, 1.8166330
2: -0.5728458, 0.9206786, -1.3670100, 1.5426455, -2.1154914, 2.2876885
3: -0.3928201, 0.8792998, -1.4627523, 1.3668486, -1.7596687, 2.3420522
4: -0.6208358, 0.7263362, -1.6000729, 1.3368379, -1.9576738, 2.3264091
5: -0.5858924, 0.7834985, -1.3429221, 1.3630651, -1.9489576, 2.1264205
6: -0.5808225, 0.7396041, -1.3915867, 1.3749884, -1.9558109, 2.1311908
7: -0.6378068, 0.6953799, -1.4091756, 1.3943231, -2.0321298, 2.1045556
8: -0.7100611, 2.4370079, -2.1588893, 2.6090689, -3.3191299, 4.5958972
9: -0.7347291, 0.8312135, -1.2687638, 1.4991004, -2.2338295, 2.0999773

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=115, inp2_unstable=154, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4920426, upper bound: 10.4909146
time: 2.44 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6357786, upper bound: 10.7484359
time: 2.06 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.6506287, 0.8000517, -1.7281692, 1.5096272, -2.1602559, 2.5282209
1: -0.5765964, 0.7115775, -1.3994378, 1.5011170, -2.0777135, 2.1110153
2: -0.6075222, 1.0246105, -1.7035484, 1.7581879, -2.3657103, 2.7281590
3: -0.4787563, 0.8970340, -1.8298241, 1.5783305, -2.0570869, 2.7268581
4: -0.6734983, 0.6911918, -1.9016328, 1.5522850, -2.2257833, 2.5928245
5: -0.6221793, 0.7599991, -1.5864748, 1.5654094, -2.1875887, 2.3464739
6: -0.6637344, 0.7032583, -1.6457851, 1.6130786, -2.2768130, 2.3490434
7: -0.6485103, 0.6610963, -1.6710149, 1.6383038, -2.2868142, 2.3321114
8: -0.7672678, 2.4850290, -2.6084905, 2.6650815, -3.4323492, 5.0935192
9: -0.7294382, 0.8489685, -1.4673884, 1.7387915, -2.4682298, 2.3163569

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=10, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=164, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6889039, upper bound: 10.5972798
time: 2.51 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8452626, upper bound: 10.8452494
time: 3.03 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1.9876039, 1.6729136, -1.6341085, 1.4421138, -3.4297175, 3.3070221
1: -1.5278269, 1.6611063, -1.3262494, 1.4334517, -2.9612784, 2.9873557
2: -1.8768392, 1.9410697, -1.5878367, 1.6995506, -3.5763898, 3.5289063
3: -2.0894592, 1.6008039, -1.6999316, 1.5246990, -3.6141582, 3.3007355
4: -2.1541729, 1.7451100, -1.7893083, 1.4763494, -3.6305223, 3.5344183
5: -1.7674332, 1.7315589, -1.4977585, 1.4976623, -3.2650955, 3.2293174
6: -1.9094657, 1.7756639, -1.5603436, 1.5293136, -3.4387794, 3.3360076
7: -1.8331852, 1.8732913, -1.5761654, 1.5500617, -3.3832469, 3.4494567
8: -3.0504601, 2.6195223, -2.4469905, 2.6337805, -5.6842403, 5.0665131
9: -1.6387987, 1.8977200, -1.3985780, 1.6558903, -3.2946892, 3.2962980

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=176, inp2_unstable=161, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6737220, upper bound: 10.5933411
time: 2.32 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8447067, upper bound: 10.8441537
time: 1.67 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.7207297, 0.8623386, -2.2387977, 1.8818722, -2.6026020, 3.1011362
1: -0.6343191, 0.7710211, -1.8112235, 1.8703984, -2.5047174, 2.5822446
2: -0.6595692, 1.0963399, -2.3434572, 2.0860367, -2.7456059, 3.4397972
3: -0.5610950, 0.9518531, -2.5375078, 1.8643327, -2.4254277, 3.4893608
4: -0.7443541, 0.7579960, -2.5144746, 1.9666109, -2.7109652, 3.2724705
5: -0.6846889, 0.8243388, -2.0767894, 1.9417765, -2.6264653, 2.9011283
6: -0.7337345, 0.7608014, -2.1095963, 2.0777431, -2.8114777, 2.8703976
7: -0.7151073, 0.7368076, -2.2053254, 2.1372700, -2.8523774, 2.9421329
8: -0.9095598, 2.5142663, -3.4757612, 2.7910361, -3.7005959, 5.9900274
9: -0.7757834, 0.9131776, -1.8944793, 2.1884005, -2.9641838, 2.8076568

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=74, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=12, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=181, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6889039, upper bound: 10.5972798
time: 2.41 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8452626, upper bound: 10.8452478
time: 4.88 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -2.2035151, 1.7839757, -2.1355982, 1.8007196, -4.0042348, 3.9195738
1: -1.6817923, 1.7851045, -1.7240957, 1.7946668, -3.4764590, 3.5092001
2: -2.0758550, 2.0604239, -2.2104030, 2.0175402, -4.0933952, 4.2708268
3: -2.2898259, 1.7147843, -2.3951910, 1.8013387, -4.0911646, 4.1099753
4: -2.4309649, 1.8614993, -2.3932261, 1.8781884, -4.3091536, 4.2547255
5: -2.0106227, 1.8501642, -1.9675773, 1.8642371, -3.8748598, 3.8177414
6: -2.0437231, 1.9730543, -2.0129266, 1.9828782, -4.0266013, 3.9859810
7: -2.0782585, 2.0074468, -2.0954387, 2.0339868, -4.1122456, 4.1028852
8: -3.4136415, 2.6530185, -3.3011301, 2.7495515, -6.1631927, 5.9541483
9: -1.8021276, 2.0520260, -1.8022116, 2.0957422, -3.8978698, 3.8542376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=74, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=183, inp2_unstable=177, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6737220, upper bound: 10.5933411
time: 2.57 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8447067, upper bound: 10.8441537
time: 2.01 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.4694453, 0.6529000, -0.4276201, 0.6031079, -1.0725532, 1.0805202
1: -0.4308256, 0.5381819, -0.3947893, 0.4838384, -0.9146640, 0.9329712
2: -0.5157504, 0.7337205, -0.4735304, 0.6604729, -1.1762233, 1.2072510
3: -0.3096458, 0.7643185, -0.2726524, 0.7075283, -1.0171740, 1.0369709
4: -0.5159682, 0.5407785, -0.4776782, 0.4951326, -1.0111008, 1.0184567
5: -0.4675617, 0.6146805, -0.4243473, 0.5720962, -1.0396580, 1.0390278
6: -0.4750152, 0.5464661, -0.4103143, 0.5095930, -0.9846082, 0.9567804
7: -0.5024090, 0.5058461, -0.4635707, 0.4608580, -0.9632670, 0.9694169
8: -0.3893265, 2.4832556, -0.2701023, 2.3885517, -2.7778783, 2.7533579
9: -0.6466318, 0.6783789, -0.5977399, 0.6235150, -1.2701468, 1.2761188

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=69, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=44, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 69

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5934016, upper bound: 10.4522011
time: 2.45 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7749076, upper bound: 10.6884798
time: 3.44 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1.2045536, 1.3313465, -0.4115116, 0.5890689, -1.7936225, 1.7428582
1: -1.0645616, 1.1782119, -0.3808619, 0.4710928, -1.5356544, 1.5590738
2: -0.9912682, 1.5091075, -0.4658744, 0.6435844, -1.6348525, 1.9749818
3: -1.1620898, 1.3224357, -0.2617567, 0.6890308, -1.8511206, 1.5841925
4: -1.2598410, 1.2332203, -0.4674135, 0.4767891, -1.7366300, 1.7006338
5: -1.1154909, 1.2897841, -0.4092528, 0.5583978, -1.6738887, 1.6990368
6: -1.1952012, 1.1631618, -0.3933637, 0.4958337, -1.6910348, 1.5565255
7: -1.1993785, 1.3130215, -0.4517347, 0.4454527, -1.6448312, 1.7647562
8: -1.8790401, 2.5717559, -0.2399555, 2.3766830, -4.2557230, 2.8117113
9: -1.1304786, 1.3456553, -0.5893623, 0.6086237, -1.7391024, 1.9350176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=73, inp2_unstable=65, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=161, inp2_unstable=43, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 69

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5879611, upper bound: 10.4503066
time: 2.48 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7623304, upper bound: 10.6866906
time: 6.79 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.6700748, 0.8202695, -1.2395741, 1.1744721, -1.8445469, 2.0598435
1: -0.6062815, 0.7360718, -1.0256579, 1.1587013, -1.7649828, 1.7617297
2: -0.6243469, 0.9936755, -1.1414099, 1.4298596, -2.0542064, 2.1350853
3: -0.5181841, 0.9499421, -1.1834252, 1.2958721, -1.8140562, 2.1333673
4: -0.6953346, 0.7274643, -1.3209059, 1.1747738, -1.8701084, 2.0483704
5: -0.6516439, 0.7822532, -1.1409409, 1.2123230, -1.8639668, 1.9231942
6: -0.6879987, 0.7176587, -1.2029917, 1.1857358, -1.8737345, 1.9206505
7: -0.6780669, 0.6925409, -1.1905929, 1.1935657, -1.8716326, 1.8831338
8: -0.7985743, 2.5394235, -1.7795602, 2.5668201, -3.3653946, 4.3189836
9: -0.7494843, 0.8712636, -1.1084423, 1.3259530, -2.0754373, 1.9797058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=69, inp2_unstable=69, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=137, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7055027, upper bound: 10.6031621
time: 2.84 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8456860, upper bound: 10.8457536
time: 14.40 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1.9746218, 1.6639342, -1.1637559, 1.1276543, -3.1022761, 2.8276901
1: -1.5325344, 1.6627880, -0.9698576, 1.1087554, -2.6412897, 2.6326456
2: -1.8741894, 1.8866677, -1.0651000, 1.3818308, -3.2560201, 2.9517677
3: -2.0608418, 1.6458868, -1.0950379, 1.2564892, -3.3173308, 2.7409248
4: -2.1849644, 1.7361028, -1.2312050, 1.1167901, -3.3017545, 2.9673078
5: -1.8128721, 1.7062726, -1.0721095, 1.1587096, -2.9715817, 2.7783821
6: -1.8726227, 1.7809684, -1.1345339, 1.1244578, -2.9970806, 2.9155023
7: -1.8385283, 1.8592323, -1.1180942, 1.1295327, -2.9680610, 2.9773264
8: -3.0769825, 2.6543026, -1.6532750, 2.5429430, -5.6199255, 4.3075776
9: -1.6216440, 1.9278944, -1.0566082, 1.2662485, -2.8878925, 2.9845026

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=73, inp2_unstable=69, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=177, inp2_unstable=132, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6983119, upper bound: 10.6013238
time: 2.72 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8455124, upper bound: 10.8453548
time: 2.33 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.7465351, 0.8966106, -0.4150795, 0.5930120, -1.3395470, 1.3116901
1: -0.6574052, 0.7951088, -0.3835334, 0.4761210, -1.1335262, 1.1786423
2: -0.6786262, 1.0814242, -0.4714028, 0.6620464, -1.3406726, 1.5528271
3: -0.6073010, 0.9720529, -0.2662764, 0.6888413, -1.2961423, 1.2383293
4: -0.7775435, 0.7924795, -0.4705992, 0.4790593, -1.2566028, 1.2630787
5: -0.7097749, 0.8526873, -0.4118507, 0.5614027, -1.2711775, 1.2645380
6: -0.7563305, 0.7856948, -0.3977675, 0.4973940, -1.2537246, 1.1834624
7: -0.7493750, 0.7751951, -0.4540552, 0.4491329, -1.1985079, 1.2292503
8: -0.9541827, 2.5753782, -0.2586430, 2.3918302, -3.3460131, 2.8340211
9: -0.7972384, 0.9406792, -0.5934395, 0.6156373, -1.4128757, 1.5341187

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=5, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=83, inp2_unstable=43, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6100872, upper bound: 10.4571540
time: 2.24 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7940340, upper bound: 10.6929824
time: 2.52 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -2.1998842, 1.8017216, -0.3993203, 0.5794075, -2.7792916, 2.2010419
1: -1.7313070, 1.7806100, -0.3697996, 0.4642047, -2.1955118, 2.1504095
2: -2.0754449, 2.0146561, -0.4640645, 0.6455398, -2.7209847, 2.4787207
3: -2.3053186, 1.7019336, -0.2555650, 0.6708801, -2.9761987, 1.9574987
4: -2.4604816, 1.8718842, -0.4605155, 0.4612517, -2.9217334, 2.3323998
5: -1.9893636, 1.8764817, -0.3974241, 0.5479141, -2.5372777, 2.2739058
6: -2.0316086, 1.9815786, -0.3821775, 0.4838560, -2.5154645, 2.3637562
7: -2.0977538, 2.0293136, -0.4427114, 0.4339628, -2.5317166, 2.4720249
8: -3.4372487, 2.7035983, -0.2295602, 2.3801157, -5.8173647, 2.9331584
9: -1.7858515, 2.0671854, -0.5854630, 0.6010057, -2.3868570, 2.6526484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=65, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=5, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=181, inp2_unstable=42, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6053704, upper bound: 10.4550879
time: 2.62 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7815601, upper bound: 10.6909031
time: 3.30 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1.2987497, 1.2309320, -1.3535562, 1.2442929, -2.5430427, 2.5844882
1: -1.0604417, 1.1940243, -1.1095760, 1.2348181, -2.2952600, 2.3036003
2: -1.2121027, 1.4971941, -1.2602552, 1.5218059, -2.7339087, 2.7574492
3: -1.2406735, 1.3034413, -1.3114239, 1.3666087, -2.6072822, 2.6148653
4: -1.3866558, 1.2131051, -1.4519858, 1.2584186, -2.6450744, 2.6650910
5: -1.1885052, 1.2553871, -1.2386136, 1.2948731, -2.4833784, 2.4940007
6: -1.2671200, 1.2373180, -1.3071373, 1.2808717, -2.5479918, 2.5444553
7: -1.2378148, 1.2425256, -1.2950454, 1.2883277, -2.5261426, 2.5375710
8: -1.9056187, 2.6665485, -1.9716592, 2.5893874, -4.4950061, 4.6382074
9: -1.1634939, 1.3766992, -1.1885052, 1.4135730, -2.5770669, 2.5652044

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=9, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=139, inp2_unstable=148, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7318947, upper bound: 10.6177226
time: 2.24 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8462860, upper bound: 10.8465753
time: 1.86 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -2.8719826, 2.3476377, -1.2754455, 1.1945536, -4.0665359, 3.6230831
1: -2.3167362, 2.3183875, -1.0503674, 1.1817983, -3.4985347, 3.3687549
2: -3.1324685, 2.4790809, -1.1766207, 1.4730213, -4.6054897, 3.6557016
3: -3.3822448, 2.1518722, -1.2151463, 1.3237869, -4.7060318, 3.3670185
4: -3.2858994, 2.4399142, -1.3586398, 1.1982636, -4.4841633, 3.7985539
5: -2.6562257, 2.3793125, -1.1675112, 1.2393675, -3.8955932, 3.5468237
6: -2.6575119, 2.6440129, -1.2372985, 1.2154679, -3.8729799, 3.8813114
7: -2.8255553, 2.7240617, -1.2189356, 1.2212034, -4.0467587, 3.9429975
8: -4.5677142, 2.8929093, -1.8436160, 2.5645123, -7.1322265, 4.7365255
9: -2.3400669, 2.7334747, -1.1347842, 1.3501769, -3.6902437, 3.8682590

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=9, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=205, inp2_unstable=139, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7247528, upper bound: 10.6159623
time: 2.49 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8461716, upper bound: 10.8461716
time: 2.71 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.5172318, 0.7050183, -0.5979371, 0.7663171, -1.2835490, 1.3029554
1: -0.4750557, 0.5896472, -0.5347099, 0.6520414, -1.1270971, 1.1243571
2: -0.5386876, 0.7937901, -0.5492778, 0.8661235, -1.4048111, 1.3430679
3: -0.3540901, 0.8071781, -0.4422859, 0.8510857, -1.2051758, 1.2494640
4: -0.5586371, 0.5861020, -0.6245806, 0.6615785, -1.2202156, 1.2106826
5: -0.5179222, 0.6524178, -0.5877079, 0.7124512, -1.2303734, 1.2401257
6: -0.5305310, 0.5920297, -0.5852681, 0.6509149, -1.1814460, 1.1772978
7: -0.5430107, 0.5526607, -0.6179329, 0.6292885, -1.1722991, 1.1705935
8: -0.4868004, 2.5082645, -0.6065448, 2.4486089, -2.9354093, 3.1148093
9: -0.6715684, 0.7302775, -0.6861031, 0.7911121, -1.4626805, 1.4163806

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=69, inp2_unstable=69, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=58, inp2_unstable=64, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5934016, upper bound: 10.4522011
time: 2.99 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7749076, upper bound: 10.6884798
time: 2.59 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1.4801620, 1.4624062, -0.5668811, 0.7400931, -2.2202551, 2.0292873
1: -1.2316577, 1.3836164, -0.5083077, 0.6245642, -1.8562219, 1.8919241
2: -1.3605956, 1.6033825, -0.5321833, 0.8315542, -2.1921496, 2.1355658
3: -1.5181490, 1.4243308, -0.4065125, 0.8246904, -2.3428395, 1.8308433
4: -1.5564102, 1.4299099, -0.5968595, 0.6296929, -2.1861031, 2.0267694
5: -1.3191689, 1.4497371, -0.5597867, 0.6861740, -2.0053430, 2.0095239
6: -1.3498297, 1.4960114, -0.5558566, 0.6279033, -1.9777330, 2.0518680
7: -1.4252164, 1.4852396, -0.5883636, 0.5984873, -2.0237036, 2.0736032
8: -2.2880831, 2.6048017, -0.5445579, 2.4308381, -4.7189212, 3.1493597
9: -1.2572235, 1.6132156, -0.6687155, 0.7615446, -2.0187681, 2.2819312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=73, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=167, inp2_unstable=61, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5879611, upper bound: 10.4503066
time: 2.54 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7623304, upper bound: 10.6866906
time: 5.84 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.7459809, 0.8847358, -1.7422148, 1.5138743, -2.2598553, 2.6269507
1: -0.6671487, 0.8012104, -1.4101571, 1.5079199, -2.1750686, 2.2113676
2: -0.6797488, 1.0675014, -1.7127388, 1.7480435, -2.4277923, 2.7802401
3: -0.6073182, 1.0073941, -1.8496084, 1.5606236, -2.1679418, 2.8570025
4: -0.7709308, 0.7988558, -1.9272691, 1.5617913, -2.3327222, 2.7261248
5: -0.7155277, 0.8517933, -1.6030878, 1.5718079, -2.2873354, 2.4548812
6: -0.7604897, 0.7818516, -1.6485229, 1.6234634, -2.3839531, 2.4303744
7: -0.7504506, 0.7705733, -1.6900994, 1.6539010, -2.4043517, 2.4606726
8: -0.9489703, 2.5684536, -2.6338062, 2.6618683, -3.6108387, 5.2022600
9: -0.7987324, 0.9399019, -1.4731649, 1.7489244, -2.5476568, 2.4130669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=69, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=83, inp2_unstable=165, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7055027, upper bound: 10.6031621
time: 2.30 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8456860, upper bound: 10.8457536
time: 3.18 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -2.1949854, 1.7961634, -1.6539887, 1.4509407, -3.6459260, 3.4501522
1: -1.7254845, 1.7858444, -1.3412626, 1.4456468, -3.1711311, 3.1271071
2: -2.0550306, 2.0069385, -1.6044955, 1.6921459, -3.7471766, 3.6114340
3: -2.3170943, 1.7339885, -1.7273405, 1.5104665, -3.8275609, 3.4613290
4: -2.4669859, 1.8834062, -1.8218744, 1.4907380, -3.9577241, 3.7052805
5: -2.0013623, 1.8806791, -1.5200970, 1.5076841, -3.5090466, 3.4007761
6: -2.0361242, 1.9711777, -1.5676606, 1.5452360, -3.5813603, 3.5388384
7: -2.1120110, 1.9872661, -1.6012831, 1.5710124, -3.6830235, 3.5885491
8: -3.4218044, 2.6954520, -2.4827385, 2.6315432, -6.0533476, 5.1781902
9: -1.7880751, 2.0727630, -1.4086487, 1.6709894, -3.4590645, 3.4814117

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=73, inp2_unstable=69, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=182, inp2_unstable=164, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6983119, upper bound: 10.6013238
time: 2.96 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8455124, upper bound: 10.8453547
time: 1.62 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.8548315, 0.9729112, -0.5654026, 0.7410037, -1.5958352, 1.5383139
1: -0.7371748, 0.8777912, -0.5064967, 0.6246005, -1.3617754, 1.3842878
2: -0.7716476, 1.1615790, -0.5356548, 0.8421050, -1.6137526, 1.6972339
3: -0.7352794, 1.0363644, -0.4047453, 0.8206455, -1.5559249, 1.4411098
4: -0.8884984, 0.8800038, -0.5945676, 0.6272699, -1.5157683, 1.4745713
5: -0.7959436, 0.9356470, -0.5582622, 0.6852054, -1.4811490, 1.4939092
6: -0.8502627, 0.8783861, -0.5565071, 0.6262538, -1.4765165, 1.4348931
7: -0.8415049, 0.8708535, -0.5858593, 0.5956960, -1.4372008, 1.4567128
8: -1.1454779, 2.6065810, -0.5516490, 2.4494874, -3.5949655, 3.1582298
9: -0.8617782, 1.0319117, -0.6702948, 0.7642033, -1.6259815, 1.7022066

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=68, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=104, inp2_unstable=63, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6100872, upper bound: 10.4571540
time: 2.59 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7940340, upper bound: 10.6929824
time: 3.72 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -2.3595486, 1.9238670, -0.5353293, 0.7163302, -3.0758789, 2.4591963
1: -1.8683621, 1.9268239, -0.4822469, 0.5977703, -2.4661324, 2.4090707
2: -2.2591367, 2.1099219, -0.5208566, 0.8093985, -3.0685353, 2.6307786
3: -2.5422058, 1.7912340, -0.3713251, 0.7960217, -3.3382275, 2.1625590
4: -2.6833220, 1.9881929, -0.5694184, 0.5973306, -3.2806525, 2.5576115
5: -2.1519923, 1.9924625, -0.5313693, 0.6601417, -2.8121340, 2.5238318
6: -2.1684113, 2.1414213, -0.5281698, 0.6035422, -2.7719536, 2.6695910
7: -2.2507727, 2.1795776, -0.5575559, 0.5693669, -2.8201396, 2.7371335
8: -3.6928582, 2.7405651, -0.4956278, 2.4334347, -6.1262932, 3.2361927
9: -1.9282557, 2.1871796, -0.6555150, 0.7363492, -2.6646049, 2.8426945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=6, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=185, inp2_unstable=57, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6053704, upper bound: 10.4550879
time: 2.38 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7815601, upper bound: 10.6909031
time: 2.26 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1.4542458, 1.3336086, -1.8286511, 1.5763104, -3.0305562, 3.1622596
1: -1.1817398, 1.3012757, -1.4772263, 1.5679286, -2.7496684, 2.7785020
2: -1.3826720, 1.5937638, -1.8236774, 1.8206487, -3.2033205, 3.4174414
3: -1.4261439, 1.3886158, -1.9659264, 1.6213303, -3.0474741, 3.3545423
4: -1.5705551, 1.3333344, -2.0250397, 1.6300628, -3.2006178, 3.3583741
5: -1.3300396, 1.3638097, -1.6792562, 1.6365993, -2.9666390, 3.0430660
6: -1.4050778, 1.3695748, -1.7308002, 1.7010179, -3.1060958, 3.1003749
7: -1.3923072, 1.3741715, -1.7740963, 1.7326549, -3.1249621, 3.1482677
8: -2.1670704, 2.7040758, -2.7788346, 2.6871822, -4.8542528, 5.4829102
9: -1.2735286, 1.5020270, -1.5380309, 1.8252078, -3.0987363, 3.0400579

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=167, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7318947, upper bound: 10.6177226
time: 2.39 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8462860, upper bound: 10.8465753
time: 2.09 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -3.0257187, 2.4723048, -1.7394091, 1.5126401, -4.5383587, 4.2117138
1: -2.4504106, 2.4407337, -1.4075046, 1.5051103, -3.9555209, 3.8482382
2: -3.3347898, 2.5942631, -1.7143441, 1.7644563, -5.0992460, 4.3086071
3: -3.6078854, 2.2497952, -1.8419312, 1.5707083, -5.1785936, 4.0917263
4: -3.4680569, 2.5706856, -1.9181325, 1.5583391, -5.0263958, 4.4888182
5: -2.8108008, 2.5008090, -1.5952160, 1.5719333, -4.3827343, 4.0960250
6: -2.8072321, 2.8044040, -1.6491621, 1.6217294, -4.4289618, 4.4535661
7: -3.0377550, 2.8797894, -1.6841174, 1.6488222, -4.6865773, 4.5639067
8: -4.8393989, 2.9502950, -2.6262064, 2.6559191, -7.4953179, 5.5765014
9: -2.5222812, 2.8679452, -1.4724617, 1.7461843, -4.2684655, 4.3404069

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=208, inp2_unstable=165, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7247528, upper bound: 10.6159623
time: 1.97 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8461716, upper bound: 10.8461716
time: 3.92 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 7.49 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.5089659, upper bound: 10.4871174
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.6611291, upper bound: 10.7331084
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.4933578, upper bound: 10.4820897
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.6319641, upper bound: 10.7267906
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.5089659, upper bound: 10.4871174
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.6611291, upper bound: 10.7331084
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.4933578, upper bound: 10.4820897
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.6319641, upper bound: 10.7267906
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.6739700, upper bound: 10.5800244
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.8423197, upper bound: 10.8412798
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.6596500, upper bound: 10.5761317
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.8307307, upper bound: 10.8376879
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.6739700, upper bound: 10.5800244
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.8423197, upper bound: 10.8412640
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.6596500, upper bound: 10.5761317
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.8307307, upper bound: 10.8376879
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.5096823, upper bound: 10.4964377
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.6651469, upper bound: 10.7546932
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.4920426, upper bound: 10.4909146
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.6357786, upper bound: 10.7484359
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.5096823, upper bound: 10.4964377
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.6651469, upper bound: 10.7546932
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.4920426, upper bound: 10.4909146
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.6357786, upper bound: 10.7484359
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.6889039, upper bound: 10.5972798
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.8452626, upper bound: 10.8452494
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.6737220, upper bound: 10.5933411
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.8447067, upper bound: 10.8441537
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.6889039, upper bound: 10.5972798
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.8452626, upper bound: 10.8452478
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.6737220, upper bound: 10.5933411
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.8447067, upper bound: 10.8441537
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.5934016, upper bound: 10.4522011
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.7749076, upper bound: 10.6884798
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.5879611, upper bound: 10.4503066
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.7623304, upper bound: 10.6866906
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.7055027, upper bound: 10.6031621
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.8456860, upper bound: 10.8457536
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.6983119, upper bound: 10.6013238
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.8455124, upper bound: 10.8453548
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.6100872, upper bound: 10.4571540
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.7940340, upper bound: 10.6929824
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.6053704, upper bound: 10.4550879
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.7815601, upper bound: 10.6909031
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.7318947, upper bound: 10.6177226
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.8462860, upper bound: 10.8465753
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.7247528, upper bound: 10.6159623
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.8461716, upper bound: 10.8461716
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.5934016, upper bound: 10.4522011
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.7749076, upper bound: 10.6884798
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.5879611, upper bound: 10.4503066
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.7623304, upper bound: 10.6866906
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.7055027, upper bound: 10.6031621
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.8456860, upper bound: 10.8457536
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.6983119, upper bound: 10.6013238
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.8455124, upper bound: 10.8453547
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.6100872, upper bound: 10.4571540
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.7940340, upper bound: 10.6929824
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.6053704, upper bound: 10.4550879
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.7815601, upper bound: 10.6909031
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.7318947, upper bound: 10.6177226
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.8462860, upper bound: 10.8465753
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.7247528, upper bound: 10.6159623
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 8, lower bound: -10.8461716, upper bound: 10.8461716

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1834026, 0.1764473, -0.1791120, 0.0765092, -0.2599118, 0.3555593
1: -0.0800586, 0.1190268, -0.0697137, 0.0772574, -0.1573161, 0.1887406
2: -0.3262616, 0.1666474, -0.3195477, 0.0984057, -0.4246672, 0.4861951
3: -0.1103528, 0.1115013, -0.1087228, 0.0523004, -0.1626532, 0.2202242
4: -0.0976256, 0.1612447, -0.0776929, 0.0993005, -0.1969261, 0.2239298
5: -0.1709299, 0.1100812, -0.1500118, 0.0487945, -0.2197244, 0.2600929
6: -0.1728325, 0.1125856, -0.1678101, 0.0825879, -0.2554204, 0.2803957
7: -0.1775251, 0.1136709, -0.1270058, 0.0795917, -0.2571167, 0.2406767
8: 0.6282828, 2.2568166, 0.8292130, 2.2556286, -1.6273458, 1.4276036
9: -0.4277024, 0.2061759, -0.4246244, 0.0642048, -0.4919072, 0.6308002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=62, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.3099583, upper bound: 10.2593744
time: 3.71 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2640191, upper bound: 10.2418163
time: 2.73 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2412965, 0.3832748, -0.3036451, 0.4737175, -0.7150140, 0.6869199
1: -0.1823967, 0.2908354, -0.2734125, 0.3755666, -0.5579634, 0.5642479
2: -0.3864003, 0.3735359, -0.4191704, 0.4976563, -0.8840566, 0.7927063
3: -0.1333454, 0.3631682, -0.1829801, 0.5457308, -0.6790763, 0.5461484
4: -0.3098158, 0.2674221, -0.3832262, 0.3519857, -0.6618015, 0.6506482
5: -0.2422853, 0.3310988, -0.3081066, 0.4302544, -0.6725397, 0.6392055
6: -0.2120025, 0.2956091, -0.2908444, 0.3761039, -0.5881064, 0.5864536
7: -0.2932711, 0.2248116, -0.3596906, 0.3087701, -0.6020412, 0.5845022
8: 0.1954161, 2.3098345, -0.0131831, 2.3599081, -2.1644921, 2.3230176
9: -0.5150667, 0.3840154, -0.5432958, 0.4861261, -1.0011928, 0.9273112

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=66, inp2_unstable=65, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=33, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4518468, upper bound: 10.4500155
time: 2.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3906070, upper bound: 10.4311742
time: 2.31 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.3397722, 0.5972340, -0.1771812, 0.0749006, -0.4146729, 0.7744151
1: -0.2790638, 0.3868424, -0.0691073, 0.0750459, -0.3541096, 0.4445450
2: -0.4223440, 0.5238652, -0.3169151, 0.0952817, -0.5176257, 0.8407803
3: -0.2119896, 0.4432561, -0.1077644, 0.0511214, -0.2631110, 0.5510206
4: -0.4098796, 0.3799648, -0.0760680, 0.0972200, -0.5070996, 0.4560328
5: -0.3434416, 0.4803380, -0.1485906, 0.0464545, -0.3898961, 0.6136907
6: -0.3051775, 0.4398654, -0.1661121, 0.0810642, -0.3862417, 0.5789990
7: -0.3981190, 0.3790659, -0.1257960, 0.0782662, -0.4763852, 0.5048618
8: -0.0507057, 2.3272233, 0.8348623, 2.2462804, -2.2969861, 1.4923611
9: -0.5478314, 0.5362103, -0.4220487, 0.0605982, -0.6084297, 0.9582590

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=62, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2949447, upper bound: 10.2548473
time: 2.36 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2360759, upper bound: 10.2317539
time: 2.44 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.5264078, 0.7847248, -0.2946350, 0.4594832, -0.9858910, 1.0793599
1: -0.4650661, 0.5355406, -0.2630968, 0.3646730, -0.8297391, 0.7986374
2: -0.5103696, 0.7826304, -0.4130050, 0.4811913, -0.9915609, 1.1956353
3: -0.3217826, 0.7515637, -0.1749537, 0.5289830, -0.8507656, 0.9265174
4: -0.5464209, 0.6123769, -0.3732445, 0.3407606, -0.8871815, 0.9856213
5: -0.4986731, 0.6824483, -0.2984511, 0.4169059, -0.9155791, 0.9808994
6: -0.4842524, 0.6405575, -0.2799693, 0.3649483, -0.8492007, 0.9205268
7: -0.5525504, 0.5884560, -0.3509088, 0.2966950, -0.8492454, 0.9393648
8: -0.4955102, 2.3866658, 0.0119740, 2.3463500, -2.8418603, 2.3746917
9: -0.6577470, 0.7249359, -0.5375638, 0.4721926, -1.1299396, 1.2624997

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=65, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=33, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 92

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6314970, upper bound: 10.7267906
time: 1.66 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6314970, upper bound: 10.7267906
time: 2.15 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1899148, 0.2068106, -0.1885968, 0.1378345, -0.3277493, 0.3954074
1: -0.0892890, 0.1332570, -0.0766421, 0.1032509, -0.1925398, 0.2098991
2: -0.3332822, 0.1914296, -0.3345287, 0.1380781, -0.4713603, 0.5259584
3: -0.1136551, 0.1309398, -0.1139035, 0.0845147, -0.1981698, 0.2448434
4: -0.1089217, 0.1757824, -0.0923967, 0.1447089, -0.2536305, 0.2681790
5: -0.1783367, 0.1316034, -0.1662403, 0.0816970, -0.2600338, 0.2978436
6: -0.1767288, 0.1303246, -0.1764469, 0.0970212, -0.2737499, 0.3067715
7: -0.1936997, 0.1288736, -0.1578196, 0.1043441, -0.2980438, 0.2866932
8: 0.5739703, 2.2793279, 0.6962041, 2.2994108, -1.7254405, 1.5831238
9: -0.4365484, 0.2321771, -0.4416606, 0.1696829, -0.6062313, 0.6738377

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=63, inp2_unstable=64, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3166770, upper bound: 10.2594505
time: 3.04 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2656324, upper bound: 10.2418163
time: 2.61 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2554555, 0.4138108, -0.3921732, 0.5807011, -0.8361566, 0.8059841
1: -0.1997797, 0.3087630, -0.3619731, 0.4567900, -0.6565697, 0.6707362
2: -0.3954350, 0.4037646, -0.4658814, 0.6251750, -1.0206100, 0.8696460
3: -0.1432922, 0.3895264, -0.2491045, 0.6592833, -0.8025755, 0.6386308
4: -0.3268787, 0.2838901, -0.4563432, 0.4528587, -0.7797374, 0.7402333
5: -0.2570049, 0.3529486, -0.3925681, 0.5408685, -0.7978733, 0.7455167
6: -0.2262357, 0.3170287, -0.3798470, 0.4777821, -0.7040178, 0.6968757
7: -0.3100711, 0.2442461, -0.4366936, 0.4254894, -0.7355605, 0.6809398
8: 0.1448242, 2.3336868, -0.2151144, 2.4129944, -2.2681701, 2.5488012
9: -0.5253584, 0.4099215, -0.5897624, 0.5951633, -1.1205217, 0.9996840

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=66, inp2_unstable=68, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=43, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4573517, upper bound: 10.4506307
time: 3.28 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3907717, upper bound: 10.4317282
time: 44.32 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.3656377, 0.6285113, -0.1850634, 0.1285653, -0.4942030, 0.8135747
1: -0.3033233, 0.4071929, -0.0747519, 0.0986975, -0.4020209, 0.4819449
2: -0.4350048, 0.5593778, -0.3307958, 0.1320857, -0.5670905, 0.8901737
3: -0.2281181, 0.4774151, -0.1124245, 0.0777100, -0.3058281, 0.5898396
4: -0.4303830, 0.4117566, -0.0888105, 0.1403531, -0.5707361, 0.5005671
5: -0.3662443, 0.5072734, -0.1641309, 0.0764610, -0.4427052, 0.6661026
6: -0.3287994, 0.4700850, -0.1745906, 0.0928706, -0.4216701, 0.6271623
7: -0.4201081, 0.4106033, -0.1519522, 0.1020224, -0.5221305, 0.5625556
8: -0.1135096, 2.3471603, 0.7089939, 2.2861819, -2.3996916, 1.6381664
9: -0.5630724, 0.5654861, -0.4370255, 0.1610476, -0.7241200, 1.0025115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=63, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4933578, upper bound: 10.4820897
time: 2.73 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4933578, upper bound: 10.4820897
time: 2.63 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.5621834, 0.8213363, -0.3753376, 0.5625460, -1.1247294, 1.1966739
1: -0.4898159, 0.5601392, -0.3452935, 0.4419125, -0.9317284, 0.9054327
2: -0.5337468, 0.8265116, -0.4561038, 0.6026912, -1.1364379, 1.2826154
3: -0.3636509, 0.7850611, -0.2367326, 0.6359648, -0.9996156, 1.0217936
4: -0.5748242, 0.6456550, -0.4429095, 0.4310102, -1.0058343, 1.0885645
5: -0.5342237, 0.7113045, -0.3762266, 0.5223534, -1.0565771, 1.0875311
6: -0.5345932, 0.6701932, -0.3606584, 0.4593026, -0.9938958, 1.0308516
7: -0.5748042, 0.6304934, -0.4223571, 0.4048313, -0.9796355, 1.0528505
8: -0.5616231, 2.4076796, -0.1758693, 2.3969479, -2.9585710, 2.5835490
9: -0.6795173, 0.7702864, -0.5796244, 0.5757311, -1.2552484, 1.3499109

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=67, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=85, inp2_unstable=40, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 92

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6314970, upper bound: 10.7267906
time: 1.96 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6314970, upper bound: 10.7267906
time: 2.13 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2859486, 0.4454438, -0.1863355, 0.1125881, -0.3985367, 0.6317793
1: -0.2247686, 0.3357489, -0.0754686, 0.0994985, -0.3242671, 0.4112175
2: -0.4232617, 0.5304785, -0.3325357, 0.1427315, -0.5659931, 0.8630142
3: -0.1621612, 0.4193389, -0.1131490, 0.0851077, -0.2472689, 0.5324879
4: -0.3595446, 0.3050852, -0.0905447, 0.1402117, -0.4997563, 0.3956299
5: -0.2768674, 0.3849950, -0.1651702, 0.0745266, -0.3513940, 0.5501653
6: -0.2523857, 0.3449912, -0.1753135, 0.0925112, -0.3448969, 0.5203047
7: -0.3314928, 0.2754574, -0.1494354, 0.0986111, -0.4301039, 0.4248929
8: 0.0095262, 2.3651323, 0.6853876, 2.2859247, -2.2763984, 1.6797447
9: -0.5448874, 0.4546399, -0.4340597, 0.1598900, -0.7047774, 0.8886995

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=62, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 6.98 + 593.24 = 600.23 seconds
