## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00710256


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0038321, 0.0086154, 0.0038321, 0.0086154, -0.0047833, 0.0047833)
1: (-0.0001844, 0.0047693, -0.0001844, 0.0047693, -0.0049537, 0.0049537)
2: (-0.0256816, -0.0049325, -0.0256816, -0.0049325, -0.0207491, 0.0207491)
3: (-0.0020720, 0.0079101, -0.0020720, 0.0079101, -0.0099821, 0.0099821)
4: (0.0112831, 0.0184802, 0.0112831, 0.0184802, -0.0071971, 0.0071971)
5: (-0.0032784, 0.0098897, -0.0032784, 0.0098897, -0.0131682, 0.0131682)
6: (0.9942955, 1.0041162, 0.9942955, 1.0041162, -0.0098207, 0.0098207)
7: (0.0070415, 0.0200694, 0.0070415, 0.0200694, -0.0097383, 0.0097383)
8: (0.0018589, 0.0072760, 0.0018589, 0.0072760, -0.0054171, 0.0054171)
9: (-0.0275324, -0.0137047, -0.0275324, -0.0137047, -0.0138277, 0.0138277)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.52 + 2.58 = 4.10 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0086815, upper bound: 0.0086815

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0082128, upper bound: 0.0082780
time: 1.33 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0081837, upper bound: 0.0081837
time: 1.72 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 3.20 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 3.20
Output dim: 6, lower bound: -0.0082128, upper bound: 0.0082780
IS_A2, status: Status.UNKNOWN, split count: 1, time: 3.20
Output dim: 6, lower bound: -0.0081837, upper bound: 0.0081837

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0039924, 0.0086006, 0.0038321, 0.0086154, -0.0046229, 0.0047685
1: -0.0000317, 0.0047407, -0.0001844, 0.0047693, -0.0048010, 0.0049251
2: -0.0252994, -0.0049694, -0.0256816, -0.0049325, -0.0203669, 0.0207122
3: -0.0020687, 0.0075784, -0.0020720, 0.0079101, -0.0099788, 0.0096504
4: 0.0112991, 0.0183802, 0.0112831, 0.0184802, -0.0071811, 0.0070971
5: -0.0032635, 0.0094226, -0.0032784, 0.0098897, -0.0131533, 0.0127011
6: 0.9942998, 1.0038015, 0.9942955, 1.0041162, -0.0098163, 0.0095060
7: 0.0070704, 0.0198885, 0.0070415, 0.0200694, -0.0097066, 0.0095814
8: 0.0020004, 0.0072193, 0.0018589, 0.0072760, -0.0052755, 0.0053604
9: -0.0272113, -0.0137228, -0.0275324, -0.0137047, -0.0135066, 0.0138096

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=22, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 132

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0081837, upper bound: 0.0081837
time: 1.09 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0081837, upper bound: 0.0081837
time: 1.35 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0043444, 0.0085682, 0.0040180, 0.0085983, -0.0042539, 0.0045502
1: 0.0003036, 0.0046779, -0.0000073, 0.0047362, -0.0044325, 0.0046853
2: -0.0244603, -0.0039456, -0.0252385, -0.0049896, -0.0194707, 0.0212929
3: -0.0021602, 0.0068502, -0.0020669, 0.0075255, -0.0096857, 0.0089171
4: 0.0108554, 0.0181608, 0.0113078, 0.0183643, -0.0075089, 0.0068530
5: -0.0032308, 0.0083972, -0.0032611, 0.0093482, -0.0125789, 0.0116583
6: 0.9941784, 1.0031105, 0.9943023, 1.0037513, -0.0095729, 0.0088083
7: 0.0062673, 0.0194913, 0.0070863, 0.0198597, -0.0103394, 0.0093788
8: 0.0023113, 0.0070948, 0.0020230, 0.0072102, -0.0048989, 0.0050718
9: -0.0265064, -0.0132207, -0.0271601, -0.0137327, -0.0127737, 0.0139395

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=22, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0078842, upper bound: 0.0080354
time: 1.49 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0081031, upper bound: 0.0081032
time: 1.71 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.78 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.78
Output dim: 6, lower bound: -0.0081837, upper bound: 0.0081837
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.78
Output dim: 6, lower bound: -0.0081837, upper bound: 0.0081837
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.78
Output dim: 6, lower bound: -0.0078842, upper bound: 0.0080354
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.78
Output dim: 6, lower bound: -0.0081031, upper bound: 0.0081032

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.0039924, 0.0086006, 0.0039924, 0.0086006, -0.0046082, 0.0046082
1: -0.0000317, 0.0047407, -0.0000317, 0.0047407, -0.0047724, 0.0047724
2: -0.0252994, -0.0049694, -0.0252994, -0.0049694, -0.0203300, 0.0203300
3: -0.0020687, 0.0075784, -0.0020687, 0.0075784, -0.0096471, 0.0096471
4: 0.0112991, 0.0183802, 0.0112991, 0.0183802, -0.0070811, 0.0070811
5: -0.0032635, 0.0094226, -0.0032635, 0.0094226, -0.0126862, 0.0126862
6: 0.9942998, 1.0038015, 0.9942998, 1.0038015, -0.0095016, 0.0095016
7: 0.0070704, 0.0198885, 0.0070704, 0.0198885, -0.0095497, 0.0095497
8: 0.0020004, 0.0072193, 0.0020004, 0.0072193, -0.0052189, 0.0052189
9: -0.0272113, -0.0137228, -0.0272113, -0.0137228, -0.0134885, 0.0134885

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0080850, upper bound: 0.0080127
time: 1.38 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0081328, upper bound: 0.0081965
time: 1.95 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0039924, 0.0086006, 0.0043444, 0.0085682, -0.0045758, 0.0042562
1: -0.0000317, 0.0047407, 0.0003036, 0.0046779, -0.0047096, 0.0044371
2: -0.0252994, -0.0049694, -0.0244603, -0.0039456, -0.0213538, 0.0194908
3: -0.0020687, 0.0075784, -0.0021602, 0.0068502, -0.0089189, 0.0097385
4: 0.0112991, 0.0183802, 0.0108554, 0.0181608, -0.0068617, 0.0075248
5: -0.0032635, 0.0094226, -0.0032308, 0.0083972, -0.0116607, 0.0126534
6: 0.9942998, 1.0038015, 0.9941784, 1.0031105, -0.0088107, 0.0096231
7: 0.0070704, 0.0198885, 0.0062673, 0.0194913, -0.0092097, 0.0103821
8: 0.0020004, 0.0072193, 0.0023113, 0.0070948, -0.0050944, 0.0049080
9: -0.0272113, -0.0137228, -0.0265064, -0.0132207, -0.0139907, 0.0127835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0080850, upper bound: 0.0080127
time: 1.43 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0081328, upper bound: 0.0081965
time: 1.35 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.0044347, 0.0085599, 0.0042527, 0.0085767, -0.0041419, 0.0043072
1: 0.0003896, 0.0046618, 0.0002163, 0.0046943, -0.0043046, 0.0044455
2: -0.0242450, -0.0040007, -0.0246788, -0.0052955, -0.0189495, 0.0206780
3: -0.0021552, 0.0066634, -0.0020396, 0.0070398, -0.0091950, 0.0087030
4: 0.0108793, 0.0181045, 0.0114404, 0.0182179, -0.0073386, 0.0066641
5: -0.0032224, 0.0081341, -0.0032393, 0.0086642, -0.0118866, 0.0113734
6: 0.9941850, 1.0029333, 0.9943386, 1.0032903, -0.0091053, 0.0085947
7: 0.0063106, 0.0193894, 0.0073262, 0.0195947, -0.0100404, 0.0090840
8: 0.0023911, 0.0070629, 0.0022303, 0.0071272, -0.0047362, 0.0048326
9: -0.0263255, -0.0132477, -0.0266899, -0.0138827, -0.0124428, 0.0134422

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 90

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0076731, upper bound: 0.0078627
time: 1.35 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0076515, upper bound: 0.0077958
time: 1.81 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.0043913, 0.0085639, 0.0041430, 0.0085868, -0.0041955, 0.0044209
1: 0.0003483, 0.0046696, 0.0001118, 0.0047139, -0.0043656, 0.0045578
2: -0.0243485, -0.0039747, -0.0249404, -0.0050658, -0.0192828, 0.0209658
3: -0.0021576, 0.0067532, -0.0020601, 0.0072669, -0.0094245, 0.0088134
4: 0.0108680, 0.0181316, 0.0113408, 0.0182864, -0.0074183, 0.0067907
5: -0.0032264, 0.0082607, -0.0032495, 0.0089840, -0.0122104, 0.0115102
6: 0.9941819, 1.0030185, 0.9943113, 1.0035059, -0.0093241, 0.0087072
7: 0.0062901, 0.0194384, 0.0071460, 0.0197186, -0.0101086, 0.0092666
8: 0.0023527, 0.0070783, 0.0021334, 0.0071660, -0.0048134, 0.0049449
9: -0.0264125, -0.0132349, -0.0269098, -0.0137701, -0.0126424, 0.0136749

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 90

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0078885, upper bound: 0.0079480
time: 1.27 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0078645, upper bound: 0.0078645
time: 1.54 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.29 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.29
Output dim: 6, lower bound: -0.0080850, upper bound: 0.0080127
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.29
Output dim: 6, lower bound: -0.0081328, upper bound: 0.0081965
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.29
Output dim: 6, lower bound: -0.0080850, upper bound: 0.0080127
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.29
Output dim: 6, lower bound: -0.0081328, upper bound: 0.0081965
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.29
Output dim: 6, lower bound: -0.0076731, upper bound: 0.0078627
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.29
Output dim: 6, lower bound: -0.0076515, upper bound: 0.0077958
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.29
Output dim: 6, lower bound: -0.0078885, upper bound: 0.0079480
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.29
Output dim: 6, lower bound: -0.0078645, upper bound: 0.0078645

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0042255, 0.0085792, 0.0040800, 0.0085926, -0.0043671, 0.0044992
1: 0.0001903, 0.0046992, 0.0000518, 0.0047251, -0.0045348, 0.0046474
2: -0.0247438, -0.0052749, -0.0250907, -0.0050247, -0.0197191, 0.0198157
3: -0.0020414, 0.0070962, -0.0020638, 0.0073972, -0.0094387, 0.0091600
4: 0.0114315, 0.0182349, 0.0113230, 0.0183256, -0.0068942, 0.0069119
5: -0.0032418, 0.0087437, -0.0032554, 0.0091675, -0.0124094, 0.0119991
6: 0.9943361, 1.0033439, 0.9943063, 1.0036296, -0.0092935, 0.0090376
7: 0.0073101, 0.0196255, 0.0071138, 0.0197897, -0.0092597, 0.0092541
8: 0.0022063, 0.0071369, 0.0020778, 0.0071883, -0.0049820, 0.0050591
9: -0.0267446, -0.0138727, -0.0270359, -0.0137499, -0.0129947, 0.0131633

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 92

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0083773, upper bound: 0.0082723
time: 1.65 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0083735, upper bound: 0.0082639
time: 1.79 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0041149, 0.0085893, 0.0040394, 0.0085963, -0.0044814, 0.0045499
1: 0.0000850, 0.0047189, 0.0000131, 0.0047323, -0.0046473, 0.0047057
2: -0.0250075, -0.0050454, -0.0251873, -0.0049985, -0.0200090, 0.0201419
3: -0.0020619, 0.0073250, -0.0020661, 0.0074811, -0.0095430, 0.0093911
4: 0.0113320, 0.0183039, 0.0113117, 0.0183509, -0.0070189, 0.0069922
5: -0.0032521, 0.0090659, -0.0032591, 0.0092856, -0.0125378, 0.0123250
6: 0.9943089, 1.0035611, 0.9943033, 1.0037091, -0.0094002, 0.0092578
7: 0.0071300, 0.0197503, 0.0070932, 0.0198354, -0.0094340, 0.0093183
8: 0.0021086, 0.0071760, 0.0020420, 0.0072027, -0.0050940, 0.0051340
9: -0.0269660, -0.0137601, -0.0271171, -0.0137371, -0.0132290, 0.0133571

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 92

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0084235, upper bound: 0.0084253
time: 1.69 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0084171, upper bound: 0.0084171
time: 1.53 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0042255, 0.0085792, 0.0044347, 0.0085599, -0.0043344, 0.0041445
1: 0.0001903, 0.0046992, 0.0003896, 0.0046618, -0.0044715, 0.0043095
2: -0.0247438, -0.0052749, -0.0242450, -0.0040007, -0.0207431, 0.0189700
3: -0.0020414, 0.0070962, -0.0021552, 0.0066634, -0.0087048, 0.0092515
4: 0.0114315, 0.0182349, 0.0108793, 0.0181045, -0.0066730, 0.0073556
5: -0.0032418, 0.0087437, -0.0032224, 0.0081341, -0.0113760, 0.0119661
6: 0.9943361, 1.0033439, 0.9941850, 1.0029333, -0.0085972, 0.0091590
7: 0.0073101, 0.0196255, 0.0063106, 0.0193894, -0.0089174, 0.0100877
8: 0.0022063, 0.0071369, 0.0023911, 0.0070629, -0.0048566, 0.0047458
9: -0.0267446, -0.0138727, -0.0263255, -0.0132477, -0.0134969, 0.0124529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 92

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0079416, upper bound: 0.0078687
time: 1.63 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0078622, upper bound: 0.0078294
time: 1.48 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0041149, 0.0085893, 0.0043913, 0.0085639, -0.0044490, 0.0041981
1: 0.0000850, 0.0047189, 0.0003483, 0.0046696, -0.0045846, 0.0043706
2: -0.0250075, -0.0050454, -0.0243485, -0.0039747, -0.0210328, 0.0193032
3: -0.0020619, 0.0073250, -0.0021576, 0.0067532, -0.0088152, 0.0094826
4: 0.0113320, 0.0183039, 0.0108680, 0.0181316, -0.0067996, 0.0074359
5: -0.0032521, 0.0090659, -0.0032264, 0.0082607, -0.0115128, 0.0122923
6: 0.9943089, 1.0035611, 0.9941819, 1.0030185, -0.0087096, 0.0093793
7: 0.0071300, 0.0197503, 0.0062901, 0.0194384, -0.0090933, 0.0101511
8: 0.0021086, 0.0071760, 0.0023527, 0.0070783, -0.0049696, 0.0048233
9: -0.0269660, -0.0137601, -0.0264125, -0.0132349, -0.0137311, 0.0126525

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 92

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0079974, upper bound: 0.0080592
time: 1.53 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0079103, upper bound: 0.0080216
time: 1.33 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0044509, 0.0085584, 0.0042567, 0.0085763, -0.0041253, 0.0043017
1: 0.0004051, 0.0046589, 0.0002201, 0.0046936, -0.0042884, 0.0044388
2: -0.0242063, -0.0042651, -0.0246693, -0.0053669, -0.0188394, 0.0204042
3: -0.0021316, 0.0066298, -0.0020332, 0.0070316, -0.0091632, 0.0086630
4: 0.0109939, 0.0180943, 0.0114713, 0.0182154, -0.0072216, 0.0066230
5: -0.0032208, 0.0080868, -0.0032389, 0.0086527, -0.0118735, 0.0113257
6: 0.9942163, 1.0029014, 0.9943470, 1.0032825, -0.0090663, 0.0085545
7: 0.0065179, 0.0193710, 0.0073822, 0.0195902, -0.0098332, 0.0090134
8: 0.0024054, 0.0070572, 0.0022339, 0.0071258, -0.0047204, 0.0048233
9: -0.0262930, -0.0133773, -0.0266820, -0.0139178, -0.0123752, 0.0133046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074985, upper bound: 0.0077131
time: 2.02 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0075588, upper bound: 0.0077457
time: 1.33 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0042914, 0.0085731, 0.0042630, 0.0085757, -0.0042843, 0.0043101
1: 0.0002531, 0.0046874, 0.0002261, 0.0046925, -0.0044393, 0.0044613
2: -0.0245867, -0.0044210, -0.0246543, -0.0055109, -0.0190758, 0.0202334
3: -0.0021177, 0.0069599, -0.0020204, 0.0070186, -0.0091363, 0.0089802
4: 0.0110614, 0.0181938, 0.0115337, 0.0182115, -0.0071501, 0.0066601
5: -0.0032357, 0.0085517, -0.0032383, 0.0086343, -0.0118700, 0.0117900
6: 0.9942348, 1.0032145, 0.9943641, 1.0032703, -0.0090355, 0.0088504
7: 0.0066402, 0.0195511, 0.0074952, 0.0195831, -0.0098032, 0.0090047
8: 0.0022645, 0.0071136, 0.0022394, 0.0071236, -0.0048591, 0.0048742
9: -0.0266126, -0.0134538, -0.0266694, -0.0139884, -0.0126242, 0.0132156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074850, upper bound: 0.0076793
time: 1.71 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0075376, upper bound: 0.0077051
time: 1.44 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0044074, 0.0085624, 0.0041469, 0.0085864, -0.0041790, 0.0044155
1: 0.0003636, 0.0046667, 0.0001155, 0.0047132, -0.0043496, 0.0045512
2: -0.0243102, -0.0042389, -0.0249311, -0.0051356, -0.0191746, 0.0206922
3: -0.0021340, 0.0067199, -0.0020539, 0.0072587, -0.0093927, 0.0087738
4: 0.0109825, 0.0181215, 0.0113711, 0.0182839, -0.0073014, 0.0067505
5: -0.0032249, 0.0082138, -0.0032491, 0.0089725, -0.0121974, 0.0114629
6: 0.9942132, 1.0029870, 0.9943196, 1.0034982, -0.0092850, 0.0086674
7: 0.0064974, 0.0194202, 0.0072008, 0.0197141, -0.0099025, 0.0091962
8: 0.0023669, 0.0070726, 0.0021369, 0.0071646, -0.0047978, 0.0049357
9: -0.0263803, -0.0133645, -0.0269019, -0.0138043, -0.0125760, 0.0135374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0076920, upper bound: 0.0078038
time: 1.51 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0077999, upper bound: 0.0078364
time: 1.79 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0042443, 0.0085774, 0.0041528, 0.0085859, -0.0043415, 0.0044246
1: 0.0002083, 0.0046958, 0.0001211, 0.0047121, -0.0045038, 0.0045747
2: -0.0246989, -0.0043949, -0.0249170, -0.0052848, -0.0194141, 0.0205221
3: -0.0021200, 0.0070572, -0.0020406, 0.0072465, -0.0093666, 0.0090978
4: 0.0110501, 0.0182232, 0.0114357, 0.0182802, -0.0072301, 0.0067874
5: -0.0032401, 0.0086888, -0.0032486, 0.0089554, -0.0121954, 0.0119374
6: 0.9942317, 1.0033069, 0.9943373, 1.0034865, -0.0092548, 0.0089695
7: 0.0066197, 0.0196042, 0.0073178, 0.0197075, -0.0098727, 0.0091636
8: 0.0022229, 0.0071302, 0.0021421, 0.0071626, -0.0049397, 0.0049881
9: -0.0267068, -0.0134410, -0.0268900, -0.0138775, -0.0128293, 0.0134490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0076764, upper bound: 0.0077529
time: 1.41 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0077755, upper bound: 0.0077755
time: 1.40 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.34 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.34
Output dim: 6, lower bound: -0.0083773, upper bound: 0.0082723
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.34
Output dim: 6, lower bound: -0.0083735, upper bound: 0.0082639
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.34
Output dim: 6, lower bound: -0.0084235, upper bound: 0.0084253
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.34
Output dim: 6, lower bound: -0.0084171, upper bound: 0.0084171
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.34
Output dim: 6, lower bound: -0.0079416, upper bound: 0.0078687
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.34
Output dim: 6, lower bound: -0.0078622, upper bound: 0.0078294
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.34
Output dim: 6, lower bound: -0.0079974, upper bound: 0.0080592
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.34
Output dim: 6, lower bound: -0.0079103, upper bound: 0.0080216
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.34
Output dim: 6, lower bound: -0.0074985, upper bound: 0.0077131
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.34
Output dim: 6, lower bound: -0.0075588, upper bound: 0.0077457
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.34
Output dim: 6, lower bound: -0.0074850, upper bound: 0.0076793
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.34
Output dim: 6, lower bound: -0.0075376, upper bound: 0.0077051
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.34
Output dim: 6, lower bound: -0.0076920, upper bound: 0.0078038
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.34
Output dim: 6, lower bound: -0.0077999, upper bound: 0.0078364
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.34
Output dim: 6, lower bound: -0.0076764, upper bound: 0.0077529
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.34
Output dim: 6, lower bound: -0.0077755, upper bound: 0.0077755

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0042292, 0.0085788, 0.0040943, 0.0085912, -0.0043620, 0.0044845
1: 0.0001939, 0.0046985, 0.0000654, 0.0047226, -0.0045286, 0.0046331
2: -0.0247348, -0.0053464, -0.0250566, -0.0052838, -0.0194510, 0.0197101
3: -0.0020351, 0.0070885, -0.0020406, 0.0073676, -0.0094027, 0.0091291
4: 0.0114624, 0.0182326, 0.0114353, 0.0183167, -0.0068543, 0.0067973
5: -0.0032415, 0.0087328, -0.0032540, 0.0091259, -0.0123674, 0.0119868
6: 0.9943447, 1.0033365, 0.9943373, 1.0036014, -0.0092568, 0.0089993
7: 0.0073662, 0.0196213, 0.0073170, 0.0197735, -0.0091918, 0.0090574
8: 0.0022096, 0.0071355, 0.0020904, 0.0071833, -0.0049737, 0.0050451
9: -0.0267370, -0.0139077, -0.0270073, -0.0138770, -0.0128600, 0.0130996

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0082604, upper bound: 0.0081080
time: 1.89 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0082946, upper bound: 0.0081922
time: 1.62 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0042354, 0.0085783, 0.0039411, 0.0086054, -0.0043700, 0.0046372
1: 0.0001998, 0.0046974, -0.0000806, 0.0047499, -0.0045501, 0.0047780
2: -0.0247203, -0.0054904, -0.0254219, -0.0054580, -0.0192623, 0.0199315
3: -0.0020222, 0.0070758, -0.0020251, 0.0076846, -0.0097068, 0.0091009
4: 0.0115248, 0.0182288, 0.0115108, 0.0184122, -0.0068874, 0.0067180
5: -0.0032409, 0.0087149, -0.0032683, 0.0095723, -0.0128132, 0.0119832
6: 0.9943616, 1.0033246, 0.9943578, 1.0039023, -0.0095407, 0.0089668
7: 0.0074791, 0.0196143, 0.0074537, 0.0199465, -0.0092244, 0.0090254
8: 0.0022150, 0.0071334, 0.0019550, 0.0072374, -0.0050224, 0.0051783
9: -0.0267248, -0.0139783, -0.0273142, -0.0139624, -0.0127623, 0.0133359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0082543, upper bound: 0.0081027
time: 1.53 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0082942, upper bound: 0.0081831
time: 1.49 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0041187, 0.0085890, 0.0040536, 0.0085950, -0.0044763, 0.0045354
1: 0.0000886, 0.0047182, 0.0000266, 0.0047298, -0.0046412, 0.0046916
2: -0.0249984, -0.0051152, -0.0251535, -0.0052577, -0.0197407, 0.0200383
3: -0.0020557, 0.0073172, -0.0020430, 0.0074518, -0.0095075, 0.0093601
4: 0.0113622, 0.0183015, 0.0114240, 0.0183421, -0.0069798, 0.0068775
5: -0.0032518, 0.0090548, -0.0032578, 0.0092444, -0.0124961, 0.0123126
6: 0.9943172, 1.0035536, 0.9943341, 1.0036812, -0.0093640, 0.0092195
7: 0.0071848, 0.0197460, 0.0072966, 0.0198194, -0.0093661, 0.0091196
8: 0.0021120, 0.0071746, 0.0020545, 0.0071976, -0.0050857, 0.0051202
9: -0.0269584, -0.0137943, -0.0270888, -0.0138642, -0.0130942, 0.0132945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0083005, upper bound: 0.0082224
time: 1.69 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0083423, upper bound: 0.0083494
time: 1.79 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0041245, 0.0085885, 0.0039016, 0.0086090, -0.0044845, 0.0046869
1: 0.0000941, 0.0047172, -0.0001182, 0.0047569, -0.0046628, 0.0048354
2: -0.0249846, -0.0052644, -0.0255160, -0.0054311, -0.0195535, 0.0202517
3: -0.0020424, 0.0073052, -0.0020275, 0.0077664, -0.0098087, 0.0093327
4: 0.0114269, 0.0182979, 0.0114991, 0.0184369, -0.0070100, 0.0067988
5: -0.0032512, 0.0090379, -0.0032720, 0.0096874, -0.0129386, 0.0123099
6: 0.9943349, 1.0035422, 0.9943547, 1.0039797, -0.0096447, 0.0091875
7: 0.0073018, 0.0197395, 0.0074326, 0.0199910, -0.0093962, 0.0090872
8: 0.0021170, 0.0071726, 0.0019202, 0.0072514, -0.0051344, 0.0052524
9: -0.0269468, -0.0138675, -0.0273933, -0.0139492, -0.0129976, 0.0135258

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0082962, upper bound: 0.0082182
time: 1.56 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0083390, upper bound: 0.0083390
time: 1.75 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0042292, 0.0085788, 0.0044509, 0.0085584, -0.0043292, 0.0041279
1: 0.0001939, 0.0046985, 0.0004051, 0.0046589, -0.0044650, 0.0042933
2: -0.0247348, -0.0053464, -0.0242063, -0.0042651, -0.0204698, 0.0188598
3: -0.0020351, 0.0070885, -0.0021316, 0.0066298, -0.0086648, 0.0092201
4: 0.0114624, 0.0182326, 0.0109939, 0.0180943, -0.0066319, 0.0072387
5: -0.0032415, 0.0087328, -0.0032208, 0.0080868, -0.0113283, 0.0119536
6: 0.9943447, 1.0033365, 0.9942163, 1.0029014, -0.0085568, 0.0091203
7: 0.0073662, 0.0196213, 0.0065179, 0.0193710, -0.0088454, 0.0098808
8: 0.0022096, 0.0071355, 0.0024054, 0.0070572, -0.0048476, 0.0047301
9: -0.0267370, -0.0139077, -0.0262930, -0.0133773, -0.0133597, 0.0123852

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0077929, upper bound: 0.0076935
time: 1.55 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0078205, upper bound: 0.0077696
time: 1.44 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0042354, 0.0085783, 0.0042914, 0.0085731, -0.0043377, 0.0042869
1: 0.0001998, 0.0046974, 0.0002531, 0.0046874, -0.0044876, 0.0044443
2: -0.0247203, -0.0054904, -0.0245867, -0.0044210, -0.0202993, 0.0190963
3: -0.0020222, 0.0070758, -0.0021177, 0.0069599, -0.0089821, 0.0091935
4: 0.0115248, 0.0182288, 0.0110614, 0.0181938, -0.0066690, 0.0071673
5: -0.0032409, 0.0087149, -0.0032357, 0.0085517, -0.0117926, 0.0119506
6: 0.9943616, 1.0033246, 0.9942348, 1.0032145, -0.0088528, 0.0090898
7: 0.0074791, 0.0196143, 0.0066402, 0.0195511, -0.0088983, 0.0098511
8: 0.0022150, 0.0071334, 0.0022645, 0.0071136, -0.0048986, 0.0048689
9: -0.0267248, -0.0139783, -0.0266126, -0.0134538, -0.0132710, 0.0126342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0077449, upper bound: 0.0076693
time: 1.89 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0077723, upper bound: 0.0077303
time: 1.54 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0041187, 0.0085890, 0.0044074, 0.0085624, -0.0044437, 0.0041816
1: 0.0000886, 0.0047182, 0.0003636, 0.0046667, -0.0045781, 0.0043546
2: -0.0249984, -0.0051152, -0.0243102, -0.0042389, -0.0207595, 0.0191950
3: -0.0020557, 0.0073172, -0.0021340, 0.0067199, -0.0087756, 0.0094511
4: 0.0113622, 0.0183015, 0.0109825, 0.0181215, -0.0067593, 0.0073190
5: -0.0032518, 0.0090548, -0.0032249, 0.0082138, -0.0114656, 0.0122797
6: 0.9943172, 1.0035536, 0.9942132, 1.0029870, -0.0086699, 0.0093404
7: 0.0071848, 0.0197460, 0.0064974, 0.0194202, -0.0090213, 0.0099454
8: 0.0021120, 0.0071746, 0.0023669, 0.0070726, -0.0049606, 0.0048077
9: -0.0269584, -0.0137943, -0.0263803, -0.0133645, -0.0135939, 0.0125860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0078641, upper bound: 0.0078753
time: 1.40 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0078843, upper bound: 0.0079745
time: 1.52 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0041245, 0.0085885, 0.0042443, 0.0085774, -0.0044529, 0.0043441
1: 0.0000941, 0.0047172, 0.0002083, 0.0046958, -0.0046017, 0.0045089
2: -0.0249846, -0.0052644, -0.0246989, -0.0043949, -0.0205897, 0.0194345
3: -0.0020424, 0.0073052, -0.0021200, 0.0070572, -0.0090996, 0.0094252
4: 0.0114269, 0.0182979, 0.0110501, 0.0182232, -0.0067963, 0.0072478
5: -0.0032512, 0.0090379, -0.0032401, 0.0086888, -0.0119400, 0.0122780
6: 0.9943349, 1.0035422, 0.9942317, 1.0033069, -0.0089719, 0.0093105
7: 0.0073018, 0.0197395, 0.0066197, 0.0196042, -0.0090588, 0.0099161
8: 0.0021170, 0.0071726, 0.0022229, 0.0071302, -0.0050132, 0.0049497
9: -0.0269468, -0.0138675, -0.0267068, -0.0134410, -0.0135058, 0.0128393

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0077990, upper bound: 0.0078430
time: 1.84 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0078200, upper bound: 0.0079363
time: 1.42 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0045125, 0.0085527, 0.0044173, 0.0085615, -0.0040490, 0.0041354
1: 0.0004638, 0.0046480, 0.0003731, 0.0046649, -0.0042012, 0.0042749
2: -0.0240595, -0.0043202, -0.0242865, -0.0055066, -0.0185530, 0.0199663
3: -0.0021267, 0.0065025, -0.0020208, 0.0066994, -0.0088261, 0.0085232
4: 0.0110177, 0.0180560, 0.0115318, 0.0181153, -0.0070976, 0.0065241
5: -0.0032151, 0.0079075, -0.0032240, 0.0081848, -0.0113999, 0.0111315
6: 0.9942228, 1.0027806, 0.9943635, 1.0029675, -0.0087447, 0.0084170
7: 0.0065612, 0.0193016, 0.0074918, 0.0194090, -0.0096227, 0.0088350
8: 0.0024598, 0.0070354, 0.0023757, 0.0070691, -0.0046093, 0.0046597
9: -0.0261697, -0.0134044, -0.0263604, -0.0139863, -0.0121835, 0.0129560

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069942, upper bound: 0.0072348
time: 1.40 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069409, upper bound: 0.0072175
time: 1.29 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0045226, 0.0085518, 0.0044075, 0.0085624, -0.0040398, 0.0041443
1: 0.0004734, 0.0046462, 0.0003637, 0.0046667, -0.0041933, 0.0042824
2: -0.0240354, -0.0043363, -0.0243099, -0.0053020, -0.0187334, 0.0199736
3: -0.0021253, 0.0064815, -0.0020390, 0.0067197, -0.0088450, 0.0085205
4: 0.0110247, 0.0180497, 0.0114432, 0.0181215, -0.0070968, 0.0066065
5: -0.0032142, 0.0078780, -0.0032249, 0.0082135, -0.0114277, 0.0111029
6: 0.9942247, 1.0027608, 0.9943394, 1.0029867, -0.0087619, 0.0084214
7: 0.0065738, 0.0192901, 0.0073313, 0.0194201, -0.0096265, 0.0090178
8: 0.0024687, 0.0070318, 0.0023670, 0.0070725, -0.0046038, 0.0046648
9: -0.0261494, -0.0134123, -0.0263801, -0.0138859, -0.0122635, 0.0129678

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0070337, upper bound: 0.0072348
time: 1.25 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069912, upper bound: 0.0072175
time: 1.10 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0043506, 0.0085676, 0.0044239, 0.0085609, -0.0042103, 0.0041437
1: 0.0003096, 0.0046768, 0.0003794, 0.0046638, -0.0043542, 0.0042974
2: -0.0244454, -0.0044769, -0.0242707, -0.0056498, -0.0187956, 0.0197938
3: -0.0021127, 0.0068373, -0.0020080, 0.0066856, -0.0087984, 0.0088452
4: 0.0110856, 0.0181569, 0.0115939, 0.0181112, -0.0070256, 0.0065630
5: -0.0032302, 0.0083790, -0.0032234, 0.0081655, -0.0113956, 0.0116024
6: 0.9942415, 1.0030981, 0.9943807, 1.0029544, -0.0087129, 0.0087175
7: 0.0066841, 0.0194842, 0.0076042, 0.0194015, -0.0095929, 0.0088272
8: 0.0023168, 0.0070926, 0.0023816, 0.0070667, -0.0047499, 0.0047111
9: -0.0264939, -0.0134812, -0.0263471, -0.0140565, -0.0124373, 0.0128659

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 90

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069782, upper bound: 0.0071950
time: 1.47 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069282, upper bound: 0.0071775
time: 1.30 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0043672, 0.0085661, 0.0044136, 0.0085618, -0.0041947, 0.0041526
1: 0.0003253, 0.0046739, 0.0003695, 0.0046656, -0.0043403, 0.0043044
2: -0.0244060, -0.0044894, -0.0242954, -0.0054562, -0.0189498, 0.0198060
3: -0.0021116, 0.0068031, -0.0020253, 0.0067071, -0.0088187, 0.0088283
4: 0.0110911, 0.0181466, 0.0115100, 0.0181177, -0.0070266, 0.0066366
5: -0.0032286, 0.0083309, -0.0032243, 0.0081957, -0.0114244, 0.0115552
6: 0.9942430, 1.0030657, 0.9943576, 1.0029747, -0.0087318, 0.0087081
7: 0.0066939, 0.0194656, 0.0074522, 0.0194132, -0.0095968, 0.0089642
8: 0.0023314, 0.0070868, 0.0023724, 0.0070704, -0.0047390, 0.0047144
9: -0.0264608, -0.0134874, -0.0263679, -0.0139616, -0.0124992, 0.0128805

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 90

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0070110, upper bound: 0.0071962
time: 1.41 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069649, upper bound: 0.0071793
time: 1.49 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0044688, 0.0085568, 0.0043055, 0.0085718, -0.0041030, 0.0042513
1: 0.0004221, 0.0046558, 0.0002665, 0.0046849, -0.0042628, 0.0043892
2: -0.0241637, -0.0042942, -0.0245531, -0.0052813, -0.0188825, 0.0202590
3: -0.0021290, 0.0065929, -0.0020409, 0.0069308, -0.0090598, 0.0086337
4: 0.0110065, 0.0180832, 0.0114342, 0.0181851, -0.0071786, 0.0066490
5: -0.0032192, 0.0080349, -0.0032344, 0.0085107, -0.0117299, 0.0112692
6: 0.9942197, 1.0028664, 0.9943368, 1.0031869, -0.0089672, 0.0085295
7: 0.0065407, 0.0193509, 0.0073150, 0.0195352, -0.0096907, 0.0090123
8: 0.0024212, 0.0070509, 0.0022769, 0.0071086, -0.0046874, 0.0047739
9: -0.0262573, -0.0133916, -0.0265844, -0.0138758, -0.0123815, 0.0131928

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 90

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073741, upper bound: 0.0074578
time: 1.73 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073278, upper bound: 0.0074334
time: 1.36 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0044792, 0.0085558, 0.0042976, 0.0085725, -0.0040933, 0.0042582
1: 0.0004321, 0.0046539, 0.0002591, 0.0046863, -0.0042542, 0.0043948
2: -0.0241388, -0.0043096, -0.0245717, -0.0050765, -0.0190622, 0.0202621
3: -0.0021277, 0.0065712, -0.0020592, 0.0069469, -0.0090745, 0.0086304
4: 0.0110132, 0.0180767, 0.0113455, 0.0181899, -0.0071768, 0.0067312
5: -0.0032182, 0.0080044, -0.0032351, 0.0085334, -0.0117516, 0.0112395
6: 0.9942216, 1.0028458, 0.9943126, 1.0032022, -0.0089806, 0.0085332
7: 0.0065529, 0.0193391, 0.0071545, 0.0195440, -0.0096956, 0.0091963
8: 0.0024304, 0.0070471, 0.0022700, 0.0071114, -0.0046810, 0.0047771
9: -0.0262363, -0.0133992, -0.0266000, -0.0137754, -0.0124609, 0.0132008

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 90

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074478, upper bound: 0.0074796
time: 1.06 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074025, upper bound: 0.0074521
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0043034, 0.0085720, 0.0043116, 0.0085712, -0.0042679, 0.0042604
1: 0.0002645, 0.0046853, 0.0002724, 0.0046838, -0.0044193, 0.0044129
2: -0.0245581, -0.0044505, -0.0245384, -0.0054304, -0.0191277, 0.0200879
3: -0.0021151, 0.0069351, -0.0020276, 0.0069180, -0.0090331, 0.0089627
4: 0.0110742, 0.0181864, 0.0114988, 0.0181812, -0.0071070, 0.0066875
5: -0.0032346, 0.0085168, -0.0032338, 0.0084928, -0.0117274, 0.0117506
6: 0.9942383, 1.0031912, 0.9943545, 1.0031750, -0.0089367, 0.0088367
7: 0.0066634, 0.0195376, 0.0074320, 0.0195283, -0.0096610, 0.0089794
8: 0.0022751, 0.0071093, 0.0022824, 0.0071064, -0.0048314, 0.0048270
9: -0.0265886, -0.0134683, -0.0265720, -0.0139489, -0.0126397, 0.0131037

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 90

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073603, upper bound: 0.0074202
time: 1.24 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073130, upper bound: 0.0073714
time: 1.24 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0043208, 0.0085704, 0.0043035, 0.0085720, -0.0042512, 0.0042668
1: 0.0002811, 0.0046822, 0.0002647, 0.0046852, -0.0044041, 0.0044174
2: -0.0245166, -0.0044631, -0.0245577, -0.0052359, -0.0192806, 0.0200946
3: -0.0021140, 0.0068991, -0.0020449, 0.0069347, -0.0090487, 0.0089440
4: 0.0110796, 0.0181755, 0.0114146, 0.0181863, -0.0071066, 0.0067609
5: -0.0032330, 0.0084660, -0.0032346, 0.0085163, -0.0117492, 0.0117006
6: 0.9942398, 1.0031569, 0.9943315, 1.0031906, -0.0089508, 0.0088254
7: 0.0066732, 0.0195179, 0.0072795, 0.0195374, -0.0096651, 0.0091214
8: 0.0022905, 0.0071032, 0.0022752, 0.0071093, -0.0048188, 0.0048280
9: -0.0265537, -0.0134744, -0.0265882, -0.0138535, -0.0127001, 0.0131138

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 90

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074247, upper bound: 0.0074353
time: 1.35 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073771, upper bound: 0.0073771
time: 1.30 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.23 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -0.0082604, upper bound: 0.0081080
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -0.0082946, upper bound: 0.0081922
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -0.0082543, upper bound: 0.0081027
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -0.0082942, upper bound: 0.0081831
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -0.0083005, upper bound: 0.0082224
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -0.0083423, upper bound: 0.0083494
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -0.0082962, upper bound: 0.0082182
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -0.0083390, upper bound: 0.0083390
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -0.0077929, upper bound: 0.0076935
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -0.0078205, upper bound: 0.0077696
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -0.0077449, upper bound: 0.0076693
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -0.0077723, upper bound: 0.0077303
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -0.0078641, upper bound: 0.0078753
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -0.0078843, upper bound: 0.0079745
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -0.0077990, upper bound: 0.0078430
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -0.0078200, upper bound: 0.0079363
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -0.0069942, upper bound: 0.0072348
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -0.0069409, upper bound: 0.0072175
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -0.0070337, upper bound: 0.0072348
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -0.0069912, upper bound: 0.0072175
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -0.0069782, upper bound: 0.0071950
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -0.0069282, upper bound: 0.0071775
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -0.0070110, upper bound: 0.0071962
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -0.0069649, upper bound: 0.0071793
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -0.0073741, upper bound: 0.0074578
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -0.0073278, upper bound: 0.0074334
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -0.0074478, upper bound: 0.0074796
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -0.0074025, upper bound: 0.0074521
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -0.0073603, upper bound: 0.0074202
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -0.0073130, upper bound: 0.0073714
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -0.0074247, upper bound: 0.0074353
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -0.0073771, upper bound: 0.0073771

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0043905, 0.0085640, 0.0041572, 0.0085855, -0.0041950, 0.0044067
1: 0.0003475, 0.0046697, 0.0001253, 0.0047113, -0.0043638, 0.0045444
2: -0.0243504, -0.0054868, -0.0249065, -0.0053404, -0.0190100, 0.0194197
3: -0.0020225, 0.0067549, -0.0020356, 0.0072375, -0.0092600, 0.0087905
4: 0.0115233, 0.0181321, 0.0114598, 0.0182775, -0.0067542, 0.0066722
5: -0.0032265, 0.0082630, -0.0032482, 0.0089426, -0.0121690, 0.0115112
6: 0.9943612, 1.0030200, 0.9943439, 1.0034778, -0.0091166, 0.0086762
7: 0.0074763, 0.0194393, 0.0073615, 0.0197025, -0.0090106, 0.0088388
8: 0.0023520, 0.0070785, 0.0021459, 0.0071610, -0.0048090, 0.0049326
9: -0.0264141, -0.0139766, -0.0268813, -0.0139048, -0.0125093, 0.0129047

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0079810, upper bound: 0.0078608
time: 1.59 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0079860, upper bound: 0.0078360
time: 1.73 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0043775, 0.0085652, 0.0041625, 0.0085850, -0.0042075, 0.0044027
1: 0.0003351, 0.0046720, 0.0001303, 0.0047104, -0.0043753, 0.0045417
2: -0.0243814, -0.0052818, -0.0248940, -0.0053574, -0.0190240, 0.0196122
3: -0.0020408, 0.0067817, -0.0020341, 0.0072266, -0.0092674, 0.0088158
4: 0.0114344, 0.0181401, 0.0114672, 0.0182742, -0.0068398, 0.0066729
5: -0.0032277, 0.0083008, -0.0032477, 0.0089273, -0.0121549, 0.0115485
6: 0.9943370, 1.0030456, 0.9943460, 1.0034677, -0.0091307, 0.0086996
7: 0.0073154, 0.0194539, 0.0073748, 0.0196966, -0.0091990, 0.0088405
8: 0.0023405, 0.0070831, 0.0021506, 0.0071592, -0.0048186, 0.0049325
9: -0.0264401, -0.0138760, -0.0268707, -0.0139131, -0.0125270, 0.0129947

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0079960, upper bound: 0.0079142
time: 1.68 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0080013, upper bound: 0.0078948
time: 1.68 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0043969, 0.0085634, 0.0040005, 0.0085999, -0.0042030, 0.0045629
1: 0.0003536, 0.0046686, -0.0000240, 0.0047393, -0.0043857, 0.0046926
2: -0.0243352, -0.0056298, -0.0252802, -0.0055151, -0.0188200, 0.0196503
3: -0.0020097, 0.0067416, -0.0020200, 0.0075617, -0.0095714, 0.0087616
4: 0.0115853, 0.0181281, 0.0115355, 0.0183752, -0.0067899, 0.0065925
5: -0.0032259, 0.0082443, -0.0032628, 0.0093992, -0.0126250, 0.0115071
6: 0.9943783, 1.0030075, 0.9943646, 1.0037856, -0.0094073, 0.0086430
7: 0.0075885, 0.0194320, 0.0074985, 0.0198794, -0.0090533, 0.0088067
8: 0.0023576, 0.0070763, 0.0020076, 0.0072164, -0.0048588, 0.0050687
9: -0.0264013, -0.0140467, -0.0271951, -0.0139905, -0.0124108, 0.0131484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0079684, upper bound: 0.0078558
time: 1.52 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0079721, upper bound: 0.0078306
time: 1.68 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0043833, 0.0085646, 0.0040147, 0.0085986, -0.0042153, 0.0045499
1: 0.0003406, 0.0046710, -0.0000104, 0.0047367, -0.0043961, 0.0046814
2: -0.0243676, -0.0054358, -0.0252462, -0.0055280, -0.0188396, 0.0198104
3: -0.0020271, 0.0067698, -0.0020188, 0.0075322, -0.0095593, 0.0087886
4: 0.0115012, 0.0181365, 0.0115411, 0.0183663, -0.0068651, 0.0065954
5: -0.0032271, 0.0082840, -0.0032614, 0.0093576, -0.0125848, 0.0115454
6: 0.9943553, 1.0030342, 0.9943661, 1.0037577, -0.0094025, 0.0086681
7: 0.0074363, 0.0194474, 0.0075086, 0.0198633, -0.0091963, 0.0088086
8: 0.0023457, 0.0070811, 0.0020202, 0.0072114, -0.0048657, 0.0050609
9: -0.0264286, -0.0139516, -0.0271666, -0.0139968, -0.0124318, 0.0132150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0079886, upper bound: 0.0079026
time: 1.52 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0079910, upper bound: 0.0078771
time: 1.42 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0042783, 0.0085743, 0.0041162, 0.0085892, -0.0043109, 0.0044581
1: 0.0002407, 0.0046897, 0.0000863, 0.0047186, -0.0044780, 0.0046034
2: -0.0246179, -0.0052611, -0.0250042, -0.0053143, -0.0193036, 0.0197431
3: -0.0020427, 0.0069869, -0.0020379, 0.0073222, -0.0093649, 0.0090248
4: 0.0114254, 0.0182020, 0.0114485, 0.0183030, -0.0068776, 0.0067535
5: -0.0032369, 0.0085897, -0.0032520, 0.0090619, -0.0122988, 0.0118417
6: 0.9943345, 1.0032402, 0.9943408, 1.0035583, -0.0092238, 0.0088995
7: 0.0072992, 0.0195659, 0.0073409, 0.0197488, -0.0091795, 0.0089006
8: 0.0022530, 0.0071182, 0.0021098, 0.0071755, -0.0049225, 0.0050084
9: -0.0266387, -0.0138659, -0.0269633, -0.0138920, -0.0127468, 0.0130975

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0080805, upper bound: 0.0080356
time: 1.56 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0080748, upper bound: 0.0080080
time: 1.59 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0042659, 0.0085754, 0.0041221, 0.0085887, -0.0043228, 0.0044533
1: 0.0002289, 0.0046919, 0.0000919, 0.0047176, -0.0044887, 0.0046001
2: -0.0246474, -0.0050568, -0.0249902, -0.0053307, -0.0193167, 0.0199335
3: -0.0020609, 0.0070126, -0.0020365, 0.0073101, -0.0093710, 0.0090490
4: 0.0113369, 0.0182097, 0.0114556, 0.0182994, -0.0069624, 0.0067541
5: -0.0032381, 0.0086259, -0.0032514, 0.0090448, -0.0122829, 0.0118773
6: 0.9943103, 1.0032645, 0.9943427, 1.0035468, -0.0092366, 0.0089218
7: 0.0071389, 0.0195799, 0.0073538, 0.0197422, -0.0093688, 0.0089028
8: 0.0022420, 0.0071226, 0.0021149, 0.0071734, -0.0049315, 0.0050076
9: -0.0266636, -0.0137657, -0.0269516, -0.0139000, -0.0127635, 0.0131859

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0081092, upper bound: 0.0081343
time: 2.22 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0081004, upper bound: 0.0081087
time: 1.45 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0042844, 0.0085737, 0.0039612, 0.0086035, -0.0043191, 0.0046126
1: 0.0002465, 0.0046886, -0.0000614, 0.0047463, -0.0044998, 0.0047501
2: -0.0246033, -0.0054100, -0.0253739, -0.0054881, -0.0191153, 0.0199639
3: -0.0020294, 0.0069743, -0.0020224, 0.0076430, -0.0096724, 0.0089967
4: 0.0114900, 0.0181982, 0.0115238, 0.0183997, -0.0069097, 0.0066744
5: -0.0032363, 0.0085720, -0.0032664, 0.0095137, -0.0127500, 0.0118384
6: 0.9943522, 1.0032283, 0.9943613, 1.0038627, -0.0095106, 0.0088670
7: 0.0074161, 0.0195590, 0.0074773, 0.0199238, -0.0092200, 0.0088683
8: 0.0022583, 0.0071160, 0.0019728, 0.0072303, -0.0049720, 0.0051432
9: -0.0266266, -0.0139389, -0.0272739, -0.0139772, -0.0126494, 0.0133349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 187

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0080742, upper bound: 0.0080284
time: 1.79 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0080637, upper bound: 0.0080001
time: 1.85 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0042715, 0.0085749, 0.0039751, 0.0086022, -0.0043307, 0.0045998
1: 0.0002342, 0.0046909, -0.0000482, 0.0047438, -0.0045096, 0.0047391
2: -0.0246340, -0.0052159, -0.0253408, -0.0055005, -0.0191335, 0.0201249
3: -0.0020467, 0.0070010, -0.0020213, 0.0076143, -0.0096610, 0.0090222
4: 0.0114059, 0.0182062, 0.0115292, 0.0183910, -0.0069851, 0.0066770
5: -0.0032375, 0.0086095, -0.0032651, 0.0094732, -0.0127108, 0.0118746
6: 0.9943291, 1.0032536, 0.9943628, 1.0038354, -0.0095063, 0.0088907
7: 0.0072638, 0.0195735, 0.0074870, 0.0199081, -0.0093661, 0.0088704
8: 0.0022470, 0.0071206, 0.0019851, 0.0072254, -0.0049784, 0.0051355
9: -0.0266523, -0.0138437, -0.0272461, -0.0139833, -0.0126690, 0.0134023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0081036, upper bound: 0.0081218
time: 1.59 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0080928, upper bound: 0.0080928
time: 1.49 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0043905, 0.0085640, 0.0045125, 0.0085527, -0.0041623, 0.0040515
1: 0.0003475, 0.0046697, 0.0004638, 0.0046480, -0.0043005, 0.0042060
2: -0.0243504, -0.0054868, -0.0240595, -0.0043202, -0.0200302, 0.0185727
3: -0.0020225, 0.0067549, -0.0021267, 0.0065025, -0.0085250, 0.0088816
4: 0.0115233, 0.0181321, 0.0110177, 0.0180560, -0.0065327, 0.0071143
5: -0.0032265, 0.0082630, -0.0032151, 0.0079075, -0.0111340, 0.0114781
6: 0.9943612, 1.0030200, 0.9942228, 1.0027806, -0.0084193, 0.0087972
7: 0.0074763, 0.0194393, 0.0065612, 0.0193016, -0.0086672, 0.0096637
8: 0.0023520, 0.0070785, 0.0024598, 0.0070354, -0.0046834, 0.0046188
9: -0.0264141, -0.0139766, -0.0261697, -0.0134044, -0.0130097, 0.0121931

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 90

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073953, upper bound: 0.0073470
time: 1.82 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073729, upper bound: 0.0072739
time: 1.34 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0043775, 0.0085652, 0.0045226, 0.0085518, -0.0041743, 0.0040426
1: 0.0003351, 0.0046720, 0.0004734, 0.0046462, -0.0043110, 0.0041986
2: -0.0243814, -0.0052818, -0.0240354, -0.0043363, -0.0200451, 0.0187536
3: -0.0020408, 0.0067817, -0.0021253, 0.0064815, -0.0085223, 0.0089070
4: 0.0114344, 0.0181401, 0.0110247, 0.0180497, -0.0066152, 0.0071154
5: -0.0032277, 0.0083008, -0.0032142, 0.0078780, -0.0111057, 0.0115150
6: 0.9943370, 1.0030456, 0.9942247, 1.0027608, -0.0084238, 0.0088208
7: 0.0073154, 0.0194539, 0.0065738, 0.0192901, -0.0088444, 0.0096663
8: 0.0023405, 0.0070831, 0.0024687, 0.0070318, -0.0046913, 0.0046144
9: -0.0264401, -0.0138760, -0.0261494, -0.0134123, -0.0130279, 0.0122734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 90

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073966, upper bound: 0.0074045
time: 1.27 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073740, upper bound: 0.0073378
time: 1.61 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0043969, 0.0085634, 0.0043506, 0.0085676, -0.0041708, 0.0042128
1: 0.0003536, 0.0046686, 0.0003096, 0.0046768, -0.0043232, 0.0043590
2: -0.0243352, -0.0056298, -0.0244454, -0.0044769, -0.0198583, 0.0188156
3: -0.0020097, 0.0067416, -0.0021127, 0.0068373, -0.0088470, 0.0088543
4: 0.0115853, 0.0181281, 0.0110856, 0.0181569, -0.0065716, 0.0070424
5: -0.0032259, 0.0082443, -0.0032302, 0.0083790, -0.0116049, 0.0114745
6: 0.9943783, 1.0030075, 0.9942415, 1.0030981, -0.0087199, 0.0087661
7: 0.0075885, 0.0194320, 0.0066841, 0.0194842, -0.0087178, 0.0096343
8: 0.0023576, 0.0070763, 0.0023168, 0.0070926, -0.0047350, 0.0047595
9: -0.0264013, -0.0140467, -0.0264939, -0.0134812, -0.0129200, 0.0124471

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 90

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073190, upper bound: 0.0072996
time: 1.34 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072817, upper bound: 0.0072143
time: 1.44 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0043833, 0.0085646, 0.0043672, 0.0085661, -0.0041829, 0.0041974
1: 0.0003406, 0.0046710, 0.0003253, 0.0046739, -0.0043332, 0.0043457
2: -0.0243676, -0.0054358, -0.0244060, -0.0044894, -0.0198782, 0.0189702
3: -0.0020271, 0.0067698, -0.0021116, 0.0068031, -0.0088301, 0.0088814
4: 0.0115012, 0.0181365, 0.0110911, 0.0181466, -0.0066454, 0.0070455
5: -0.0032271, 0.0082840, -0.0032286, 0.0083309, -0.0115580, 0.0115126
6: 0.9943553, 1.0030342, 0.9942430, 1.0030657, -0.0087104, 0.0087913
7: 0.0074363, 0.0194474, 0.0066939, 0.0194656, -0.0088578, 0.0096372
8: 0.0023457, 0.0070811, 0.0023314, 0.0070868, -0.0047411, 0.0047497
9: -0.0264286, -0.0139516, -0.0264608, -0.0134874, -0.0129412, 0.0125092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073246, upper bound: 0.0073511
time: 1.38 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072830, upper bound: 0.0072559
time: 1.87 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0042783, 0.0085743, 0.0044688, 0.0085568, -0.0042785, 0.0041055
1: 0.0002407, 0.0046897, 0.0004221, 0.0046558, -0.0044151, 0.0042676
2: -0.0246179, -0.0052611, -0.0241637, -0.0042942, -0.0203237, 0.0189027
3: -0.0020427, 0.0069869, -0.0021290, 0.0065929, -0.0086356, 0.0091160
4: 0.0114254, 0.0182020, 0.0110065, 0.0180832, -0.0066578, 0.0071955
5: -0.0032369, 0.0085897, -0.0032192, 0.0080349, -0.0112718, 0.0118089
6: 0.9943345, 1.0032402, 0.9942197, 1.0028664, -0.0085319, 0.0090205
7: 0.0072992, 0.0195659, 0.0065407, 0.0193509, -0.0088372, 0.0097284
8: 0.0022530, 0.0071182, 0.0024212, 0.0070509, -0.0047979, 0.0046970
9: -0.0266387, -0.0138659, -0.0262573, -0.0133916, -0.0132471, 0.0123914

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0075664, upper bound: 0.0076449
time: 1.31 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0075277, upper bound: 0.0075670
time: 1.24 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0042659, 0.0085754, 0.0044792, 0.0085558, -0.0042899, 0.0040962
1: 0.0002289, 0.0046919, 0.0004321, 0.0046539, -0.0044250, 0.0042598
2: -0.0246474, -0.0050568, -0.0241388, -0.0043096, -0.0203378, 0.0190820
3: -0.0020609, 0.0070126, -0.0021277, 0.0065712, -0.0086321, 0.0091402
4: 0.0113369, 0.0182097, 0.0110132, 0.0180767, -0.0067398, 0.0071966
5: -0.0032381, 0.0086259, -0.0032182, 0.0080044, -0.0112424, 0.0118441
6: 0.9943103, 1.0032645, 0.9942216, 1.0028458, -0.0085355, 0.0090430
7: 0.0071389, 0.0195799, 0.0065529, 0.0193391, -0.0090161, 0.0097322
8: 0.0022420, 0.0071226, 0.0024304, 0.0070471, -0.0048052, 0.0046922
9: -0.0266636, -0.0137657, -0.0262363, -0.0133992, -0.0132644, 0.0124706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0075890, upper bound: 0.0077266
time: 1.40 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0075391, upper bound: 0.0076634
time: 1.44 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0042844, 0.0085737, 0.0043034, 0.0085720, -0.0042876, 0.0042704
1: 0.0002465, 0.0046886, 0.0002645, 0.0046853, -0.0044388, 0.0044241
2: -0.0246033, -0.0054100, -0.0245581, -0.0044505, -0.0201528, 0.0191481
3: -0.0020294, 0.0069743, -0.0021151, 0.0069351, -0.0089645, 0.0090894
4: 0.0114900, 0.0181982, 0.0110742, 0.0181864, -0.0066964, 0.0071240
5: -0.0032363, 0.0085720, -0.0032346, 0.0085168, -0.0117531, 0.0118066
6: 0.9943522, 1.0032283, 0.9942383, 1.0031912, -0.0088391, 0.0089900
7: 0.0074161, 0.0195590, 0.0066634, 0.0195376, -0.0088718, 0.0096991
8: 0.0022583, 0.0071160, 0.0022751, 0.0071093, -0.0048510, 0.0048410
9: -0.0266266, -0.0139389, -0.0265886, -0.0134683, -0.0131583, 0.0126497

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 90

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0075044, upper bound: 0.0076126
time: 1.51 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074200, upper bound: 0.0075126
time: 1.67 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0042715, 0.0085749, 0.0043208, 0.0085704, -0.0042989, 0.0042541
1: 0.0002342, 0.0046909, 0.0002811, 0.0046822, -0.0044479, 0.0044098
2: -0.0246340, -0.0052159, -0.0245166, -0.0044631, -0.0201710, 0.0193007
3: -0.0020467, 0.0070010, -0.0021140, 0.0068991, -0.0089458, 0.0091149
4: 0.0114059, 0.0182062, 0.0110796, 0.0181755, -0.0067696, 0.0071266
5: -0.0032375, 0.0086095, -0.0032330, 0.0084660, -0.0117036, 0.0118425
6: 0.9943291, 1.0032536, 0.9942398, 1.0031569, -0.0088278, 0.0090138
7: 0.0072638, 0.0195735, 0.0066732, 0.0195179, -0.0090157, 0.0097022
8: 0.0022470, 0.0071206, 0.0022905, 0.0071032, -0.0048562, 0.0048301
9: -0.0266523, -0.0138437, -0.0265537, -0.0134744, -0.0131779, 0.0127100

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 90

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0075223, upper bound: 0.0076805
time: 1.78 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074257, upper bound: 0.0075914
time: 1.26 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0046254, 0.0085423, 0.0044320, 0.0085601, -0.0039347, 0.0041104
1: 0.0005714, 0.0046278, 0.0003871, 0.0046623, -0.0040910, 0.0042407
2: -0.0237902, -0.0043914, -0.0242514, -0.0055161, -0.0182741, 0.0198601
3: -0.0021204, 0.0062687, -0.0020199, 0.0066689, -0.0087893, 0.0082886
4: 0.0110486, 0.0179856, 0.0115360, 0.0181062, -0.0070576, 0.0064496
5: -0.0032046, 0.0075784, -0.0032226, 0.0081420, -0.0113466, 0.0108010
6: 0.9942313, 1.0025587, 0.9943647, 1.0029385, -0.0087072, 0.0081940
7: 0.0066170, 0.0191741, 0.0074993, 0.0193924, -0.0095506, 0.0086346
8: 0.0025595, 0.0069955, 0.0023887, 0.0070639, -0.0045043, 0.0046068
9: -0.0259435, -0.0134393, -0.0263309, -0.0139910, -0.0119525, 0.0128916

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069409, upper bound: 0.0072175
time: 1.27 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069409, upper bound: 0.0072175
time: 1.34 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0046815, 0.0085372, 0.0044994, 0.0085539, -0.0038725, 0.0040377
1: 0.0006247, 0.0046178, 0.0004513, 0.0046503, -0.0040255, 0.0041665
2: -0.0236566, -0.0038115, -0.0240907, -0.0055773, -0.0180793, 0.0202791
3: -0.0021721, 0.0061528, -0.0020144, 0.0065295, -0.0087016, 0.0081672
4: 0.0107973, 0.0179506, 0.0115625, 0.0180641, -0.0072668, 0.0063881
5: -0.0031994, 0.0074152, -0.0032163, 0.0079456, -0.0111450, 0.0106315
6: 0.9941625, 1.0024488, 0.9943721, 1.0028062, -0.0086437, 0.0080767
7: 0.0061621, 0.0191109, 0.0075473, 0.0193163, -0.0097507, 0.0086560
8: 0.0026091, 0.0069756, 0.0024482, 0.0070400, -0.0044309, 0.0045274
9: -0.0258312, -0.0131549, -0.0261959, -0.0140210, -0.0118103, 0.0130410

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069409, upper bound: 0.0072175
time: 1.27 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069409, upper bound: 0.0072175
time: 1.47 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0046367, 0.0085413, 0.0044224, 0.0085610, -0.0039244, 0.0041189
1: 0.0005821, 0.0046258, 0.0003779, 0.0046640, -0.0040820, 0.0042479
2: -0.0237634, -0.0044073, -0.0242744, -0.0053114, -0.0184520, 0.0198672
3: -0.0021189, 0.0062455, -0.0020382, 0.0066889, -0.0088078, 0.0082837
4: 0.0110555, 0.0179785, 0.0114473, 0.0181122, -0.0070567, 0.0065313
5: -0.0032036, 0.0075457, -0.0032235, 0.0081701, -0.0113736, 0.0107692
6: 0.9942332, 1.0025368, 0.9943405, 1.0029575, -0.0087243, 0.0081963
7: 0.0066295, 0.0191614, 0.0073387, 0.0194033, -0.0095546, 0.0088142
8: 0.0025694, 0.0069915, 0.0023801, 0.0070673, -0.0044978, 0.0046114
9: -0.0259210, -0.0134471, -0.0263502, -0.0138906, -0.0120304, 0.0129031

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 187

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069912, upper bound: 0.0072175
time: 1.20 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069912, upper bound: 0.0072175
time: 1.21 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0046925, 0.0085362, 0.0044883, 0.0085550, -0.0038625, 0.0040478
1: 0.0006353, 0.0046158, 0.0004407, 0.0046523, -0.0040170, 0.0041751
2: -0.0236303, -0.0038287, -0.0241172, -0.0053724, -0.0182579, 0.0202885
3: -0.0021706, 0.0061300, -0.0020327, 0.0065525, -0.0087231, 0.0081627
4: 0.0108047, 0.0179437, 0.0114737, 0.0180711, -0.0072663, 0.0064700
5: -0.0031984, 0.0073830, -0.0032174, 0.0079780, -0.0111764, 0.0106004
6: 0.9941646, 1.0024272, 0.9943477, 1.0028280, -0.0086634, 0.0080795
7: 0.0061756, 0.0190984, 0.0073866, 0.0193289, -0.0097510, 0.0088368
8: 0.0026188, 0.0069717, 0.0024384, 0.0070439, -0.0044252, 0.0045334
9: -0.0258091, -0.0131633, -0.0262182, -0.0139205, -0.0118886, 0.0130549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069912, upper bound: 0.0072175
time: 1.28 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069912, upper bound: 0.0072175
time: 1.41 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0044637, 0.0085572, 0.0044386, 0.0085595, -0.0040958, 0.0041186
1: 0.0004173, 0.0046567, 0.0003934, 0.0046611, -0.0042439, 0.0042633
2: -0.0241759, -0.0045456, -0.0242356, -0.0056594, -0.0185165, 0.0196900
3: -0.0021066, 0.0066034, -0.0020071, 0.0066552, -0.0087618, 0.0086105
4: 0.0111154, 0.0180864, 0.0115981, 0.0181020, -0.0069866, 0.0064883
5: -0.0032197, 0.0080497, -0.0032220, 0.0081227, -0.0113423, 0.0112717
6: 0.9942496, 1.0028764, 0.9943818, 1.0029255, -0.0086759, 0.0084946
7: 0.0067380, 0.0193566, 0.0076116, 0.0193849, -0.0093980, 0.0087106
8: 0.0024167, 0.0070526, 0.0023945, 0.0070615, -0.0046448, 0.0046581
9: -0.0262674, -0.0135149, -0.0263176, -0.0140612, -0.0122062, 0.0128027

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069282, upper bound: 0.0071775
time: 1.36 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069282, upper bound: 0.0071775
time: 1.37 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0045110, 0.0085529, 0.0045061, 0.0085533, -0.0040424, 0.0040468
1: 0.0004623, 0.0046482, 0.0004577, 0.0046491, -0.0041868, 0.0041906
2: -0.0240632, -0.0039587, -0.0240748, -0.0057212, -0.0183420, 0.0201161
3: -0.0021590, 0.0065056, -0.0020016, 0.0065156, -0.0086746, 0.0085072
4: 0.0108611, 0.0180569, 0.0116249, 0.0180600, -0.0071989, 0.0064321
5: -0.0032153, 0.0079120, -0.0032157, 0.0079261, -0.0111414, 0.0111277
6: 0.9941800, 1.0027835, 0.9943891, 1.0027931, -0.0086131, 0.0083945
7: 0.0062776, 0.0193033, 0.0076601, 0.0193088, -0.0097209, 0.0087318
8: 0.0024584, 0.0070359, 0.0024541, 0.0070377, -0.0045792, 0.0045818
9: -0.0261728, -0.0132271, -0.0261825, -0.0140916, -0.0120812, 0.0129555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069282, upper bound: 0.0071775
time: 1.34 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069282, upper bound: 0.0071775
time: 1.58 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0044781, 0.0085559, 0.0044285, 0.0085605, -0.0040824, 0.0041274
1: 0.0004310, 0.0046541, 0.0003837, 0.0046629, -0.0042319, 0.0042704
2: -0.0241415, -0.0045582, -0.0242599, -0.0054656, -0.0186759, 0.0197017
3: -0.0021055, 0.0065736, -0.0020244, 0.0066763, -0.0087817, 0.0085980
4: 0.0111209, 0.0180774, 0.0115141, 0.0181084, -0.0069875, 0.0065633
5: -0.0032183, 0.0080077, -0.0032229, 0.0081523, -0.0113706, 0.0112306
6: 0.9942511, 1.0028481, 0.9943587, 1.0029455, -0.0086945, 0.0084895
7: 0.0067478, 0.0193404, 0.0074596, 0.0193964, -0.0093987, 0.0088492
8: 0.0024294, 0.0070476, 0.0023855, 0.0070651, -0.0046357, 0.0046620
9: -0.0262386, -0.0135211, -0.0263380, -0.0139662, -0.0122724, 0.0128169

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069649, upper bound: 0.0071793
time: 1.00 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069649, upper bound: 0.0071793
time: 1.57 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0045280, 0.0085513, 0.0044944, 0.0085544, -0.0040264, 0.0040569
1: 0.0004785, 0.0046452, 0.0004466, 0.0046512, -0.0041727, 0.0041986
2: -0.0240226, -0.0039723, -0.0241026, -0.0055278, -0.0184949, 0.0201303
3: -0.0021578, 0.0064704, -0.0020189, 0.0065398, -0.0086976, 0.0084893
4: 0.0108670, 0.0180463, 0.0115410, 0.0180672, -0.0072003, 0.0065053
5: -0.0032137, 0.0078624, -0.0032168, 0.0079601, -0.0111738, 0.0110792
6: 0.9941816, 1.0027502, 0.9943661, 1.0028161, -0.0086344, 0.0083840
7: 0.0062882, 0.0192841, 0.0075085, 0.0193220, -0.0097213, 0.0088715
8: 0.0024734, 0.0070299, 0.0024438, 0.0070418, -0.0045684, 0.0045861
9: -0.0261387, -0.0132337, -0.0262059, -0.0139967, -0.0121420, 0.0129722

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069649, upper bound: 0.0071793
time: 1.14 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069649, upper bound: 0.0071793
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0045850, 0.0085461, 0.0043205, 0.0085704, -0.0039854, 0.0042255
1: 0.0005328, 0.0046350, 0.0002809, 0.0046822, -0.0041494, 0.0043541
2: -0.0238867, -0.0043660, -0.0245171, -0.0052908, -0.0185958, 0.0201511
3: -0.0021226, 0.0063524, -0.0020400, 0.0068996, -0.0090222, 0.0083924
4: 0.0110376, 0.0180108, 0.0114384, 0.0181756, -0.0071381, 0.0065724
5: -0.0032084, 0.0076963, -0.0032330, 0.0084667, -0.0116751, 0.0109292
6: 0.9942283, 1.0026382, 0.9943380, 1.0031573, -0.0089290, 0.0083002
7: 0.0065971, 0.0192197, 0.0073226, 0.0195182, -0.0096169, 0.0087891
8: 0.0025238, 0.0070098, 0.0022903, 0.0071033, -0.0045795, 0.0047195
9: -0.0260245, -0.0134268, -0.0265542, -0.0138805, -0.0121440, 0.0131273

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073278, upper bound: 0.0074334
time: 1.19 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073278, upper bound: 0.0074334
time: 1.29 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0046383, 0.0085411, 0.0043848, 0.0085645, -0.0039262, 0.0041564
1: 0.0005836, 0.0046255, 0.0003421, 0.0046707, -0.0040871, 0.0042834
2: -0.0237595, -0.0037837, -0.0243640, -0.0053473, -0.0184121, 0.0205803
3: -0.0021746, 0.0062421, -0.0020350, 0.0067667, -0.0089413, 0.0082771
4: 0.0107853, 0.0179775, 0.0114628, 0.0181356, -0.0073503, 0.0065147
5: -0.0032034, 0.0075409, -0.0032270, 0.0082796, -0.0114830, 0.0107679
6: 0.9941592, 1.0025334, 0.9943447, 1.0030313, -0.0088721, 0.0081887
7: 0.0061403, 0.0191595, 0.0073669, 0.0194457, -0.0097975, 0.0088147
8: 0.0025709, 0.0069909, 0.0023470, 0.0070806, -0.0045097, 0.0046439
9: -0.0259177, -0.0131412, -0.0264255, -0.0139082, -0.0120095, 0.0132843

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073278, upper bound: 0.0074334
time: 1.36 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073278, upper bound: 0.0074334
time: 1.33 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0045950, 0.0085451, 0.0043127, 0.0085711, -0.0039762, 0.0042325
1: 0.0005423, 0.0046332, 0.0002734, 0.0046836, -0.0041413, 0.0043598
2: -0.0238629, -0.0043812, -0.0245359, -0.0050859, -0.0187770, 0.0201547
3: -0.0021213, 0.0063318, -0.0020583, 0.0069158, -0.0090371, 0.0083901
4: 0.0110442, 0.0180046, 0.0113496, 0.0181806, -0.0071364, 0.0066550
5: -0.0032074, 0.0076672, -0.0032337, 0.0084896, -0.0116970, 0.0109009
6: 0.9942302, 1.0026186, 0.9943137, 1.0031729, -0.0089427, 0.0083049
7: 0.0066090, 0.0192085, 0.0071618, 0.0195271, -0.0096220, 0.0089718
8: 0.0025326, 0.0070062, 0.0022833, 0.0071060, -0.0045734, 0.0047229
9: -0.0260045, -0.0134343, -0.0265699, -0.0137799, -0.0122246, 0.0131356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074025, upper bound: 0.0074521
time: 1.43 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074025, upper bound: 0.0074521
time: 1.23 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0046482, 0.0085402, 0.0043774, 0.0085652, -0.0039170, 0.0041628
1: 0.0005930, 0.0046238, 0.0003351, 0.0046721, -0.0040790, 0.0042887
2: -0.0237360, -0.0038002, -0.0243816, -0.0051417, -0.0185943, 0.0205813
3: -0.0021732, 0.0062217, -0.0020533, 0.0067819, -0.0089550, 0.0082750
4: 0.0107924, 0.0179714, 0.0113737, 0.0181402, -0.0073478, 0.0065976
5: -0.0032025, 0.0075122, -0.0032277, 0.0083010, -0.0115035, 0.0107399
6: 0.9941612, 1.0025142, 0.9943203, 1.0030458, -0.0088846, 0.0081939
7: 0.0061533, 0.0191484, 0.0072056, 0.0194540, -0.0097998, 0.0089983
8: 0.0025796, 0.0069874, 0.0023404, 0.0070832, -0.0045035, 0.0046470
9: -0.0258980, -0.0131493, -0.0264402, -0.0138073, -0.0120906, 0.0132909

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 187

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074025, upper bound: 0.0074521
time: 1.56 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074025, upper bound: 0.0074521
time: 1.55 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0044182, 0.0085614, 0.0043267, 0.0085698, -0.0041517, 0.0042347
1: 0.0003739, 0.0046648, 0.0002868, 0.0046811, -0.0043072, 0.0043780
2: -0.0242844, -0.0045198, -0.0245024, -0.0054400, -0.0188444, 0.0199826
3: -0.0021089, 0.0066976, -0.0020267, 0.0068867, -0.0089956, 0.0087243
4: 0.0111042, 0.0181148, 0.0115030, 0.0181718, -0.0070676, 0.0066118
5: -0.0032239, 0.0081824, -0.0032324, 0.0084487, -0.0116726, 0.0114148
6: 0.9942465, 1.0029657, 0.9943557, 1.0031452, -0.0088987, 0.0086100
7: 0.0067177, 0.0194080, 0.0074396, 0.0195112, -0.0094662, 0.0088595
8: 0.0023765, 0.0070688, 0.0022957, 0.0071011, -0.0047246, 0.0047731
9: -0.0263587, -0.0135023, -0.0265418, -0.0139536, -0.0124050, 0.0130395

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073130, upper bound: 0.0073714
time: 1.30 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073130, upper bound: 0.0073714
time: 1.26 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0044651, 0.0085571, 0.0043910, 0.0085639, -0.0040989, 0.0041661
1: 0.0004186, 0.0046564, 0.0003480, 0.0046696, -0.0042511, 0.0043084
2: -0.0241726, -0.0039316, -0.0243493, -0.0054973, -0.0186753, 0.0204177
3: -0.0021614, 0.0066006, -0.0020216, 0.0067539, -0.0089153, 0.0086221
4: 0.0108494, 0.0180855, 0.0115278, 0.0181317, -0.0072824, 0.0065577
5: -0.0032195, 0.0080457, -0.0032264, 0.0082616, -0.0114811, 0.0112721
6: 0.9941767, 1.0028735, 0.9943625, 1.0030191, -0.0088423, 0.0085111
7: 0.0062564, 0.0193551, 0.0074845, 0.0194387, -0.0097689, 0.0088860
8: 0.0024179, 0.0070522, 0.0023524, 0.0070784, -0.0046605, 0.0046998
9: -0.0262647, -0.0132138, -0.0264131, -0.0139817, -0.0122830, 0.0131993

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073130, upper bound: 0.0073714
time: 1.42 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073130, upper bound: 0.0073714
time: 1.26 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0044344, 0.0085599, 0.0043185, 0.0085706, -0.0041361, 0.0042414
1: 0.0003894, 0.0046619, 0.0002790, 0.0046826, -0.0042932, 0.0043829
2: -0.0242456, -0.0045323, -0.0245220, -0.0052452, -0.0190004, 0.0199896
3: -0.0021078, 0.0066639, -0.0020441, 0.0069037, -0.0090115, 0.0087080
4: 0.0111097, 0.0181046, 0.0114186, 0.0181769, -0.0070672, 0.0066861
5: -0.0032224, 0.0081349, -0.0032332, 0.0084726, -0.0116949, 0.0113680
6: 0.9942480, 1.0029337, 0.9943326, 1.0031613, -0.0089133, 0.0086012
7: 0.0067276, 0.0193896, 0.0072867, 0.0195205, -0.0094689, 0.0090028
8: 0.0023908, 0.0070630, 0.0022885, 0.0071040, -0.0047131, 0.0047745
9: -0.0263260, -0.0135084, -0.0265582, -0.0138581, -0.0124679, 0.0130498

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073771, upper bound: 0.0073771
time: 1.20 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073771, upper bound: 0.0073771
time: 1.21 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0044835, 0.0085554, 0.0043834, 0.0085646, -0.0040811, 0.0041720
1: 0.0004362, 0.0046531, 0.0003408, 0.0046710, -0.0042348, 0.0043123
2: -0.0241286, -0.0039451, -0.0243673, -0.0053018, -0.0188268, 0.0204222
3: -0.0021602, 0.0065624, -0.0020390, 0.0067695, -0.0089297, 0.0086014
4: 0.0108552, 0.0180740, 0.0114431, 0.0181365, -0.0072813, 0.0066309
5: -0.0032178, 0.0079919, -0.0032271, 0.0082836, -0.0115014, 0.0112190
6: 0.9941784, 1.0028374, 0.9943393, 1.0030339, -0.0088555, 0.0084981
7: 0.0062669, 0.0193343, 0.0073311, 0.0194473, -0.0097711, 0.0090299
8: 0.0024342, 0.0070456, 0.0023458, 0.0070810, -0.0046469, 0.0046999
9: -0.0262277, -0.0132204, -0.0264283, -0.0138858, -0.0123419, 0.0132079

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073771, upper bound: 0.0073771
time: 1.37 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073771, upper bound: 0.0073771
time: 1.28 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 6.37 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0079810, upper bound: 0.0078608
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0079860, upper bound: 0.0078360
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0079960, upper bound: 0.0079142
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0080013, upper bound: 0.0078948
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0079684, upper bound: 0.0078558
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0079721, upper bound: 0.0078306
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0079886, upper bound: 0.0079026
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0079910, upper bound: 0.0078771
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0080805, upper bound: 0.0080356
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0080748, upper bound: 0.0080080
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0081092, upper bound: 0.0081343
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0081004, upper bound: 0.0081087
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0080742, upper bound: 0.0080284
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0080637, upper bound: 0.0080001
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0081036, upper bound: 0.0081218
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0080928, upper bound: 0.0080928
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0073953, upper bound: 0.0073470
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0073729, upper bound: 0.0072739
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0073966, upper bound: 0.0074045
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0073740, upper bound: 0.0073378
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0073190, upper bound: 0.0072996
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0072817, upper bound: 0.0072143
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0073246, upper bound: 0.0073511
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0072830, upper bound: 0.0072559
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0075664, upper bound: 0.0076449
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0075277, upper bound: 0.0075670
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0075890, upper bound: 0.0077266
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0075391, upper bound: 0.0076634
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0075044, upper bound: 0.0076126
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0074200, upper bound: 0.0075126
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0075223, upper bound: 0.0076805
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0074257, upper bound: 0.0075914
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0069409, upper bound: 0.0072175
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0069409, upper bound: 0.0072175
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0069409, upper bound: 0.0072175
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0069409, upper bound: 0.0072175
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0069912, upper bound: 0.0072175
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0069912, upper bound: 0.0072175
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0069912, upper bound: 0.0072175
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0069912, upper bound: 0.0072175
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0069282, upper bound: 0.0071775
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0069282, upper bound: 0.0071775
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0069282, upper bound: 0.0071775
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0069282, upper bound: 0.0071775
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0069649, upper bound: 0.0071793
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0069649, upper bound: 0.0071793
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0069649, upper bound: 0.0071793
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0069649, upper bound: 0.0071793
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0073278, upper bound: 0.0074334
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0073278, upper bound: 0.0074334
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0073278, upper bound: 0.0074334
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0073278, upper bound: 0.0074334
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0074025, upper bound: 0.0074521
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0074025, upper bound: 0.0074521
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0074025, upper bound: 0.0074521
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0074025, upper bound: 0.0074521
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0073130, upper bound: 0.0073714
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0073130, upper bound: 0.0073714
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0073130, upper bound: 0.0073714
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0073130, upper bound: 0.0073714
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0073771, upper bound: 0.0073771
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0073771, upper bound: 0.0073771
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0073771, upper bound: 0.0073771
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 6, lower bound: -0.0073771, upper bound: 0.0073771

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0044049, 0.0085626, 0.0042683, 0.0085752, -0.0041703, 0.0042943
1: 0.0003613, 0.0046672, 0.0002311, 0.0046915, -0.0043303, 0.0044360
2: -0.0243160, -0.0054964, -0.0246417, -0.0054132, -0.0189028, 0.0191453
3: -0.0020217, 0.0067250, -0.0020291, 0.0070076, -0.0090293, 0.0087541
4: 0.0115274, 0.0181231, 0.0114914, 0.0182082, -0.0066808, 0.0066317
5: -0.0032251, 0.0082210, -0.0032378, 0.0086190, -0.0118441, 0.0114588
6: 0.9943624, 1.0029918, 0.9943526, 1.0032599, -0.0088975, 0.0086392
7: 0.0074838, 0.0194230, 0.0074185, 0.0195772, -0.0089013, 0.0087667
8: 0.0023648, 0.0070734, 0.0022441, 0.0071217, -0.0047570, 0.0048294
9: -0.0263852, -0.0139813, -0.0266588, -0.0139405, -0.0124447, 0.0126775

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074833, upper bound: 0.0074216
time: 1.41 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072301, upper bound: 0.0070438
time: 1.12 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0044751, 0.0085562, 0.0043319, 0.0085694, -0.0040943, 0.0042243
1: 0.0004282, 0.0046546, 0.0002917, 0.0046802, -0.0042520, 0.0043630
2: -0.0241486, -0.0055578, -0.0244902, -0.0048507, -0.0192979, 0.0189324
3: -0.0020162, 0.0065798, -0.0020793, 0.0068762, -0.0088923, 0.0086591
4: 0.0115541, 0.0180793, 0.0112476, 0.0181686, -0.0066146, 0.0068317
5: -0.0032186, 0.0080164, -0.0032319, 0.0084338, -0.0116524, 0.0112483
6: 0.9943697, 1.0028540, 0.9942858, 1.0031351, -0.0087654, 0.0085682
7: 0.0075320, 0.0193438, 0.0069773, 0.0195055, -0.0089275, 0.0090809
8: 0.0024268, 0.0070486, 0.0023002, 0.0070993, -0.0046725, 0.0047484
9: -0.0262446, -0.0140114, -0.0265315, -0.0136646, -0.0125800, 0.0125201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073742, upper bound: 0.0072419
time: 1.31 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071735, upper bound: 0.0069678
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0043919, 0.0085638, 0.0042734, 0.0085747, -0.0041829, 0.0042904
1: 0.0003488, 0.0046695, 0.0002360, 0.0046906, -0.0043418, 0.0044334
2: -0.0243471, -0.0052912, -0.0246294, -0.0054299, -0.0189172, 0.0193382
3: -0.0020400, 0.0067520, -0.0020276, 0.0069970, -0.0090370, 0.0087796
4: 0.0114385, 0.0181312, 0.0114986, 0.0182050, -0.0067665, 0.0066326
5: -0.0032263, 0.0082590, -0.0032374, 0.0086039, -0.0118303, 0.0114963
6: 0.9943380, 1.0030174, 0.9943545, 1.0032499, -0.0089118, 0.0086629
7: 0.0073229, 0.0194377, 0.0074316, 0.0195714, -0.0090907, 0.0087685
8: 0.0023532, 0.0070780, 0.0022487, 0.0071199, -0.0047667, 0.0048294
9: -0.0264113, -0.0138807, -0.0266485, -0.0139487, -0.0124627, 0.0127678

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074994, upper bound: 0.0074670
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072523, upper bound: 0.0070825
time: 1.41 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0044615, 0.0085574, 0.0043396, 0.0085687, -0.0041071, 0.0042178
1: 0.0004152, 0.0046571, 0.0002990, 0.0046788, -0.0042636, 0.0043580
2: -0.0241811, -0.0053523, -0.0244718, -0.0048675, -0.0193136, 0.0191195
3: -0.0020345, 0.0066079, -0.0020778, 0.0068602, -0.0088947, 0.0086857
4: 0.0114650, 0.0180878, 0.0112549, 0.0181638, -0.0066988, 0.0068329
5: -0.0032199, 0.0080560, -0.0032312, 0.0084113, -0.0116312, 0.0112872
6: 0.9943453, 1.0028806, 0.9942878, 1.0031199, -0.0087746, 0.0085928
7: 0.0073708, 0.0193591, 0.0069905, 0.0194967, -0.0091188, 0.0090835
8: 0.0024147, 0.0070534, 0.0023071, 0.0070965, -0.0046818, 0.0047464
9: -0.0262718, -0.0139106, -0.0265161, -0.0136728, -0.0125990, 0.0126054

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073822, upper bound: 0.0072911
time: 1.29 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071935, upper bound: 0.0070053
time: 1.12 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0044113, 0.0085620, 0.0041135, 0.0085895, -0.0041781, 0.0044485
1: 0.0003674, 0.0046660, 0.0000837, 0.0047191, -0.0043517, 0.0045823
2: -0.0243007, -0.0056394, -0.0250106, -0.0055885, -0.0187122, 0.0193713
3: -0.0020089, 0.0067117, -0.0020134, 0.0073278, -0.0093367, 0.0087252
4: 0.0115894, 0.0181190, 0.0115673, 0.0183047, -0.0067153, 0.0065517
5: -0.0032245, 0.0082022, -0.0032522, 0.0090698, -0.0122943, 0.0114545
6: 0.9943793, 1.0029790, 0.9943733, 1.0035635, -0.0091842, 0.0086058
7: 0.0075960, 0.0194157, 0.0075560, 0.0197518, -0.0089366, 0.0087343
8: 0.0023704, 0.0070712, 0.0021074, 0.0071765, -0.0048061, 0.0049638
9: -0.0263723, -0.0140514, -0.0269688, -0.0140264, -0.0123459, 0.0129173

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074781, upper bound: 0.0074159
time: 1.44 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072206, upper bound: 0.0070357
time: 1.28 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0044816, 0.0085556, 0.0041779, 0.0085835, -0.0041019, 0.0043777
1: 0.0004344, 0.0046535, 0.0001450, 0.0047076, -0.0042733, 0.0045085
2: -0.0241331, -0.0057014, -0.0248573, -0.0050179, -0.0191152, 0.0191559
3: -0.0020033, 0.0065663, -0.0020644, 0.0071947, -0.0091981, 0.0086307
4: 0.0116163, 0.0180752, 0.0113201, 0.0182646, -0.0066483, 0.0067551
5: -0.0032180, 0.0079974, -0.0032463, 0.0088824, -0.0121003, 0.0112437
6: 0.9943866, 1.0028410, 0.9943056, 1.0034373, -0.0090507, 0.0085354
7: 0.0076446, 0.0193364, 0.0071084, 0.0196792, -0.0088730, 0.0089173
8: 0.0024325, 0.0070463, 0.0021642, 0.0071537, -0.0047212, 0.0048821
9: -0.0262315, -0.0140819, -0.0268399, -0.0137466, -0.0124849, 0.0127580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073136, upper bound: 0.0072218
time: 1.22 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071312, upper bound: 0.0069565
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0043977, 0.0085633, 0.0041266, 0.0085883, -0.0041906, 0.0044367
1: 0.0003543, 0.0046684, 0.0000961, 0.0047168, -0.0043624, 0.0045723
2: -0.0243333, -0.0054453, -0.0249796, -0.0056012, -0.0187321, 0.0195343
3: -0.0020262, 0.0067400, -0.0020123, 0.0073008, -0.0093270, 0.0087523
4: 0.0115053, 0.0181276, 0.0115728, 0.0182966, -0.0067913, 0.0065547
5: -0.0032258, 0.0082421, -0.0032510, 0.0090318, -0.0122576, 0.0114931
6: 0.9943563, 1.0030060, 0.9943748, 1.0035380, -0.0091817, 0.0086312
7: 0.0074437, 0.0194312, 0.0075660, 0.0197371, -0.0090816, 0.0087362
8: 0.0023583, 0.0070760, 0.0021189, 0.0071718, -0.0048135, 0.0049571
9: -0.0263997, -0.0139562, -0.0269426, -0.0140327, -0.0123670, 0.0129864

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074958, upper bound: 0.0074616
time: 1.47 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072463, upper bound: 0.0070743
time: 1.29 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0044673, 0.0085569, 0.0041919, 0.0085823, -0.0041150, 0.0043650
1: 0.0004207, 0.0046560, 0.0001584, 0.0047051, -0.0042845, 0.0044976
2: -0.0241674, -0.0055075, -0.0248237, -0.0050309, -0.0191364, 0.0193162
3: -0.0020207, 0.0065960, -0.0020632, 0.0071656, -0.0091863, 0.0086592
4: 0.0115322, 0.0180842, 0.0113257, 0.0182558, -0.0067236, 0.0067584
5: -0.0032193, 0.0080393, -0.0032449, 0.0088414, -0.0120607, 0.0112842
6: 0.9943637, 1.0028694, 0.9943072, 1.0034097, -0.0090461, 0.0085622
7: 0.0074925, 0.0193526, 0.0071187, 0.0196633, -0.0090212, 0.0089177
8: 0.0024199, 0.0070514, 0.0021766, 0.0071487, -0.0047289, 0.0048748
9: -0.0262603, -0.0139867, -0.0268117, -0.0137530, -0.0125073, 0.0128250

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073255, upper bound: 0.0072685
time: 1.31 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071560, upper bound: 0.0069914
time: 1.08 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0042933, 0.0085729, 0.0042290, 0.0085788, -0.0042855, 0.0043439
1: 0.0002549, 0.0046871, 0.0001937, 0.0046985, -0.0044436, 0.0044933
2: -0.0245822, -0.0052706, -0.0247353, -0.0053873, -0.0191948, 0.0194647
3: -0.0020418, 0.0069559, -0.0020314, 0.0070889, -0.0091307, 0.0089873
4: 0.0114296, 0.0181926, 0.0114802, 0.0182327, -0.0068031, 0.0067125
5: -0.0032355, 0.0085461, -0.0032415, 0.0087333, -0.0119688, 0.0117876
6: 0.9943356, 1.0032108, 0.9943495, 1.0033370, -0.0090014, 0.0088613
7: 0.0073067, 0.0195490, 0.0073982, 0.0196215, -0.0090684, 0.0088262
8: 0.0022662, 0.0071129, 0.0022094, 0.0071356, -0.0048695, 0.0049035
9: -0.0266088, -0.0138705, -0.0267374, -0.0139278, -0.0126810, 0.0128669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0076298, upper bound: 0.0076116
time: 1.35 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074235, upper bound: 0.0073082
time: 1.20 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0043589, 0.0085669, 0.0042884, 0.0085734, -0.0042145, 0.0042785
1: 0.0003175, 0.0046754, 0.0002503, 0.0046879, -0.0043705, 0.0044251
2: -0.0244257, -0.0053273, -0.0245937, -0.0048227, -0.0196029, 0.0192664
3: -0.0020368, 0.0068202, -0.0020818, 0.0069660, -0.0090028, 0.0089020
4: 0.0114542, 0.0181517, 0.0112355, 0.0181957, -0.0067415, 0.0069162
5: -0.0032294, 0.0083550, -0.0032360, 0.0085604, -0.0117898, 0.0115909
6: 0.9943424, 1.0030819, 0.9942825, 1.0032204, -0.0088781, 0.0087994
7: 0.0073512, 0.0194749, 0.0069554, 0.0195545, -0.0090982, 0.0091193
8: 0.0023241, 0.0070897, 0.0022618, 0.0071146, -0.0047905, 0.0048279
9: -0.0264773, -0.0138984, -0.0266185, -0.0136509, -0.0128264, 0.0127202

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074885, upper bound: 0.0074378
time: 1.26 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073592, upper bound: 0.0072306
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0042808, 0.0085741, 0.0042338, 0.0085784, -0.0042976, 0.0043403
1: 0.0002430, 0.0046893, 0.0001983, 0.0046977, -0.0044546, 0.0044910
2: -0.0246120, -0.0050661, -0.0247239, -0.0054035, -0.0192085, 0.0196578
3: -0.0020601, 0.0069818, -0.0020300, 0.0070789, -0.0091390, 0.0090118
4: 0.0113410, 0.0182004, 0.0114872, 0.0182297, -0.0068887, 0.0067133
5: -0.0032367, 0.0085826, -0.0032411, 0.0087193, -0.0119560, 0.0118237
6: 0.9943113, 1.0032353, 0.9943513, 1.0033275, -0.0090162, 0.0088840
7: 0.0071463, 0.0195631, 0.0074110, 0.0196161, -0.0092589, 0.0088288
8: 0.0022551, 0.0071173, 0.0022137, 0.0071339, -0.0048788, 0.0049037
9: -0.0266338, -0.0137702, -0.0267278, -0.0139357, -0.0126981, 0.0129576

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0076798, upper bound: 0.0077342
time: 1.68 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0075026, upper bound: 0.0074429
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0043476, 0.0085679, 0.0042959, 0.0085727, -0.0042251, 0.0042720
1: 0.0003066, 0.0046774, 0.0002575, 0.0046866, -0.0043799, 0.0044199
2: -0.0244527, -0.0051220, -0.0245759, -0.0048393, -0.0196134, 0.0194539
3: -0.0020551, 0.0068436, -0.0020803, 0.0069505, -0.0090056, 0.0089240
4: 0.0113652, 0.0181588, 0.0112427, 0.0181910, -0.0068258, 0.0069161
5: -0.0032305, 0.0083880, -0.0032353, 0.0085384, -0.0117689, 0.0116233
6: 0.9943179, 1.0031043, 0.9942845, 1.0032057, -0.0088877, 0.0088199
7: 0.0071901, 0.0194877, 0.0069684, 0.0195460, -0.0092902, 0.0091228
8: 0.0023141, 0.0070937, 0.0022685, 0.0071120, -0.0047979, 0.0048252
9: -0.0265000, -0.0137976, -0.0266035, -0.0136590, -0.0128410, 0.0128058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0075090, upper bound: 0.0075667
time: 1.27 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074128, upper bound: 0.0073696
time: 1.24 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0042994, 0.0085724, 0.0040749, 0.0085930, -0.0042937, 0.0044975
1: 0.0002607, 0.0046860, 0.0000468, 0.0047260, -0.0044653, 0.0046391
2: -0.0245676, -0.0054196, -0.0251029, -0.0055616, -0.0190060, 0.0196833
3: -0.0020285, 0.0069433, -0.0020158, 0.0074079, -0.0094364, 0.0089592
4: 0.0114942, 0.0181888, 0.0115557, 0.0183288, -0.0068346, 0.0066332
5: -0.0032350, 0.0085284, -0.0032558, 0.0091825, -0.0124175, 0.0117842
6: 0.9943532, 1.0031989, 0.9943702, 1.0036397, -0.0092865, 0.0088287
7: 0.0074236, 0.0195421, 0.0075349, 0.0197955, -0.0090995, 0.0087941
8: 0.0022715, 0.0071107, 0.0020732, 0.0071901, -0.0049186, 0.0050375
9: -0.0265966, -0.0139436, -0.0270462, -0.0140133, -0.0125833, 0.0131026

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0076226, upper bound: 0.0076048
time: 1.46 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074124, upper bound: 0.0073021
time: 1.25 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0043650, 0.0085663, 0.0041369, 0.0085873, -0.0042223, 0.0044294
1: 0.0003232, 0.0046743, 0.0001059, 0.0047150, -0.0043917, 0.0045683
2: -0.0244112, -0.0054771, -0.0249550, -0.0049899, -0.0194213, 0.0194779
3: -0.0020234, 0.0068076, -0.0020669, 0.0072795, -0.0093029, 0.0088745
4: 0.0115191, 0.0181479, 0.0113079, 0.0182902, -0.0067711, 0.0068400
5: -0.0032288, 0.0083372, -0.0032501, 0.0090018, -0.0122306, 0.0115873
6: 0.9943601, 1.0030701, 0.9943024, 1.0035179, -0.0091577, 0.0087677
7: 0.0074687, 0.0194681, 0.0070865, 0.0197255, -0.0090265, 0.0089540
8: 0.0023295, 0.0070876, 0.0021281, 0.0071682, -0.0048387, 0.0049595
9: -0.0264652, -0.0139718, -0.0269220, -0.0137328, -0.0127323, 0.0129502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074182, upper bound: 0.0074100
time: 1.61 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073176, upper bound: 0.0072199
time: 1.24 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0042864, 0.0085736, 0.0040877, 0.0085919, -0.0043055, 0.0044858
1: 0.0002484, 0.0046883, 0.0000591, 0.0047237, -0.0044753, 0.0046292
2: -0.0245986, -0.0052252, -0.0250722, -0.0055740, -0.0190246, 0.0198469
3: -0.0020459, 0.0069702, -0.0020147, 0.0073812, -0.0094271, 0.0089849
4: 0.0114099, 0.0181969, 0.0115611, 0.0183208, -0.0069109, 0.0066359
5: -0.0032362, 0.0085662, -0.0032546, 0.0091450, -0.0123811, 0.0118209
6: 0.9943302, 1.0032244, 0.9943715, 1.0036143, -0.0092841, 0.0088528
7: 0.0072711, 0.0195567, 0.0075447, 0.0197809, -0.0092466, 0.0087963
8: 0.0022601, 0.0071153, 0.0020846, 0.0071856, -0.0049255, 0.0050307
9: -0.0266225, -0.0138483, -0.0270204, -0.0140193, -0.0126032, 0.0131721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0076739, upper bound: 0.0077275
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074954, upper bound: 0.0074339
time: 1.23 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0043532, 0.0085674, 0.0041513, 0.0085860, -0.0042328, 0.0044161
1: 0.0003120, 0.0046764, 0.0001197, 0.0047124, -0.0044003, 0.0045567
2: -0.0244392, -0.0052817, -0.0249205, -0.0050028, -0.0194365, 0.0196388
3: -0.0020408, 0.0068319, -0.0020657, 0.0072496, -0.0092904, 0.0088977
4: 0.0114344, 0.0181553, 0.0113135, 0.0182811, -0.0068467, 0.0068417
5: -0.0032299, 0.0083715, -0.0032487, 0.0089597, -0.0121896, 0.0116203
6: 0.9943370, 1.0030931, 0.9943039, 1.0034895, -0.0091525, 0.0087892
7: 0.0073154, 0.0194813, 0.0070966, 0.0197092, -0.0091766, 0.0089568
8: 0.0023191, 0.0070917, 0.0021408, 0.0071631, -0.0048440, 0.0049509
9: -0.0264887, -0.0138760, -0.0268930, -0.0137392, -0.0127495, 0.0130170

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 90

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074398, upper bound: 0.0075394
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073544, upper bound: 0.0073544
time: 1.39 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0044049, 0.0085626, 0.0046254, 0.0085423, -0.0041374, 0.0039372
1: 0.0003613, 0.0046672, 0.0005714, 0.0046278, -0.0042665, 0.0040958
2: -0.0243160, -0.0054964, -0.0237902, -0.0043914, -0.0199247, 0.0182939
3: -0.0020217, 0.0067250, -0.0021204, 0.0062687, -0.0082904, 0.0088454
4: 0.0115274, 0.0181231, 0.0110486, 0.0179856, -0.0064581, 0.0070745
5: -0.0032251, 0.0082210, -0.0032046, 0.0075784, -0.0108035, 0.0114256
6: 0.9943624, 1.0029918, 0.9942313, 1.0025587, -0.0081964, 0.0087605
7: 0.0074838, 0.0194230, 0.0066170, 0.0191741, -0.0084912, 0.0095918
8: 0.0023648, 0.0070734, 0.0025595, 0.0069955, -0.0046307, 0.0045139
9: -0.0263852, -0.0139813, -0.0259435, -0.0134393, -0.0129459, 0.0119622

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073729, upper bound: 0.0072739
time: 1.35 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073729, upper bound: 0.0072739
time: 1.30 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0044751, 0.0085562, 0.0046815, 0.0085372, -0.0040621, 0.0038747
1: 0.0004282, 0.0046546, 0.0006247, 0.0046178, -0.0041897, 0.0040299
2: -0.0241486, -0.0055578, -0.0236566, -0.0038115, -0.0203371, 0.0180988
3: -0.0020162, 0.0065798, -0.0021721, 0.0061528, -0.0081690, 0.0087519
4: 0.0115541, 0.0180793, 0.0107973, 0.0179506, -0.0063966, 0.0072820
5: -0.0032186, 0.0080164, -0.0031994, 0.0074152, -0.0106337, 0.0112158
6: 0.9943697, 1.0028540, 0.9941625, 1.0024488, -0.0080791, 0.0086915
7: 0.0075320, 0.0193438, 0.0061621, 0.0191109, -0.0085082, 0.0097828
8: 0.0024268, 0.0070486, 0.0026091, 0.0069756, -0.0045489, 0.0044395
9: -0.0262446, -0.0140114, -0.0258312, -0.0131549, -0.0130897, 0.0118198

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073729, upper bound: 0.0072739
time: 1.74 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073729, upper bound: 0.0072739
time: 1.53 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0043919, 0.0085638, 0.0046367, 0.0085413, -0.0041494, 0.0039272
1: 0.0003488, 0.0046695, 0.0005821, 0.0046258, -0.0042770, 0.0040874
2: -0.0243471, -0.0052912, -0.0237634, -0.0044073, -0.0199399, 0.0184722
3: -0.0020400, 0.0067520, -0.0021189, 0.0062455, -0.0082855, 0.0088709
4: 0.0114385, 0.0181312, 0.0110555, 0.0179785, -0.0065400, 0.0070757
5: -0.0032263, 0.0082590, -0.0032036, 0.0075457, -0.0107720, 0.0114625
6: 0.9943380, 1.0030174, 0.9942332, 1.0025368, -0.0081987, 0.0087842
7: 0.0073229, 0.0194377, 0.0066295, 0.0191614, -0.0086664, 0.0095945
8: 0.0023532, 0.0070780, 0.0025694, 0.0069915, -0.0046382, 0.0045086
9: -0.0264113, -0.0138807, -0.0259210, -0.0134471, -0.0129642, 0.0120403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073740, upper bound: 0.0073378
time: 1.57 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073740, upper bound: 0.0073378
time: 1.87 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0044615, 0.0085574, 0.0046925, 0.0085362, -0.0040746, 0.0038649
1: 0.0004152, 0.0046571, 0.0006353, 0.0046158, -0.0042007, 0.0040218
2: -0.0241811, -0.0053523, -0.0236303, -0.0038287, -0.0203524, 0.0182780
3: -0.0020345, 0.0066079, -0.0021706, 0.0061300, -0.0081645, 0.0087785
4: 0.0114650, 0.0180878, 0.0108047, 0.0179437, -0.0064787, 0.0072830
5: -0.0032199, 0.0080560, -0.0031984, 0.0073830, -0.0106028, 0.0112544
6: 0.9943453, 1.0028806, 0.9941646, 1.0024272, -0.0080819, 0.0087160
7: 0.0073708, 0.0193591, 0.0061756, 0.0190984, -0.0086844, 0.0097834
8: 0.0024147, 0.0070534, 0.0026188, 0.0069717, -0.0045570, 0.0044346
9: -0.0262718, -0.0139106, -0.0258091, -0.0131633, -0.0131085, 0.0118985

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073740, upper bound: 0.0073378
time: 1.51 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073740, upper bound: 0.0073378
time: 1.42 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0044113, 0.0085620, 0.0044637, 0.0085572, -0.0041459, 0.0040984
1: 0.0003674, 0.0046660, 0.0004173, 0.0046567, -0.0042893, 0.0042487
2: -0.0243007, -0.0056394, -0.0241759, -0.0045456, -0.0197551, 0.0185365
3: -0.0020089, 0.0067117, -0.0021066, 0.0066034, -0.0086123, 0.0088183
4: 0.0115894, 0.0181190, 0.0111154, 0.0180864, -0.0064970, 0.0070036
5: -0.0032245, 0.0082022, -0.0032197, 0.0080497, -0.0112742, 0.0114219
6: 0.9943793, 1.0029790, 0.9942496, 1.0028764, -0.0084971, 0.0087295
7: 0.0075960, 0.0194157, 0.0067380, 0.0193566, -0.0086042, 0.0094312
8: 0.0023704, 0.0070712, 0.0024167, 0.0070526, -0.0046822, 0.0046545
9: -0.0263723, -0.0140514, -0.0262674, -0.0135149, -0.0128574, 0.0122160

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072817, upper bound: 0.0072143
time: 1.37 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072817, upper bound: 0.0072143
time: 1.40 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0044816, 0.0085556, 0.0045110, 0.0085529, -0.0040713, 0.0040446
1: 0.0004344, 0.0046535, 0.0004623, 0.0046482, -0.0042139, 0.0041912
2: -0.0241331, -0.0057014, -0.0240632, -0.0039587, -0.0201744, 0.0183618
3: -0.0020033, 0.0065663, -0.0021590, 0.0065056, -0.0085090, 0.0087253
4: 0.0116163, 0.0180752, 0.0108611, 0.0180569, -0.0064407, 0.0072141
5: -0.0032180, 0.0079974, -0.0032153, 0.0079120, -0.0111300, 0.0112127
6: 0.9943866, 1.0028410, 0.9941800, 1.0027835, -0.0083969, 0.0086610
7: 0.0076446, 0.0193364, 0.0062776, 0.0193033, -0.0086204, 0.0097535
8: 0.0024325, 0.0070463, 0.0024584, 0.0070359, -0.0046034, 0.0045879
9: -0.0262315, -0.0140819, -0.0261728, -0.0132271, -0.0130045, 0.0120909

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 165

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072817, upper bound: 0.0072143
time: 1.49 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072817, upper bound: 0.0072143
time: 1.88 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0043977, 0.0085633, 0.0044781, 0.0085559, -0.0041582, 0.0040852
1: 0.0003543, 0.0046684, 0.0004310, 0.0046541, -0.0042997, 0.0042374
2: -0.0243333, -0.0054453, -0.0241415, -0.0045582, -0.0197751, 0.0186962
3: -0.0020262, 0.0067400, -0.0021055, 0.0065736, -0.0085998, 0.0088455
4: 0.0115053, 0.0181276, 0.0111209, 0.0180774, -0.0065721, 0.0070067
5: -0.0032258, 0.0082421, -0.0032183, 0.0080077, -0.0112335, 0.0114604
6: 0.9943563, 1.0030060, 0.9942511, 1.0028481, -0.0084918, 0.0087549
7: 0.0074437, 0.0194312, 0.0067478, 0.0193404, -0.0087443, 0.0094345
8: 0.0023583, 0.0070760, 0.0024294, 0.0070476, -0.0046893, 0.0046466
9: -0.0263997, -0.0139562, -0.0262386, -0.0135211, -0.0128786, 0.0122824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072830, upper bound: 0.0072559
time: 1.30 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072830, upper bound: 0.0072559
time: 1.39 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0044673, 0.0085569, 0.0045280, 0.0085513, -0.0040841, 0.0040289
1: 0.0004207, 0.0046560, 0.0004785, 0.0046452, -0.0042245, 0.0041775
2: -0.0241674, -0.0055075, -0.0240226, -0.0039723, -0.0201951, 0.0185151
3: -0.0020207, 0.0065960, -0.0021578, 0.0064704, -0.0084911, 0.0087538
4: 0.0115322, 0.0180842, 0.0108670, 0.0180463, -0.0065141, 0.0072172
5: -0.0032193, 0.0080393, -0.0032137, 0.0078624, -0.0110817, 0.0112530
6: 0.9943637, 1.0028694, 0.9941816, 1.0027502, -0.0083865, 0.0086877
7: 0.0074925, 0.0193526, 0.0062882, 0.0192841, -0.0087627, 0.0097544
8: 0.0024199, 0.0070514, 0.0024734, 0.0070299, -0.0046101, 0.0045780
9: -0.0262603, -0.0139867, -0.0261387, -0.0132337, -0.0130266, 0.0121520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 165

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072830, upper bound: 0.0072558
time: 1.59 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072830, upper bound: 0.0072558
time: 3.70 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0042933, 0.0085729, 0.0045850, 0.0085461, -0.0042527, 0.0039879
1: 0.0002549, 0.0046871, 0.0005328, 0.0046350, -0.0043801, 0.0041542
2: -0.0245822, -0.0052706, -0.0238867, -0.0043660, -0.0202162, 0.0186160
3: -0.0020418, 0.0069559, -0.0021226, 0.0063524, -0.0083942, 0.0090786
4: 0.0114296, 0.0181926, 0.0110376, 0.0180108, -0.0065812, 0.0071551
5: -0.0032355, 0.0085461, -0.0032084, 0.0076963, -0.0109318, 0.0117545
6: 0.9943356, 1.0032108, 0.9942283, 1.0026382, -0.0083026, 0.0089825
7: 0.0073067, 0.0195490, 0.0065971, 0.0192197, -0.0086455, 0.0096555
8: 0.0022662, 0.0071129, 0.0025238, 0.0070098, -0.0047436, 0.0045891
9: -0.0266088, -0.0138705, -0.0260245, -0.0134268, -0.0131819, 0.0121539

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0075277, upper bound: 0.0075670
time: 1.62 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0075277, upper bound: 0.0075670
time: 1.39 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0043589, 0.0085669, 0.0046383, 0.0085411, -0.0041823, 0.0039286
1: 0.0003175, 0.0046754, 0.0005836, 0.0046255, -0.0043081, 0.0040917
2: -0.0244257, -0.0053273, -0.0237595, -0.0037837, -0.0206420, 0.0184321
3: -0.0020368, 0.0068202, -0.0021746, 0.0062421, -0.0082788, 0.0089948
4: 0.0114542, 0.0181517, 0.0107853, 0.0179775, -0.0065233, 0.0073665
5: -0.0032294, 0.0083550, -0.0032034, 0.0075409, -0.0107703, 0.0115584
6: 0.9943424, 1.0030819, 0.9941592, 1.0025334, -0.0081910, 0.0089228
7: 0.0073512, 0.0194749, 0.0061403, 0.0191595, -0.0086675, 0.0098195
8: 0.0023241, 0.0070897, 0.0025709, 0.0069909, -0.0046668, 0.0045188
9: -0.0264773, -0.0138984, -0.0259177, -0.0131412, -0.0133361, 0.0120193

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0075277, upper bound: 0.0075670
time: 1.50 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0075277, upper bound: 0.0075670
time: 1.32 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0042808, 0.0085741, 0.0045950, 0.0085451, -0.0042644, 0.0039791
1: 0.0002430, 0.0046893, 0.0005423, 0.0046332, -0.0043902, 0.0041470
2: -0.0246120, -0.0050661, -0.0238629, -0.0043812, -0.0202307, 0.0187968
3: -0.0020601, 0.0069818, -0.0021213, 0.0063318, -0.0083919, 0.0091031
4: 0.0113410, 0.0182004, 0.0110442, 0.0180046, -0.0066636, 0.0071562
5: -0.0032367, 0.0085826, -0.0032074, 0.0076672, -0.0109039, 0.0117900
6: 0.9943113, 1.0032353, 0.9942302, 1.0026186, -0.0083072, 0.0090052
7: 0.0071463, 0.0195631, 0.0066090, 0.0192085, -0.0088241, 0.0096594
8: 0.0022551, 0.0071173, 0.0025326, 0.0070062, -0.0047511, 0.0045847
9: -0.0266338, -0.0137702, -0.0260045, -0.0134343, -0.0131995, 0.0122343

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0063437, upper bound: 0.0065016
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0075391, upper bound: 0.0076634
time: 1.30 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0075391, upper bound: 0.0076634
time: 1.53 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0043476, 0.0085679, 0.0046482, 0.0085402, -0.0041927, 0.0039197
1: 0.0003066, 0.0046774, 0.0005930, 0.0046238, -0.0043171, 0.0040844
2: -0.0244527, -0.0051220, -0.0237360, -0.0038002, -0.0206525, 0.0186141
3: -0.0020551, 0.0068436, -0.0021732, 0.0062217, -0.0082768, 0.0090168
4: 0.0113652, 0.0181588, 0.0107924, 0.0179714, -0.0066062, 0.0073664
5: -0.0032305, 0.0083880, -0.0032025, 0.0075122, -0.0107426, 0.0115905
6: 0.9943179, 1.0031043, 0.9941612, 1.0025142, -0.0081963, 0.0089431
7: 0.0071901, 0.0194877, 0.0061533, 0.0191484, -0.0088462, 0.0098226
8: 0.0023141, 0.0070937, 0.0025796, 0.0069874, -0.0046733, 0.0045141
9: -0.0265000, -0.0137976, -0.0258980, -0.0131493, -0.0133507, 0.0121003

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0075391, upper bound: 0.0076634
time: 1.41 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0075391, upper bound: 0.0076634
time: 1.32 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0042994, 0.0085724, 0.0044182, 0.0085614, -0.0042620, 0.0041542
1: 0.0002607, 0.0046860, 0.0003739, 0.0046648, -0.0044041, 0.0043121
2: -0.0245676, -0.0054196, -0.0242844, -0.0045198, -0.0200478, 0.0188648
3: -0.0020285, 0.0069433, -0.0021089, 0.0066976, -0.0087261, 0.0090522
4: 0.0114942, 0.0181888, 0.0111042, 0.0181148, -0.0066206, 0.0070846
5: -0.0032350, 0.0085284, -0.0032239, 0.0081824, -0.0114173, 0.0117523
6: 0.9943532, 1.0031989, 0.9942465, 1.0029657, -0.0086125, 0.0089524
7: 0.0074236, 0.0195421, 0.0067177, 0.0194080, -0.0087547, 0.0094932
8: 0.0022715, 0.0071107, 0.0023765, 0.0070688, -0.0047972, 0.0047343
9: -0.0265966, -0.0139436, -0.0263587, -0.0135023, -0.0130943, 0.0124150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074200, upper bound: 0.0075126
time: 1.20 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074200, upper bound: 0.0075126
time: 1.66 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0043650, 0.0085663, 0.0044651, 0.0085571, -0.0041921, 0.0041013
1: 0.0003232, 0.0046743, 0.0004186, 0.0046564, -0.0043332, 0.0042557
2: -0.0244112, -0.0054771, -0.0241726, -0.0039316, -0.0204796, 0.0186955
3: -0.0020234, 0.0068076, -0.0021614, 0.0066006, -0.0086239, 0.0089690
4: 0.0115191, 0.0181479, 0.0108494, 0.0180855, -0.0065665, 0.0072986
5: -0.0032288, 0.0083372, -0.0032195, 0.0080457, -0.0112745, 0.0115568
6: 0.9943601, 1.0030701, 0.9941767, 1.0028735, -0.0085134, 0.0088934
7: 0.0074687, 0.0194681, 0.0062564, 0.0193551, -0.0087757, 0.0097912
8: 0.0023295, 0.0070876, 0.0024179, 0.0070522, -0.0047227, 0.0046696
9: -0.0264652, -0.0139718, -0.0262647, -0.0132138, -0.0132514, 0.0122929

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 165

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074200, upper bound: 0.0075126
time: 1.34 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074200, upper bound: 0.0075126
time: 1.70 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0042864, 0.0085736, 0.0044344, 0.0085599, -0.0042735, 0.0041391
1: 0.0002484, 0.0046883, 0.0003894, 0.0046619, -0.0044135, 0.0042989
2: -0.0245986, -0.0052252, -0.0242456, -0.0045323, -0.0200663, 0.0190203
3: -0.0020459, 0.0069702, -0.0021078, 0.0066639, -0.0087098, 0.0090780
4: 0.0114099, 0.0181969, 0.0111097, 0.0181046, -0.0066947, 0.0070873
5: -0.0032362, 0.0085662, -0.0032224, 0.0081349, -0.0113710, 0.0117886
6: 0.9943302, 1.0032244, 0.9942480, 1.0029337, -0.0086035, 0.0089764
7: 0.0072711, 0.0195567, 0.0067276, 0.0193896, -0.0089000, 0.0094983
8: 0.0022601, 0.0071153, 0.0023908, 0.0070630, -0.0048029, 0.0047245
9: -0.0266225, -0.0138483, -0.0263260, -0.0135084, -0.0131141, 0.0124777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074257, upper bound: 0.0075914
time: 1.44 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074257, upper bound: 0.0075914
time: 1.35 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0043532, 0.0085674, 0.0044835, 0.0085554, -0.0042022, 0.0040839
1: 0.0003120, 0.0046764, 0.0004362, 0.0046531, -0.0043411, 0.0042402
2: -0.0244392, -0.0052817, -0.0241286, -0.0039451, -0.0204941, 0.0188468
3: -0.0020408, 0.0068319, -0.0021602, 0.0065624, -0.0086032, 0.0089922
4: 0.0114344, 0.0181553, 0.0108552, 0.0180740, -0.0066396, 0.0073001
5: -0.0032299, 0.0083715, -0.0032178, 0.0079919, -0.0112218, 0.0115893
6: 0.9943370, 1.0030931, 0.9941784, 1.0028374, -0.0085005, 0.0089148
7: 0.0073154, 0.0194813, 0.0062669, 0.0193343, -0.0089218, 0.0097944
8: 0.0023191, 0.0070917, 0.0024342, 0.0070456, -0.0047266, 0.0046575
9: -0.0264887, -0.0138760, -0.0262277, -0.0132204, -0.0132683, 0.0123517

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 90

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 165

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074257, upper bound: 0.0075914
time: 1.16 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074257, upper bound: 0.0075914
time: 1.35 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0046254, 0.0085423, 0.0045304, 0.0085511, -0.0039256, 0.0040119
1: 0.0005714, 0.0046278, 0.0004809, 0.0046448, -0.0040734, 0.0041470
2: -0.0237902, -0.0043914, -0.0240167, -0.0055804, -0.0182098, 0.0196253
3: -0.0021204, 0.0062687, -0.0020142, 0.0064653, -0.0085857, 0.0082829
4: 0.0110486, 0.0179856, 0.0115638, 0.0180448, -0.0069962, 0.0064217
5: -0.0032046, 0.0075784, -0.0032134, 0.0078552, -0.0110598, 0.0107919
6: 0.9942313, 1.0025587, 0.9943724, 1.0027453, -0.0085139, 0.0081863
7: 0.0066170, 0.0191741, 0.0075497, 0.0192813, -0.0094658, 0.0085803
8: 0.0025595, 0.0069955, 0.0024756, 0.0070291, -0.0044695, 0.0045199
9: -0.0259435, -0.0134393, -0.0261338, -0.0140225, -0.0119210, 0.0126945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069942, upper bound: 0.0071234
time: 1.58 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069942, upper bound: 0.0072348
time: 1.69 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0046254, 0.0085423, 0.0045924, 0.0085454, -0.0039199, 0.0039499
1: 0.0005714, 0.0046278, 0.0005399, 0.0046337, -0.0040623, 0.0040879
2: -0.0237902, -0.0043914, -0.0238689, -0.0050482, -0.0187420, 0.0194776
3: -0.0021204, 0.0062687, -0.0020617, 0.0063370, -0.0084574, 0.0083304
4: 0.0110486, 0.0179856, 0.0113332, 0.0180061, -0.0069575, 0.0066523
5: -0.0032046, 0.0075784, -0.0032077, 0.0076746, -0.0108792, 0.0107861
6: 0.9942313, 1.0025587, 0.9943092, 1.0026236, -0.0083922, 0.0082495
7: 0.0066170, 0.0191741, 0.0071322, 0.0192114, -0.0093296, 0.0088988
8: 0.0025595, 0.0069955, 0.0025303, 0.0070071, -0.0044476, 0.0044651
9: -0.0259435, -0.0134393, -0.0260096, -0.0137615, -0.0121820, 0.0125703

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069942, upper bound: 0.0071234
time: 1.32 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069942, upper bound: 0.0072348
time: 1.32 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0046815, 0.0085372, 0.0044751, 0.0085562, -0.0038747, 0.0040621
1: 0.0006247, 0.0046178, 0.0004282, 0.0046546, -0.0040299, 0.0041897
2: -0.0236566, -0.0038115, -0.0241486, -0.0055578, -0.0180988, 0.0203371
3: -0.0021721, 0.0061528, -0.0020162, 0.0065798, -0.0087519, 0.0081690
4: 0.0107973, 0.0179506, 0.0115541, 0.0180793, -0.0072820, 0.0063966
5: -0.0031994, 0.0074152, -0.0032186, 0.0080164, -0.0112158, 0.0106337
6: 0.9941625, 1.0024488, 0.9943697, 1.0028540, -0.0086915, 0.0080791
7: 0.0061621, 0.0191109, 0.0075320, 0.0193438, -0.0097828, 0.0085082
8: 0.0026091, 0.0069756, 0.0024268, 0.0070486, -0.0044395, 0.0045489
9: -0.0258312, -0.0131549, -0.0262446, -0.0140114, -0.0118198, 0.0130897

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069328, upper bound: 0.0070671
time: 1.23 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069328, upper bound: 0.0072175
time: 1.28 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0046815, 0.0085372, 0.0048162, 0.0085248, -0.0038433, 0.0037209
1: 0.0006247, 0.0046178, 0.0007531, 0.0045938, -0.0039690, 0.0038647
2: -0.0236566, -0.0038115, -0.0233354, -0.0046101, -0.0190465, 0.0195239
3: -0.0021721, 0.0061528, -0.0021008, 0.0058740, -0.0080462, 0.0082536
4: 0.0107973, 0.0179506, 0.0111434, 0.0178666, -0.0070693, 0.0068072
5: -0.0031994, 0.0074152, -0.0031868, 0.0070226, -0.0102220, 0.0106020
6: 0.9941625, 1.0024488, 0.9942573, 1.0021844, -0.0080219, 0.0081915
7: 0.0061621, 0.0191109, 0.0067886, 0.0189588, -0.0090304, 0.0089248
8: 0.0026091, 0.0069756, 0.0027281, 0.0069280, -0.0043189, 0.0042476
9: -0.0258312, -0.0131549, -0.0255614, -0.0135466, -0.0122847, 0.0124065

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069328, upper bound: 0.0070671
time: 1.36 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069328, upper bound: 0.0072175
time: 1.36 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0046367, 0.0085413, 0.0045214, 0.0085519, -0.0039153, 0.0040199
1: 0.0005821, 0.0046258, 0.0004722, 0.0046464, -0.0040643, 0.0041536
2: -0.0237634, -0.0044073, -0.0240383, -0.0053751, -0.0183884, 0.0196311
3: -0.0021189, 0.0062455, -0.0020325, 0.0064841, -0.0086030, 0.0082780
4: 0.0110555, 0.0179785, 0.0114749, 0.0180504, -0.0069950, 0.0065037
5: -0.0032036, 0.0075457, -0.0032143, 0.0078816, -0.0110852, 0.0107600
6: 0.9942332, 1.0025368, 0.9943480, 1.0027632, -0.0085300, 0.0081888
7: 0.0066295, 0.0191614, 0.0073886, 0.0192915, -0.0094689, 0.0087612
8: 0.0025694, 0.0069915, 0.0024676, 0.0070323, -0.0044628, 0.0045239
9: -0.0259210, -0.0134471, -0.0261519, -0.0139218, -0.0119992, 0.0127048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0070337, upper bound: 0.0071253
time: 1.21 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0070337, upper bound: 0.0072348
time: 1.31 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0046367, 0.0085413, 0.0045834, 0.0085462, -0.0039095, 0.0039579
1: 0.0005821, 0.0046258, 0.0005313, 0.0046353, -0.0040532, 0.0040945
2: -0.0237634, -0.0044073, -0.0238904, -0.0048659, -0.0188975, 0.0194831
3: -0.0021189, 0.0062455, -0.0020780, 0.0063557, -0.0084746, 0.0083235
4: 0.0110555, 0.0179785, 0.0112542, 0.0180117, -0.0069563, 0.0067243
5: -0.0032036, 0.0075457, -0.0032085, 0.0077008, -0.0109044, 0.0107542
6: 0.9942332, 1.0025368, 0.9942876, 1.0026412, -0.0084080, 0.0082492
7: 0.0066295, 0.0191614, 0.0069892, 0.0192215, -0.0093342, 0.0090515
8: 0.0025694, 0.0069915, 0.0025224, 0.0070103, -0.0044409, 0.0044691
9: -0.0259210, -0.0134471, -0.0260276, -0.0136720, -0.0122489, 0.0125806

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0070337, upper bound: 0.0071253
time: 1.22 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0070337, upper bound: 0.0072348
time: 1.28 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0046925, 0.0085362, 0.0044615, 0.0085574, -0.0038649, 0.0040746
1: 0.0006353, 0.0046158, 0.0004152, 0.0046571, -0.0040218, 0.0042007
2: -0.0236303, -0.0038287, -0.0241811, -0.0053523, -0.0182780, 0.0203524
3: -0.0021706, 0.0061300, -0.0020345, 0.0066079, -0.0087785, 0.0081645
4: 0.0108047, 0.0179437, 0.0114650, 0.0180878, -0.0072830, 0.0064787
5: -0.0031984, 0.0073830, -0.0032199, 0.0080560, -0.0112544, 0.0106028
6: 0.9941646, 1.0024272, 0.9943453, 1.0028806, -0.0087160, 0.0080819
7: 0.0061756, 0.0190984, 0.0073708, 0.0193591, -0.0097834, 0.0086844
8: 0.0026188, 0.0069717, 0.0024147, 0.0070534, -0.0044346, 0.0045570
9: -0.0258091, -0.0131633, -0.0262718, -0.0139106, -0.0118985, 0.0131085

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069845, upper bound: 0.0070675
time: 1.49 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069845, upper bound: 0.0072175
time: 1.33 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0046925, 0.0085362, 0.0048137, 0.0085250, -0.0038325, 0.0037224
1: 0.0006353, 0.0046158, 0.0007507, 0.0045942, -0.0039590, 0.0038651
2: -0.0236303, -0.0038287, -0.0233413, -0.0044309, -0.0191994, 0.0195126
3: -0.0021706, 0.0061300, -0.0021168, 0.0058792, -0.0080498, 0.0082468
4: 0.0108047, 0.0179437, 0.0110657, 0.0178682, -0.0070634, 0.0068780
5: -0.0031984, 0.0073830, -0.0031871, 0.0070298, -0.0102282, 0.0105701
6: 0.9941646, 1.0024272, 0.9942360, 1.0021892, -0.0080246, 0.0081912
7: 0.0061756, 0.0190984, 0.0066480, 0.0189616, -0.0090720, 0.0091113
8: 0.0026188, 0.0069717, 0.0027258, 0.0069289, -0.0043101, 0.0042459
9: -0.0258091, -0.0131633, -0.0255664, -0.0134587, -0.0123505, 0.0124031

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069845, upper bound: 0.0070675
time: 1.16 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069845, upper bound: 0.0072175
time: 1.38 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0044637, 0.0085572, 0.0045371, 0.0085505, -0.0040868, 0.0040202
1: 0.0004173, 0.0046567, 0.0004871, 0.0046436, -0.0042263, 0.0041695
2: -0.0241759, -0.0045456, -0.0240010, -0.0057241, -0.0184518, 0.0194554
3: -0.0021066, 0.0066034, -0.0020013, 0.0064516, -0.0085582, 0.0086047
4: 0.0111154, 0.0180864, 0.0116261, 0.0180407, -0.0069253, 0.0064603
5: -0.0032197, 0.0080497, -0.0032128, 0.0078360, -0.0110556, 0.0112625
6: 0.9942496, 1.0028764, 0.9943894, 1.0027323, -0.0084827, 0.0084870
7: 0.0067380, 0.0193566, 0.0076624, 0.0192739, -0.0093098, 0.0086562
8: 0.0024167, 0.0070526, 0.0024815, 0.0070267, -0.0046100, 0.0045712
9: -0.0262674, -0.0135149, -0.0261205, -0.0140930, -0.0121745, 0.0126056

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 4.10 + 596.56 = 600.67 seconds
