## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 8)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.144541844


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-13.1750603, -12.2906647, -13.1750603, -12.2906647, -0.3713887, 0.3713887)
1: (-4.0107384, -3.4168868, -4.0107384, -3.4168868, -0.4484067, 0.4484067)
2: (0.1172709, 0.7854009, 0.1172709, 0.7854009, -0.4432204, 0.4432206)
3: (-3.5588059, -2.8935289, -3.5588059, -2.8935289, -0.3687387, 0.3687387)
4: (-3.6335125, -2.9203374, -3.6335125, -2.9203374, -0.3927388, 0.3927386)
5: (-13.0396671, -12.2592764, -13.0396671, -12.2592764, -0.4323056, 0.4323056)
6: (-12.9865103, -12.1916008, -12.9865103, -12.1916008, -0.6171718, 0.6171718)
7: (1.6868830, 2.3739386, 1.6868830, 2.3739386, -0.3184681, 0.3184681)
8: (-2.6495357, -2.0242376, -2.6495357, -2.0242376, -0.4539795, 0.4539795)
9: (-5.0104799, -4.2063570, -5.0104799, -4.2063570, -0.4150467, 0.4150467)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.35 + 35.36 = 58.71 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.1571105, upper bound: 0.1571109

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4582
type: B, layer: 1, pos: 4582
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4582

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1565351, upper bound: 0.1570968
time: 3.43 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1570965, upper bound: 0.1570977
time: 3.75 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.39 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 7.39
Output dim: 2, lower bound: -0.1565351, upper bound: 0.1570968
NS_A2, status: Status.UNKNOWN, split count: 1, time: 7.39
Output dim: 2, lower bound: -0.1570965, upper bound: 0.1570977

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -13.1745892, -12.2906923, -13.1750603, -12.2906647, -0.3709896, 0.3711104
1: -4.0104728, -3.4169037, -4.0107384, -3.4168868, -0.4480481, 0.4482598
2: 0.1178699, 0.7853537, 0.1172709, 0.7854009, -0.4426446, 0.4431329
3: -3.5587521, -2.8935828, -3.5588059, -2.8935289, -0.3686354, 0.3686156
4: -3.6334825, -2.9203610, -3.6335125, -2.9203374, -0.3926129, 0.3926134
5: -13.0395641, -12.2592850, -13.0396671, -12.2592764, -0.4320412, 0.4321184
6: -12.9862175, -12.1917534, -12.9865103, -12.1916008, -0.6170259, 0.6169376
7: 1.6869063, 2.3733387, 1.6868830, 2.3739386, -0.3184378, 0.3178973
8: -2.6494389, -2.0242977, -2.6495357, -2.0242376, -0.4538822, 0.4538832
9: -5.0101418, -4.2063789, -5.0104799, -4.2063570, -0.4147058, 0.4150267

Time for backsubstitution: 20.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4582
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 4584
type: A, layer: 1, pos: 4584

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 4582

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1565348, upper bound: 0.1565347
time: 5.36 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1565348, upper bound: 0.1570972
time: 6.06 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -13.1751442, -12.2875433, -13.1750546, -12.2906647, -0.3761163, 0.3741598
1: -4.0111361, -3.4138761, -4.0107355, -3.4168880, -0.4513574, 0.4512868
2: 0.1167669, 0.7965226, 0.1172771, 0.7854004, -0.4449158, 0.4500513
3: -3.5591416, -2.8925710, -3.5588059, -2.8935289, -0.3702660, 0.3693163
4: -3.6338515, -2.9198232, -3.6335125, -2.9203358, -0.3927772, 0.3953984
5: -13.0398159, -12.2568760, -13.0396681, -12.2592764, -0.4365218, 0.4338541
6: -12.9905796, -12.1915417, -12.9865093, -12.1916027, -0.6229739, 0.6171441
7: 1.6762428, 2.3743439, 1.6868849, 2.3739300, -0.3256593, 0.3196275
8: -2.6496353, -2.0236902, -2.6495342, -2.0242367, -0.4548316, 0.4544134
9: -5.0113039, -4.2010088, -5.0104742, -4.2063537, -0.4154415, 0.4204059

Time for backsubstitution: 21.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4582
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 4584
type: A, layer: 1, pos: 4584

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 4582

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1570972, upper bound: 0.1565348
time: 5.73 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1570972, upper bound: 0.1570971
time: 6.08 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 33.33 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 33.33
Output dim: 2, lower bound: -0.1565348, upper bound: 0.1565347
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 33.33
Output dim: 2, lower bound: -0.1565348, upper bound: 0.1570972
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 33.33
Output dim: 2, lower bound: -0.1570972, upper bound: 0.1565348
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 33.33
Output dim: 2, lower bound: -0.1570972, upper bound: 0.1570971

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -13.1745892, -12.2906923, -13.1745892, -12.2906923, -0.3707113, 0.3707113
1: -4.0104728, -3.4169037, -4.0104728, -3.4169037, -0.4479012, 0.4479012
2: 0.1178699, 0.7853537, 0.1178699, 0.7853537, -0.4425571, 0.4425573
3: -3.5587521, -2.8935828, -3.5587521, -2.8935828, -0.3685122, 0.3685124
4: -3.6334825, -2.9203610, -3.6334825, -2.9203610, -0.3924875, 0.3924873
5: -13.0395641, -12.2592850, -13.0395641, -12.2592850, -0.4318540, 0.4318540
6: -12.9862175, -12.1917534, -12.9862175, -12.1917534, -0.6167912, 0.6167912
7: 1.6869063, 2.3733387, 1.6869063, 2.3733387, -0.3178668, 0.3178668
8: -2.6494389, -2.0242977, -2.6494389, -2.0242977, -0.4537854, 0.4537859
9: -5.0101418, -4.2063789, -5.0101418, -4.2063789, -0.4146862, 0.4146862

Time for backsubstitution: 22.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 4629

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1565215, upper bound: 0.1562811
time: 5.28 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1565215, upper bound: 0.1565206
time: 4.84 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -13.1745892, -12.2906923, -13.1751423, -12.2875433, -0.3737624, 0.3711894
1: -4.0104728, -3.4169037, -4.0111332, -3.4138746, -0.4509339, 0.4482970
2: 0.1178699, 0.7853537, 0.1167679, 0.7965212, -0.4494743, 0.4436049
3: -3.5587521, -2.8935828, -3.5591412, -2.8925729, -0.3692107, 0.3688955
4: -3.6334825, -2.9203610, -3.6338513, -2.9198277, -0.3930314, 0.3926518
5: -13.0395641, -12.2592850, -13.0398169, -12.2568779, -0.4335909, 0.4321396
6: -12.9862175, -12.1917534, -12.9905691, -12.1915417, -0.6169987, 0.6206565
7: 1.6869063, 2.3733387, 1.6762419, 2.3743429, -0.3187740, 0.3250866
8: -2.6494389, -2.0242977, -2.6496348, -2.0236893, -0.4543176, 0.4539886
9: -5.0101418, -4.2063789, -5.0113049, -4.2010083, -0.4200687, 0.4152441

Time for backsubstitution: 22.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 4629

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1565215, upper bound: 0.1568435
time: 9.13 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1565215, upper bound: 0.1570829
time: 4.99 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -13.1751423, -12.2875433, -13.1745892, -12.2906923, -0.3711894, 0.3737624
1: -4.0111332, -3.4138746, -4.0104728, -3.4169037, -0.4482970, 0.4509339
2: 0.1167679, 0.7965212, 0.1178699, 0.7853537, -0.4436049, 0.4494743
3: -3.5591412, -2.8925729, -3.5587521, -2.8935828, -0.3688953, 0.3692105
4: -3.6338513, -2.9198277, -3.6334825, -2.9203610, -0.3926516, 0.3930309
5: -13.0398169, -12.2568779, -13.0395641, -12.2592850, -0.4321396, 0.4335909
6: -12.9905691, -12.1915417, -12.9862175, -12.1917534, -0.6206565, 0.6169987
7: 1.6762419, 2.3743429, 1.6869063, 2.3733387, -0.3250866, 0.3187740
8: -2.6496348, -2.0236893, -2.6494389, -2.0242977, -0.4539886, 0.4543171
9: -5.0113049, -4.2010083, -5.0101418, -4.2063789, -0.4152441, 0.4200687

Time for backsubstitution: 22.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 4584
type: A, layer: 1, pos: 4584

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 4629

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1568429, upper bound: 0.1565217
time: 3.54 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1570825, upper bound: 0.1565214
time: 6.15 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -13.1751432, -12.2875490, -13.1751432, -12.2875490, -0.3764195, 0.3771126
1: -4.0111890, -3.4138758, -4.0111890, -3.4138758, -0.4515452, 0.4515452
2: 0.1167679, 0.7965231, 0.1167679, 0.7965231, -0.4452920, 0.4452920
3: -3.5591421, -2.8925343, -3.5591421, -2.8925343, -0.3705535, 0.3705537
4: -3.6338520, -2.9197850, -3.6338520, -2.9197850, -0.3955593, 0.3955595
5: -13.0398159, -12.2568741, -13.0398159, -12.2568741, -0.4367754, 0.4367754
6: -12.9906893, -12.1915417, -12.9906893, -12.1915417, -0.6233473, 0.6233473
7: 1.6762428, 2.3743443, 1.6762428, 2.3743443, -0.3205230, 0.3205230
8: -2.6496363, -2.0236902, -2.6496363, -2.0236902, -0.4554029, 0.4554029
9: -5.0113039, -4.2010078, -5.0113039, -4.2010078, -0.4186726, 0.4186726

Time for backsubstitution: 21.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 4584
type: A, layer: 1, pos: 4584

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4629

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1570836, upper bound: 0.1562817
time: 3.40 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1570836, upper bound: 0.1565213
time: 4.05 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 29.45 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 29.45
Output dim: 2, lower bound: -0.1565215, upper bound: 0.1562811
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.45
Output dim: 2, lower bound: -0.1565215, upper bound: 0.1565206
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.45
Output dim: 2, lower bound: -0.1565215, upper bound: 0.1568435
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.45
Output dim: 2, lower bound: -0.1565215, upper bound: 0.1570829
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 29.45
Output dim: 2, lower bound: -0.1568429, upper bound: 0.1565217
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 29.45
Output dim: 2, lower bound: -0.1570825, upper bound: 0.1565214
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.45
Output dim: 2, lower bound: -0.1570836, upper bound: 0.1562817
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.45
Output dim: 2, lower bound: -0.1570836, upper bound: 0.1565213

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -13.1744862, -12.2907562, -13.1745892, -12.2906923, -0.3705096, 0.3705697
1: -4.0103536, -3.4169240, -4.0104728, -3.4169037, -0.4476800, 0.4476829
2: 0.1179752, 0.7849522, 0.1178699, 0.7853537, -0.4424884, 0.4421339
3: -3.5586138, -2.8936291, -3.5587521, -2.8935828, -0.3682575, 0.3683052
4: -3.6334245, -2.9209309, -3.6334825, -2.9203610, -0.3924618, 0.3919253
5: -13.0394964, -12.2597828, -13.0395641, -12.2592850, -0.4318194, 0.4313507
6: -12.9861336, -12.1918602, -12.9862175, -12.1917534, -0.6167383, 0.6166725
7: 1.6871395, 2.3732786, 1.6869063, 2.3733387, -0.3176281, 0.3178275
8: -2.6492877, -2.0243821, -2.6494389, -2.0242977, -0.4536481, 0.4537249
9: -5.0092373, -4.2064090, -5.0101418, -4.2063789, -0.4137821, 0.4146466

Time for backsubstitution: 21.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 4584
type: A, layer: 1, pos: 4584

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4629

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1562815, upper bound: 0.1562812
time: 4.75 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1562815, upper bound: 0.1562812
time: 3.87 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -13.1747208, -12.2887497, -13.1745863, -12.2906952, -0.3708074, 0.3757529
1: -4.0110478, -3.4167061, -4.0104723, -3.4169149, -0.4530377, 0.4479685
2: 0.1100421, 0.7863512, 0.1178718, 0.7853489, -0.4503322, 0.4440637
3: -3.5592461, -2.8905907, -3.5587506, -2.8935843, -0.3730061, 0.3709719
4: -3.6474874, -2.9201360, -3.6334810, -2.9203684, -0.3990419, 0.3937721
5: -13.0511503, -12.2586088, -13.0395632, -12.2592888, -0.4377899, 0.4324541
6: -12.9877224, -12.1913624, -12.9862185, -12.1917543, -0.6183701, 0.6170483
7: 1.6866918, 2.3789167, 1.6869097, 2.3733368, -0.3182883, 0.3235886
8: -2.6501975, -2.0231714, -2.6494360, -2.0242982, -0.4546990, 0.4550490
9: -5.0106764, -4.1819463, -5.0101295, -4.2063823, -0.4166389, 0.4248385

Time for backsubstitution: 21.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4629

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1562815, upper bound: 0.1565217
time: 6.69 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1562815, upper bound: 0.1565215
time: 4.82 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -13.1744862, -12.2907562, -13.1751423, -12.2875433, -0.3735604, 0.3710477
1: -4.0103536, -3.4169240, -4.0111332, -3.4138746, -0.4507127, 0.4480786
2: 0.1179752, 0.7849522, 0.1167679, 0.7965212, -0.4494066, 0.4431820
3: -3.5586138, -2.8936291, -3.5591412, -2.8925729, -0.3689556, 0.3686881
4: -3.6334245, -2.9209309, -3.6338513, -2.9198277, -0.3930056, 0.3920898
5: -13.0394964, -12.2597828, -13.0398169, -12.2568779, -0.4335563, 0.4316363
6: -12.9861336, -12.1918602, -12.9905691, -12.1915417, -0.6169457, 0.6205378
7: 1.6871395, 2.3732786, 1.6762419, 2.3743429, -0.3185353, 0.3250489
8: -2.6492877, -2.0243821, -2.6496348, -2.0236893, -0.4541802, 0.4539270
9: -5.0092373, -4.2064090, -5.0113049, -4.2010083, -0.4191647, 0.4152050

Time for backsubstitution: 21.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4629

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1562813, upper bound: 0.1568426
time: 4.85 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1562813, upper bound: 0.1568422
time: 4.86 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -13.1747208, -12.2887497, -13.1751413, -12.2875452, -0.3738592, 0.3762307
1: -4.0110478, -3.4167061, -4.0111308, -3.4138861, -0.4560690, 0.4483643
2: 0.1100421, 0.7863512, 0.1167698, 0.7965178, -0.4507194, 0.4451113
3: -3.5592461, -2.8905907, -3.5591407, -2.8925738, -0.3737028, 0.3713553
4: -3.6474874, -2.9201360, -3.6338487, -2.9198351, -0.3991747, 0.3939371
5: -13.0511503, -12.2586088, -13.0398140, -12.2568817, -0.4380624, 0.4327400
6: -12.9877224, -12.1913624, -12.9905682, -12.1915417, -0.6185775, 0.6209145
7: 1.6866918, 2.3789167, 1.6762447, 2.3743439, -0.3191955, 0.3260996
8: -2.6501975, -2.0231714, -2.6496329, -2.0236912, -0.4552302, 0.4552512
9: -5.0106764, -4.1819463, -5.0112896, -4.2010107, -0.4220214, 0.4253576

Time for backsubstitution: 21.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 4629

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1562813, upper bound: 0.1570828
time: 5.01 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1562813, upper bound: 0.1570827
time: 4.60 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -13.1751423, -12.2875433, -13.1744862, -12.2907562, -0.3710477, 0.3735607
1: -4.0111332, -3.4138746, -4.0103536, -3.4169240, -0.4480786, 0.4507127
2: 0.1167679, 0.7965212, 0.1179752, 0.7849522, -0.4431818, 0.4494066
3: -3.5591412, -2.8925729, -3.5586138, -2.8936291, -0.3686881, 0.3689556
4: -3.6338513, -2.9198277, -3.6334245, -2.9209309, -0.3920898, 0.3930058
5: -13.0398169, -12.2568779, -13.0394964, -12.2597828, -0.4316363, 0.4335563
6: -12.9905691, -12.1915417, -12.9861336, -12.1918602, -0.6205378, 0.6169457
7: 1.6762419, 2.3743429, 1.6871395, 2.3732786, -0.3250489, 0.3185353
8: -2.6496348, -2.0236893, -2.6492877, -2.0243821, -0.4539270, 0.4541802
9: -5.0113049, -4.2010083, -5.0092373, -4.2064090, -0.4152050, 0.4191647

Time for backsubstitution: 21.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 4584
type: A, layer: 1, pos: 4584

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4629

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1568427, upper bound: 0.1562815
time: 4.93 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1568427, upper bound: 0.1565220
time: 3.83 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -13.1751413, -12.2875452, -13.1747208, -12.2887497, -0.3762307, 0.3738592
1: -4.0111308, -3.4138861, -4.0110478, -3.4167061, -0.4483643, 0.4560690
2: 0.1167698, 0.7965178, 0.1100421, 0.7863512, -0.4451113, 0.4507194
3: -3.5591407, -2.8925738, -3.5592461, -2.8905907, -0.3713555, 0.3737030
4: -3.6338487, -2.9198351, -3.6474874, -2.9201360, -0.3939373, 0.3991747
5: -13.0398140, -12.2568817, -13.0511503, -12.2586088, -0.4327400, 0.4380624
6: -12.9905682, -12.1915417, -12.9877224, -12.1913624, -0.6209145, 0.6185770
7: 1.6762447, 2.3743439, 1.6866918, 2.3789167, -0.3260996, 0.3191957
8: -2.6496329, -2.0236912, -2.6501975, -2.0231714, -0.4552512, 0.4552302
9: -5.0112896, -4.2010107, -5.0106764, -4.1819463, -0.4253578, 0.4220209

Time for backsubstitution: 22.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 4584
type: A, layer: 1, pos: 4584

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4629

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1570829, upper bound: 0.1562819
time: 3.74 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1570829, upper bound: 0.1565220
time: 3.55 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -13.1750422, -12.2876072, -13.1751432, -12.2875490, -0.3762174, 0.3769712
1: -4.0110674, -3.4138958, -4.0111890, -3.4138758, -0.4513230, 0.4513264
2: 0.1168718, 0.7961216, 0.1167679, 0.7965231, -0.4452240, 0.4448690
3: -3.5590038, -2.8925819, -3.5591421, -2.8925343, -0.3702991, 0.3703454
4: -3.6337931, -2.9203546, -3.6338520, -2.9197850, -0.3955331, 0.3949976
5: -13.0397491, -12.2573748, -13.0398159, -12.2568741, -0.4367411, 0.4362724
6: -12.9906101, -12.1916523, -12.9906893, -12.1915417, -0.6232963, 0.6232290
7: 1.6764746, 2.3742862, 1.6762428, 2.3743443, -0.3202848, 0.3204839
8: -2.6494842, -2.0237765, -2.6496363, -2.0236902, -0.4552650, 0.4553409
9: -5.0103946, -4.2010398, -5.0113039, -4.2010078, -0.4177675, 0.4186330

Time for backsubstitution: 22.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 4584
type: A, layer: 1, pos: 4584

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 4629

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1568434, upper bound: 0.1562816
time: 3.76 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1568434, upper bound: 0.1562811
time: 4.25 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -13.1752720, -12.2856016, -13.1751413, -12.2875452, -0.3765230, 0.3782854
1: -4.0117731, -3.4136767, -4.0111871, -3.4138865, -0.4566784, 0.4516106
2: 0.1089454, 0.7975230, 0.1167684, 0.7965178, -0.4518633, 0.4468026
3: -3.5596347, -2.8895316, -3.5591388, -2.8925347, -0.3750453, 0.3730178
4: -3.6478529, -2.9195592, -3.6338496, -2.9197934, -0.3997254, 0.3968439
5: -13.0514011, -12.2561970, -13.0398159, -12.2568789, -0.4386952, 0.4373755
6: -12.9922447, -12.1911507, -12.9906902, -12.1915417, -0.6249990, 0.6236043
7: 1.6760268, 2.3799200, 1.6762447, 2.3743439, -0.3209453, 0.3262451
8: -2.6503911, -2.0225654, -2.6496320, -2.0236921, -0.4563146, 0.4566631
9: -5.0118203, -4.1765742, -5.0112891, -4.2010093, -0.4206228, 0.4256389

Time for backsubstitution: 22.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 4629

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1568434, upper bound: 0.1565217
time: 3.92 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1568434, upper bound: 0.1565211
time: 4.42 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 30.83 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.83
Output dim: 2, lower bound: -0.1562815, upper bound: 0.1562812
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.83
Output dim: 2, lower bound: -0.1562815, upper bound: 0.1562812
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.83
Output dim: 2, lower bound: -0.1562815, upper bound: 0.1565217
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.83
Output dim: 2, lower bound: -0.1562815, upper bound: 0.1565215
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.83
Output dim: 2, lower bound: -0.1562813, upper bound: 0.1568426
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.83
Output dim: 2, lower bound: -0.1562813, upper bound: 0.1568422
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.83
Output dim: 2, lower bound: -0.1562813, upper bound: 0.1570828
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.83
Output dim: 2, lower bound: -0.1562813, upper bound: 0.1570827
NS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 30.83
Output dim: 2, lower bound: -0.1568427, upper bound: 0.1562815
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 30.83
Output dim: 2, lower bound: -0.1568427, upper bound: 0.1565220
NS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 30.83
Output dim: 2, lower bound: -0.1570829, upper bound: 0.1562819
NS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 30.83
Output dim: 2, lower bound: -0.1570829, upper bound: 0.1565220
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.83
Output dim: 2, lower bound: -0.1568434, upper bound: 0.1562816
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.83
Output dim: 2, lower bound: -0.1568434, upper bound: 0.1562811
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.83
Output dim: 2, lower bound: -0.1568434, upper bound: 0.1565217
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.83
Output dim: 2, lower bound: -0.1568434, upper bound: 0.1565211

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -13.1744862, -12.2907562, -13.1744862, -12.2907562, -0.3703680, 0.3703680
1: -4.0103536, -3.4169240, -4.0103536, -3.4169240, -0.4474616, 0.4474616
2: 0.1179752, 0.7849522, 0.1179752, 0.7849522, -0.4420652, 0.4420652
3: -3.5586138, -2.8936291, -3.5586138, -2.8936291, -0.3680503, 0.3680501
4: -3.6334245, -2.9209309, -3.6334245, -2.9209309, -0.3919001, 0.3918998
5: -13.0394964, -12.2597828, -13.0394964, -12.2597828, -0.4313161, 0.4313161
6: -12.9861336, -12.1918602, -12.9861336, -12.1918602, -0.6166191, 0.6166196
7: 1.6871395, 2.3732786, 1.6871395, 2.3732786, -0.3175886, 0.3175886
8: -2.6492877, -2.0243821, -2.6492877, -2.0243821, -0.4535875, 0.4535875
9: -5.0092373, -4.2064090, -5.0092373, -4.2064090, -0.4137421, 0.4137421

Time for backsubstitution: 22.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 4584

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1561887, upper bound: 0.1562772
time: 14.95 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1562763, upper bound: 0.1562765
time: 5.25 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -13.1744862, -12.2907562, -13.1747208, -12.2887497, -0.3724380, 0.3706656
1: -4.0103536, -3.4169240, -4.0110426, -3.4167094, -0.4477468, 0.4483633
2: 0.1179752, 0.7849522, 0.1100459, 0.7863379, -0.4435678, 0.4499083
3: -3.5586138, -2.8936291, -3.5592403, -2.8906069, -0.3707032, 0.3688955
4: -3.6334245, -2.9209309, -3.6474795, -2.9201365, -0.3927495, 0.3984745
5: -13.0394964, -12.2597828, -13.0511475, -12.2586098, -0.4320076, 0.4372835
6: -12.9861336, -12.1918602, -12.9877195, -12.1913719, -0.6169796, 0.6182523
7: 1.6871395, 2.3732786, 1.6866918, 2.3789082, -0.3233414, 0.3180590
8: -2.6492877, -2.0243821, -2.6501865, -2.0231733, -0.4549112, 0.4545741
9: -5.0092373, -4.2064090, -5.0106721, -4.1819468, -0.4239280, 0.4147911

Time for backsubstitution: 21.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4584

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1561887, upper bound: 0.1562762
time: 4.57 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1562763, upper bound: 0.1562762
time: 4.99 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -13.1747208, -12.2887497, -13.1744862, -12.2907562, -0.3706656, 0.3724380
1: -4.0110426, -3.4167094, -4.0103536, -3.4169240, -0.4483633, 0.4477468
2: 0.1100459, 0.7863379, 0.1179752, 0.7849522, -0.4499083, 0.4435678
3: -3.5592403, -2.8906069, -3.5586138, -2.8936291, -0.3688955, 0.3707032
4: -3.6474795, -2.9201365, -3.6334245, -2.9209309, -0.3984745, 0.3927491
5: -13.0511475, -12.2586098, -13.0394964, -12.2597828, -0.4372835, 0.4320076
6: -12.9877195, -12.1913719, -12.9861336, -12.1918602, -0.6182523, 0.6169796
7: 1.6866918, 2.3789082, 1.6871395, 2.3732786, -0.3180590, 0.3233414
8: -2.6501865, -2.0231733, -2.6492877, -2.0243821, -0.4545741, 0.4549108
9: -5.0106721, -4.1819468, -5.0092373, -4.2064090, -0.4147911, 0.4239283

Time for backsubstitution: 21.12 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 58.71 + 549.48 = 608.19 seconds
