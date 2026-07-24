## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.42358272


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1818094, 1.1818094)
1: (-8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6936989, 1.6936984)
2: (-9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3674593, 1.3674593)
3: (-10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0933785, 1.0933785)
4: (-5.2452884, -3.9792435, -5.2452884, -3.9792435, -0.9980512, 0.9980512)
5: (-8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9382401, 0.9382403)
6: (-12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.2460556, 1.2460556)
7: (1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.1358390, 1.1358390)
8: (-3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.3539286, 1.3539286)
9: (0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2844377, 1.2844377)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 24.24 + 33.27 = 57.52 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.4604159, upper bound: 0.4604158

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 634
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 95
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 2250
type: DSZ, layer: 3, pos: 2528
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1929
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 1206

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 634

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4604078, upper bound: 0.4600442
time: 3.72 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4600439, upper bound: 0.4604078
time: 3.49 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 7.22 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 7.22
Output dim: 7, lower bound: -0.4604078, upper bound: 0.4600442
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 7.22
Output dim: 7, lower bound: -0.4600439, upper bound: 0.4604078

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1817732, 1.1817756
1: -8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6937075, 1.6937017
2: -9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3674593, 1.3674588
3: -10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0932975, 1.0933051
4: -5.2452884, -3.9792435, -5.2452884, -3.9792435, -0.9979925, 0.9979763
5: -8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9382510, 0.9382551
6: -12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.2460537, 1.2460542
7: 1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.1359162, 1.1358991
8: -3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.3539219, 1.3539224
9: 0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2844143, 1.2844133

Time for backsubstitution: 8.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2528
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 1929
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 2250
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 95
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 227

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2528

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4603539, upper bound: 0.4510621
time: 3.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4514272, upper bound: 0.4599900
time: 3.56 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1817756, 1.1817732
1: -8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6937017, 1.6937075
2: -9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3674588, 1.3674593
3: -10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0933051, 1.0932975
4: -5.2452884, -3.9792435, -5.2452884, -3.9792435, -0.9979763, 0.9979925
5: -8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9382548, 0.9382508
6: -12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.2460542, 1.2460537
7: 1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.1358991, 1.1359162
8: -3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.3539224, 1.3539219
9: 0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2844138, 1.2844143

Time for backsubstitution: 8.57 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2250
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 95
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 2528
type: DSZ, layer: 3, pos: 1929
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 676

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2250

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4595352, upper bound: 0.4590168
time: 3.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4587055, upper bound: 0.4598960
time: 3.64 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 15.89 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 15.89
Output dim: 7, lower bound: -0.4603539, upper bound: 0.4510621
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 15.89
Output dim: 7, lower bound: -0.4514272, upper bound: 0.4599900
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 15.89
Output dim: 7, lower bound: -0.4595352, upper bound: 0.4590168
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 15.89
Output dim: 7, lower bound: -0.4587055, upper bound: 0.4598960

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1673794, 1.1657782
1: -8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6820717, 1.6833811
2: -9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3693552, 1.3686433
3: -10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0828156, 1.0809422
4: -5.2452884, -3.9792435, -5.2452884, -3.9792435, -0.9939089, 0.9939346
5: -8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9386024, 0.9387190
6: -12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.2211547, 1.2203884
7: 1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.1363339, 1.1361451
8: -3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.3542709, 1.3546438
9: 0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2789698, 1.2777691

Time for backsubstitution: 8.69 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 95
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 2250
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 1929

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1934

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4327019, upper bound: 0.4264956
time: 4.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4327019, upper bound: 0.4264956
time: 4.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1657758, 1.1673818
1: -8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6833858, 1.6820660
2: -9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3686433, 1.3693552
3: -10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0809345, 1.0828233
4: -5.2452884, -3.9792435, -5.2452884, -3.9792435, -0.9939518, 0.9938927
5: -8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9387150, 0.9386065
6: -12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.2203879, 1.2211552
7: 1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.1361623, 1.1363173
8: -3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.3546433, 1.3542709
9: 0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2777696, 1.2789693

Time for backsubstitution: 8.56 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 2250
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 1929
type: DSZ, layer: 3, pos: 95
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 1934

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 901

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4488336, upper bound: 0.4585937
time: 3.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4500260, upper bound: 0.4573994
time: 3.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1771002, 1.1783924
1: -8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6935282, 1.6932125
2: -9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3642688, 1.3652406
3: -10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0927925, 1.0924644
4: -5.2452884, -3.9792435, -5.2452884, -3.9792435, -0.9928684, 0.9935489
5: -8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9380527, 0.9383614
6: -12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.2457204, 1.2458496
7: 1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.1358013, 1.1357822
8: -3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.3538308, 1.3539038
9: 0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2805700, 1.2783208

Time for backsubstitution: 8.58 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 1929
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 2528
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 95
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 1978

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1934

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4316621, upper bound: 0.4311216
time: 3.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4316621, upper bound: 0.4311216
time: 3.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1783948, 1.1770983
1: -8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6932077, 1.6935329
2: -9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3652401, 1.3642688
3: -10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0924721, 1.0927849
4: -5.2452884, -3.9792435, -5.2452884, -3.9792435, -0.9935327, 0.9928851
5: -8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9383655, 0.9380486
6: -12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.2458501, 1.2457204
7: 1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.1357656, 1.1358180
8: -3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.3539042, 1.3538308
9: 0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2783179, 1.2805705

Time for backsubstitution: 9.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 2528
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 95
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1929
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 2622

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1499

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4586993, upper bound: 0.4584048
time: 3.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4572146, upper bound: 0.4598901
time: 3.57 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 16.37 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 16.37
Output dim: 7, lower bound: -0.4327019, upper bound: 0.4264956
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 16.37
Output dim: 7, lower bound: -0.4327019, upper bound: 0.4264956
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 16.37
Output dim: 7, lower bound: -0.4488336, upper bound: 0.4585937
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 16.37
Output dim: 7, lower bound: -0.4500260, upper bound: 0.4573994
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 16.37
Output dim: 7, lower bound: -0.4316621, upper bound: 0.4311216
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 16.37
Output dim: 7, lower bound: -0.4316621, upper bound: 0.4311216
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 16.37
Output dim: 7, lower bound: -0.4586993, upper bound: 0.4584048
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 16.37
Output dim: 7, lower bound: -0.4572146, upper bound: 0.4598901

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1666398, 1.1656780
1: -8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6801176, 1.6929569
2: -9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3693147, 1.3686247
3: -10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0824199, 1.0807047
4: -5.2452884, -3.9792435, -5.2452884, -3.9792435, -0.9937105, 0.9942684
5: -8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9386005, 0.9387183
6: -12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.2199173, 1.2198172
7: 1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.1362743, 1.1359825
8: -3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.3511648, 1.3515925
9: 0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2800183, 1.2776518

Time for backsubstitution: 8.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 2250
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 1929
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 95
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 1236

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3118

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4316009, upper bound: 0.4253151
time: 3.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4315337, upper bound: 0.4253862
time: 3.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1672792, 1.1657782
1: -8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6820717, 1.6814270
2: -9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3693552, 1.3686023
3: -10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0825782, 1.0809422
4: -5.2452884, -3.9792435, -5.2452884, -3.9792435, -0.9939089, 0.9937363
5: -8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9386024, 0.9387174
6: -12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.2205839, 1.2203884
7: 1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.1363339, 1.1360860
8: -3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.3512197, 1.3546438
9: 0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2788529, 1.2777691

Time for backsubstitution: 9.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 2250
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 1929
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 95
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 2803

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 170

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4320223, upper bound: 0.4251736
time: 3.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4320223, upper bound: 0.4258696
time: 3.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1638751, 1.1717281
1: -8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6807222, 1.6834745
2: -9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3679094, 1.3681068
3: -10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0793414, 1.0829234
4: -5.2452884, -3.9792435, -5.2452884, -3.9792435, -0.9929819, 0.9901700
5: -8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9319925, 0.9364946
6: -12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.2193265, 1.2199817
7: 1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.1348081, 1.1363912
8: -3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.3542004, 1.3539987
9: 0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2801056, 1.2749491

Time for backsubstitution: 9.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 2250
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 1929
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 95
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 1934

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1846

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4481582, upper bound: 0.4579268
time: 4.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4481582, upper bound: 0.4579268
time: 4.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1657758, 1.1654811
1: -8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6833858, 1.6794024
2: -9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3686433, 1.3686209
3: -10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0809345, 1.0812302
4: -5.2452884, -3.9792435, -5.2452884, -3.9792435, -0.9939518, 0.9929233
5: -8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9366031, 0.9386065
6: -12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.2192144, 1.2211552
7: 1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.1361623, 1.1349630
8: -3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.3546433, 1.3538280
9: 0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2737494, 1.2789693

Time for backsubstitution: 9.41 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 95
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 1929
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 2250
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 761

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 95

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4491026, upper bound: 0.4565609
time: 4.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4491027, upper bound: 0.4565637
time: 4.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1763616, 1.1782932
1: -8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6915741, 1.7027893
2: -9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3642282, 1.3652225
3: -10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0923953, 1.0922256
4: -5.2452884, -3.9792435, -5.2452884, -3.9792435, -0.9926705, 0.9938827
5: -8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9380517, 0.9383609
6: -12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.2444715, 1.2452669
7: 1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.1357431, 1.1356211
8: -3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.3513737, 1.3515015
9: 0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2816195, 1.2782049

Time for backsubstitution: 9.27 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 1929
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 95
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 2528
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 2236

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 761

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4254738, upper bound: 0.4247076
time: 3.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4252480, upper bound: 0.4249334
time: 3.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1770010, 1.1783924
1: -8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6935282, 1.6912584
2: -9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3642688, 1.3652000
3: -10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0925541, 1.0924644
4: -5.2452884, -3.9792435, -5.2452884, -3.9792435, -0.9928684, 0.9933510
5: -8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9380527, 0.9383600
6: -12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.2451377, 1.2458496
7: 1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.1358013, 1.1357241
8: -3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.3514285, 1.3539038
9: 0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2804542, 1.2783208

Time for backsubstitution: 9.33 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 2528
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 1929
type: DSZ, layer: 3, pos: 95
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 604

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 761

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4254738, upper bound: 0.4247076
time: 3.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4252480, upper bound: 0.4249334
time: 3.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1766548, 1.1753616
1: -8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6920490, 1.6922703
2: -9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3656125, 1.3640437
3: -10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0924163, 1.0925059
4: -5.2452884, -3.9792435, -5.2452884, -3.9792435, -0.9919515, 0.9908662
5: -8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9382162, 0.9378970
6: -12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.2468877, 1.2464738
7: 1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.1358633, 1.1358485
8: -3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.3540444, 1.3539066
9: 0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2784944, 1.2806940

Time for backsubstitution: 9.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 95
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 2528
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1929
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 2236

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 170

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4580764, upper bound: 0.4576105
time: 3.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4573187, upper bound: 0.4577741
time: 4.12 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1766582, 1.1753578
1: -8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6919441, 1.6923742
2: -9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3650150, 1.3646412
3: -10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0921931, 1.0927291
4: -5.2452884, -3.9792435, -5.2452884, -3.9792435, -0.9915137, 0.9913039
5: -8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9382133, 0.9378994
6: -12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.2466035, 1.2467580
7: 1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.1357961, 1.1359158
8: -3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.3539796, 1.3539715
9: 0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2784414, 1.2807469

Time for backsubstitution: 9.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 95
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 1929
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 2528
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 1779

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 219

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4468259, upper bound: 0.4504960
time: 4.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4479194, upper bound: 0.4493257
time: 3.94 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 17.20 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 17.20
Output dim: 7, lower bound: -0.4316009, upper bound: 0.4253151
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 17.20
Output dim: 7, lower bound: -0.4315337, upper bound: 0.4253862
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 17.20
Output dim: 7, lower bound: -0.4320223, upper bound: 0.4251736
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 17.20
Output dim: 7, lower bound: -0.4320223, upper bound: 0.4258696
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 17.20
Output dim: 7, lower bound: -0.4481582, upper bound: 0.4579268
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 17.20
Output dim: 7, lower bound: -0.4481582, upper bound: 0.4579268
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 17.20
Output dim: 7, lower bound: -0.4491026, upper bound: 0.4565609
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 17.20
Output dim: 7, lower bound: -0.4491027, upper bound: 0.4565637
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 17.20
Output dim: 7, lower bound: -0.4254738, upper bound: 0.4247076
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 17.20
Output dim: 7, lower bound: -0.4252480, upper bound: 0.4249334
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 17.20
Output dim: 7, lower bound: -0.4254738, upper bound: 0.4247076
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 17.20
Output dim: 7, lower bound: -0.4252480, upper bound: 0.4249334
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 17.20
Output dim: 7, lower bound: -0.4580764, upper bound: 0.4576105
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 17.20
Output dim: 7, lower bound: -0.4573187, upper bound: 0.4577741
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 17.20
Output dim: 7, lower bound: -0.4468259, upper bound: 0.4504960
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 17.20
Output dim: 7, lower bound: -0.4479194, upper bound: 0.4493257

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1552348, 1.1532383
1: -8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6799593, 1.6953888
2: -9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3664083, 1.3640265
3: -10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0859270, 1.0836644
4: -5.2452884, -3.9792435, -5.2452884, -3.9792435, -0.9776425, 0.9778280
5: -8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9323773, 0.9327857
6: -12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.2000861, 1.2015896
7: 1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.1355410, 1.1355090
8: -3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.3291311, 1.3293333
9: 0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2762008, 1.2740598

Time for backsubstitution: 9.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 95
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 1929
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 2250
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1499

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1779

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4278329, upper bound: 0.4253092
time: 3.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4315947, upper bound: 0.4213928
time: 3.91 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1542006, 1.1542726
1: -8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6825485, 1.6927986
2: -9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3647165, 1.3657184
3: -10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0853796, 1.0842118
4: -5.2452884, -3.9792435, -5.2452884, -3.9792435, -0.9772701, 0.9782004
5: -8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9326682, 0.9324949
6: -12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.2016902, 1.1999860
7: 1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.1358008, 1.1352491
8: -3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.3289056, 1.3295588
9: 0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2764263, 1.2738342

Time for backsubstitution: 8.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 2250
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 1929
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 676

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4295744, upper bound: 0.4199538
time: 3.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4262166, upper bound: 0.4235162
time: 3.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1404219, 1.1387882
1: -8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6060638, 1.6149073
2: -9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3667951, 1.3643165
3: -10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0255585, 1.0225780
4: -5.2452884, -3.9792435, -5.2452884, -3.9792435, -1.0099373, 1.0158696
5: -8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9429684, 0.9422917
6: -12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.1853008, 1.1903729
7: 1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.0578437, 1.0604496
8: -3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.3022871, 1.2979732
9: 0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2404156, 1.2426171

Time for backsubstitution: 8.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 95
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 2250
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 1929

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1096

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4269575, upper bound: 0.4211762
time: 3.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4280240, upper bound: 0.4201088
time: 3.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1403146, 1.1388955
1: -8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6150541, 1.6059170
2: -9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3650684, 1.3660488
3: -10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0242820, 1.0240028
4: -5.2452884, -3.9792435, -5.2452884, -3.9792435, -1.0161405, 1.0096660
5: -8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9421768, 0.9430828
6: -12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.1905861, 1.1850839
7: 1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.0607076, 1.0575867
8: -3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.2943430, 1.3059182
9: 0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2436786, 1.2393541

Time for backsubstitution: 8.74 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 95
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 2250
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 1929
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 3118

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1978

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4243496, upper bound: 0.4223518
time: 3.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4285147, upper bound: 0.4181934
time: 3.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1609139, 1.1682978
1: -8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6796169, 1.6822119
2: -9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3668985, 1.3671665
3: -10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0793447, 1.0829253
4: -5.2452884, -3.9792435, -5.2452884, -3.9792435, -0.9796519, 0.9779797
5: -8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9309497, 0.9353132
6: -12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.2194166, 1.2199988
7: 1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.1351376, 1.1367702
8: -3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.3534627, 1.3534517
9: 0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2828293, 1.2776952

Time for backsubstitution: 8.64 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 95
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 2250
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1929
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 1096

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 761

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4416117, upper bound: 0.4515778
time: 4.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4413863, upper bound: 0.4517885
time: 6.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1604447, 1.1687675
1: -8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6794596, 1.6823702
2: -9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3669691, 1.3670959
3: -10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0793428, 1.0829272
4: -5.2452884, -3.9792435, -5.2452884, -3.9792435, -0.9807920, 0.9768395
5: -8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9308109, 0.9354515
6: -12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.2193432, 1.2200718
7: 1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.1351871, 1.1367211
8: -3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.3536530, 1.3532615
9: 0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2828522, 1.2776728

Time for backsubstitution: 8.64 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1929
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2250
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 95
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 604

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4479911, upper bound: 0.4540257
time: 3.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4442632, upper bound: 0.4577585
time: 3.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1702518, 1.1718316
1: -8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6819077, 1.6784711
2: -9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3741856, 1.3748531
3: -10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0722523, 1.0742621
4: -5.2452884, -3.9792435, -5.2452884, -3.9792435, -0.9885249, 0.9883895
5: -8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9280152, 0.9312849
6: -12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.2111225, 1.2151556
7: 1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.1335454, 1.1325378
8: -3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.3406463, 1.3395071
9: 0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2727957, 1.2785788

Time for backsubstitution: 8.73 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 2250
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 1929
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 1978

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 411

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4471139, upper bound: 0.4564328
time: 7.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4489918, upper bound: 0.4553735
time: 3.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1721263, 1.1699567
1: -8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6824541, 1.6779237
2: -9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3748755, 1.3741632
3: -10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0739660, 1.0725479
4: -5.2452884, -3.9792435, -5.2452884, -3.9792435, -0.9894176, 0.9874988
5: -8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9292831, 0.9300179
6: -12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.2132158, 1.2130623
7: 1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.1337371, 1.1323466
8: -3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.3403215, 1.3398309
9: 0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2733583, 1.2780170

Time for backsubstitution: 9.52 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2250
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 1929
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 1779

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4485047, upper bound: 0.4552657
time: 4.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4475302, upper bound: 0.4560377
time: 4.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1749940, 1.1774206
1: -8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6901140, 1.7002726
2: -9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3651476, 1.3603721
3: -10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0920238, 1.0910792
4: -5.2452884, -3.9792435, -5.2452884, -3.9792435, -0.9873524, 0.9901910
5: -8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9339032, 0.9329338
6: -12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.2313185, 1.2353415
7: 1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.1277132, 1.1288753
8: -3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.3506174, 1.3501954
9: 0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2812977, 1.2776313

Time for backsubstitution: 9.52 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 95
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 2528
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 1929

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 219

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4135261, upper bound: 0.4164371
time: 3.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4172022, upper bound: 0.4127691
time: 3.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1754889, 1.1769252
1: -8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6890574, 1.7013292
2: -9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3593779, 1.3661413
3: -10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0912490, 1.0918536
4: -5.2452884, -3.9792435, -5.2452884, -3.9792435, -0.9889793, 0.9885640
5: -8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9326243, 0.9342122
6: -12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.2345467, 1.2321138
7: 1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.1289973, 1.1275911
8: -3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.3500676, 1.3507452
9: 0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2810459, 1.2778831

Time for backsubstitution: 9.53 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 1929
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 2528
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 95
type: DSZ, layer: 3, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1846

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4243000, upper bound: 0.4245232
time: 3.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4247430, upper bound: 0.4241104
time: 3.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1756334, 1.1775198
1: -8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6920681, 1.6887426
2: -9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3651881, 1.3603497
3: -10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0921822, 1.0913181
4: -5.2452884, -3.9792435, -5.2452884, -3.9792435, -0.9875507, 0.9896593
5: -8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9339046, 0.9329329
6: -12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.2319851, 1.2359247
7: 1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.1277719, 1.1289783
8: -3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.3506722, 1.3525982
9: 0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2801323, 1.2777472

Time for backsubstitution: 9.27 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 95
type: DSZ, layer: 3, pos: 1929
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 2528
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 2803

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 95

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4246415, upper bound: 0.4237645
time: 3.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4246377, upper bound: 0.4237361
time: 3.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1761284, 1.1770248
1: -8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6910114, 1.6897984
2: -9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3594184, 1.3661194
3: -10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0914073, 1.0920925
4: -5.2452884, -3.9792435, -5.2452884, -3.9792435, -0.9891777, 0.9880323
5: -8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9326262, 0.9342113
6: -12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.2352128, 1.2326965
7: 1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.1290560, 1.1276941
8: -3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.3501225, 1.3531475
9: 0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2798805, 1.2779994

Time for backsubstitution: 9.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 1929
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 2528
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 95
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 2803

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2622

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4221349, upper bound: 0.4112246
time: 3.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4097288, upper bound: 0.4218550
time: 3.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1506038, 1.1492033
1: -8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6194067, 1.6289387
2: -9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3625326, 1.3592315
3: -10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0360603, 1.0347257
4: -5.2452884, -3.9792435, -5.2452884, -3.9792435, -1.0076332, 1.0127506
5: -8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9425821, 0.9414716
6: -12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.2141542, 1.2190256
7: 1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.0572228, 1.0600719
8: -3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.3055553, 1.2974730
9: 0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2407131, 1.2461762

Time for backsubstitution: 9.32 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 2528
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 95
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 1929
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 2236

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2146

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4495165, upper bound: 0.4487057
time: 4.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4499067, upper bound: 0.4483459
time: 3.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1504965, 1.1493106
1: -8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6283970, 1.6196280
2: -9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3608003, 1.3611131
3: -10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0346360, 1.0361543
4: -5.2452884, -3.9792435, -5.2452884, -3.9792435, -1.0138369, 1.0065479
5: -8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9417906, 0.9422631
6: -12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.2194395, 1.2137403
7: 1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.0600863, 1.0572233
8: -3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.2976108, 1.3054171
9: 0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2439766, 1.2429132

Time for backsubstitution: 8.68 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 1929
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 95
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 2528

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 676

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4555882, upper bound: 0.4522686
time: 4.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4518190, upper bound: 0.4559657
time: 5.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1675353, 1.1682611
1: -8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6787319, 1.6761284
2: -9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3555803, 1.3565154
3: -10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0907707, 1.0916357
4: -5.2452884, -3.9792435, -5.2452884, -3.9792435, -0.9893198, 0.9886961
5: -8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9358130, 0.9359117
6: -12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.2416015, 1.2422280
7: 1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.1099405, 1.1088548
8: -3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.3463712, 1.3487525
9: 0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2649455, 1.2689500

Time for backsubstitution: 9.22 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 95
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 2528
type: DSZ, layer: 3, pos: 1929

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3118

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4457173, upper bound: 0.4493478
time: 4.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4457111, upper bound: 0.4493779
time: 4.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1695614, 1.1662350
1: -8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6756983, 1.6790619
2: -9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3568892, 1.3550577
3: -10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0910997, 1.0913067
4: -5.2452884, -3.9792435, -5.2452884, -3.9792435, -0.9889050, 0.9891109
5: -8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9361811, 0.9354987
6: -12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.2420740, 1.2417555
7: 1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.1087351, 1.1100602
8: -3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.3487611, 1.3463631
9: 0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2666249, 1.2672510

Time for backsubstitution: 9.30 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 2528
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 1929
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 95
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 761

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2236

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4366841, upper bound: 0.4328722
time: 3.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4322078, upper bound: 0.4381330
time: 3.81 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 17.01 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.01
Output dim: 7, lower bound: -0.4278329, upper bound: 0.4253092
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.01
Output dim: 7, lower bound: -0.4315947, upper bound: 0.4213928
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.01
Output dim: 7, lower bound: -0.4295744, upper bound: 0.4199538
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.01
Output dim: 7, lower bound: -0.4262166, upper bound: 0.4235162
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.01
Output dim: 7, lower bound: -0.4269575, upper bound: 0.4211762
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.01
Output dim: 7, lower bound: -0.4280240, upper bound: 0.4201088
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.01
Output dim: 7, lower bound: -0.4243496, upper bound: 0.4223518
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.01
Output dim: 7, lower bound: -0.4285147, upper bound: 0.4181934
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.01
Output dim: 7, lower bound: -0.4416117, upper bound: 0.4515778
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.01
Output dim: 7, lower bound: -0.4413863, upper bound: 0.4517885
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.01
Output dim: 7, lower bound: -0.4479911, upper bound: 0.4540257
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.01
Output dim: 7, lower bound: -0.4442632, upper bound: 0.4577585
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.01
Output dim: 7, lower bound: -0.4471139, upper bound: 0.4564328
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.01
Output dim: 7, lower bound: -0.4489918, upper bound: 0.4553735
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.01
Output dim: 7, lower bound: -0.4485047, upper bound: 0.4552657
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.01
Output dim: 7, lower bound: -0.4475302, upper bound: 0.4560377
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 17.01
Output dim: 7, lower bound: -0.4135261, upper bound: 0.4164371
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 17.01
Output dim: 7, lower bound: -0.4172022, upper bound: 0.4127691
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.01
Output dim: 7, lower bound: -0.4243000, upper bound: 0.4245232
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.01
Output dim: 7, lower bound: -0.4247430, upper bound: 0.4241104
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.01
Output dim: 7, lower bound: -0.4246415, upper bound: 0.4237645
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.01
Output dim: 7, lower bound: -0.4246377, upper bound: 0.4237361
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 17.01
Output dim: 7, lower bound: -0.4221349, upper bound: 0.4112246
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 17.01
Output dim: 7, lower bound: -0.4097288, upper bound: 0.4218550
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.01
Output dim: 7, lower bound: -0.4495165, upper bound: 0.4487057
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.01
Output dim: 7, lower bound: -0.4499067, upper bound: 0.4483459
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.01
Output dim: 7, lower bound: -0.4555882, upper bound: 0.4522686
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.01
Output dim: 7, lower bound: -0.4518190, upper bound: 0.4559657
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.01
Output dim: 7, lower bound: -0.4457173, upper bound: 0.4493478
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.01
Output dim: 7, lower bound: -0.4457111, upper bound: 0.4493779
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.01
Output dim: 7, lower bound: -0.4366841, upper bound: 0.4328722
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.01
Output dim: 7, lower bound: -0.4322078, upper bound: 0.4381330

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1552520, 1.1532669
1: -8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6799679, 1.6953945
2: -9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3664322, 1.3640461
3: -10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0859256, 1.0836630
4: -5.2452884, -3.9792435, -5.2452884, -3.9792435, -0.9776506, 0.9778337
5: -8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9324069, 0.9328210
6: -12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.2001090, 1.2016211
7: 1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.1355467, 1.1355190
8: -3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.3291373, 1.3293419
9: 0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2761755, 1.2740321

Time for backsubstitution: 9.28 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2250
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 1929
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 95
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 2146

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1978

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4201526, upper bound: 0.4217881
time: 3.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4243212, upper bound: 0.4176170
time: 3.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1552634, 1.1532555
1: -8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6799660, 1.6953964
2: -9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3664284, 1.3640499
3: -10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0859256, 1.0836625
4: -5.2452884, -3.9792435, -5.2452884, -3.9792435, -0.9776483, 0.9778361
5: -8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9324126, 0.9328158
6: -12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.2001171, 1.2016125
7: 1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.1355510, 1.1355147
8: -3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.3291397, 1.3293395
9: 0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2761726, 1.2740345

Time for backsubstitution: 9.29 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 95
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 2250
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 1929
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 761

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1978

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4239094, upper bound: 0.4178719
time: 3.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4280824, upper bound: 0.4137081
time: 3.41 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 16.15 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 16.15
Output dim: 7, lower bound: -0.4201526, upper bound: 0.4217881
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 16.15
Output dim: 7, lower bound: -0.4243212, upper bound: 0.4176170
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 16.15
Output dim: 7, lower bound: -0.4239094, upper bound: 0.4178719
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 16.15
Output dim: 7, lower bound: -0.4280824, upper bound: 0.4137081
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.15
Output dim: 7, lower bound: -0.4295744, upper bound: 0.4199538
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.15
Output dim: 7, lower bound: -0.4262166, upper bound: 0.4235162
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.15
Output dim: 7, lower bound: -0.4269575, upper bound: 0.4211762
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.15
Output dim: 7, lower bound: -0.4280240, upper bound: 0.4201088
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.15
Output dim: 7, lower bound: -0.4243496, upper bound: 0.4223518
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.15
Output dim: 7, lower bound: -0.4285147, upper bound: 0.4181934
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.15
Output dim: 7, lower bound: -0.4416117, upper bound: 0.4515778
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.15
Output dim: 7, lower bound: -0.4413863, upper bound: 0.4517885
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.15
Output dim: 7, lower bound: -0.4479911, upper bound: 0.4540257
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.15
Output dim: 7, lower bound: -0.4442632, upper bound: 0.4577585
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.15
Output dim: 7, lower bound: -0.4471139, upper bound: 0.4564328
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.15
Output dim: 7, lower bound: -0.4489918, upper bound: 0.4553735
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.15
Output dim: 7, lower bound: -0.4485047, upper bound: 0.4552657
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.15
Output dim: 7, lower bound: -0.4475302, upper bound: 0.4560377
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.15
Output dim: 7, lower bound: -0.4243000, upper bound: 0.4245232
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.15
Output dim: 7, lower bound: -0.4247430, upper bound: 0.4241104
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.15
Output dim: 7, lower bound: -0.4246415, upper bound: 0.4237645
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.15
Output dim: 7, lower bound: -0.4246377, upper bound: 0.4237361
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.15
Output dim: 7, lower bound: -0.4495165, upper bound: 0.4487057
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.15
Output dim: 7, lower bound: -0.4499067, upper bound: 0.4483459
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.15
Output dim: 7, lower bound: -0.4555882, upper bound: 0.4522686
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.15
Output dim: 7, lower bound: -0.4518190, upper bound: 0.4559657
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.15
Output dim: 7, lower bound: -0.4457173, upper bound: 0.4493478
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.15
Output dim: 7, lower bound: -0.4457111, upper bound: 0.4493779
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.15
Output dim: 7, lower bound: -0.4366841, upper bound: 0.4328722
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.15
Output dim: 7, lower bound: -0.4322078, upper bound: 0.4381330

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.52 + 547.21 = 604.73 seconds
