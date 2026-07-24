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
execution time: IAR + RelationalAnalysis = 22.90 + 34.08 = 56.98 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.4604159, upper bound: 0.4604158

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 1929
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 95
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 2528
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 634
type: DSZ, layer: 3, pos: 2250
type: DSZ, layer: 3, pos: 1779

Time for candidate selection: 0.43 seconds

### Candidate
type: DSZ, layer: 3, pos: 170

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4597930, upper bound: 0.4590513
time: 3.78 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4590513, upper bound: 0.4597930
time: 3.75 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 7.97 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 7.97
Output dim: 7, lower bound: -0.4597930, upper bound: 0.4590513
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 7.97
Output dim: 7, lower bound: -0.4590513, upper bound: 0.4597930

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1556692, 1.1555619
1: -8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6191182, 1.6281080
2: -9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3643808, 1.3626485
3: -10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0368867, 1.0354619
4: -5.2452884, -3.9792435, -5.2452884, -3.9792435, -1.0140743, 1.0202775
5: -8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9426060, 0.9418147
6: -12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.2133236, 1.2186089
7: 1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.0570621, 1.0599256
8: -3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.3054838, 1.2975397
9: 0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2465277, 1.2497907

Time for backsubstitution: 8.28 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1929
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 95
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 2528
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 634
type: DSZ, layer: 3, pos: 2250
type: DSZ, layer: 3, pos: 1779

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 1929

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4574152, upper bound: 0.4563002
time: 4.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4570461, upper bound: 0.4566703
time: 4.12 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1555619, 1.1556692
1: -8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6281085, 1.6191177
2: -9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3626485, 1.3643808
3: -10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0354624, 1.0368867
4: -5.2452884, -3.9792435, -5.2452884, -3.9792435, -1.0202780, 1.0140743
5: -8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9418144, 0.9426062
6: -12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.2186089, 1.2133236
7: 1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.0599256, 1.0570621
8: -3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.2975397, 1.3054838
9: 0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2497907, 1.2465277

Time for backsubstitution: 8.30 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1929
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 95
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 2528
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 634
type: DSZ, layer: 3, pos: 2250
type: DSZ, layer: 3, pos: 1779

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 3, pos: 1929

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4566700, upper bound: 0.4570459
time: 8.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4563003, upper bound: 0.4574170
time: 4.00 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 21.50 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 21.50
Output dim: 7, lower bound: -0.4574152, upper bound: 0.4563002
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 21.50
Output dim: 7, lower bound: -0.4570461, upper bound: 0.4566703
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 21.50
Output dim: 7, lower bound: -0.4566700, upper bound: 0.4570459
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 21.50
Output dim: 7, lower bound: -0.4563003, upper bound: 0.4574170

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1532736, 1.1540041
1: -8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6163187, 1.6255202
2: -9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3641171, 1.3626165
3: -10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0348496, 1.0325601
4: -5.2452884, -3.9792435, -5.2452884, -3.9792435, -1.0131431, 1.0195746
5: -8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9415512, 0.9377847
6: -12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.2131667, 1.2183037
7: 1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.0565009, 1.0589466
8: -3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.3041248, 1.2930918
9: 0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2461667, 1.2495360

Time for backsubstitution: 8.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 95
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 2528
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 634
type: DSZ, layer: 3, pos: 2250
type: DSZ, layer: 3, pos: 1779

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 1206

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4152441, upper bound: 0.4149822
time: 3.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4152441, upper bound: 0.4149822
time: 3.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1541109, 1.1531663
1: -8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6165295, 1.6253095
2: -9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3643489, 1.3623843
3: -10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0339847, 1.0334246
4: -5.2452884, -3.9792435, -5.2452884, -3.9792435, -1.0133715, 1.0193467
5: -8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9385762, 0.9407597
6: -12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.2130184, 1.2184515
7: 1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.0560832, 1.0593648
8: -3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.3010359, 1.2961802
9: 0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2462726, 1.2494297

Time for backsubstitution: 9.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 95
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 2528
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 634
type: DSZ, layer: 3, pos: 2250
type: DSZ, layer: 3, pos: 1779

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 3, pos: 1206

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4148694, upper bound: 0.4153575
time: 3.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4148694, upper bound: 0.4153575
time: 3.87 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1531663, 1.1541109
1: -8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6253090, 1.6165299
2: -9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3623843, 1.3643489
3: -10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0334249, 1.0339844
4: -5.2452884, -3.9792435, -5.2452884, -3.9792435, -1.0193467, 1.0133715
5: -8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9407597, 0.9385762
6: -12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.2184515, 1.2130184
7: 1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.0593648, 1.0560832
8: -3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.2961802, 1.3010359
9: 0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2494297, 1.2462726

Time for backsubstitution: 9.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 95
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 2528
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 634
type: DSZ, layer: 3, pos: 2250
type: DSZ, layer: 3, pos: 1779

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 3, pos: 1206

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4153563, upper bound: 0.4148704
time: 3.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4153563, upper bound: 0.4148704
time: 3.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1540041, 1.1532736
1: -8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6255207, 1.6163187
2: -9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3626161, 1.3641171
3: -10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0325599, 1.0348494
4: -5.2452884, -3.9792435, -5.2452884, -3.9792435, -1.0195746, 1.0131431
5: -8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9377851, 0.9415507
6: -12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.2183037, 1.2131662
7: 1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.0589466, 1.0565009
8: -3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.2930918, 1.3041248
9: 0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2495360, 1.2461667

Time for backsubstitution: 8.57 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 227
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 1741
type: DSZ, layer: 3, pos: 1096
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1499
type: DSZ, layer: 3, pos: 95
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 2528
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 604
type: DSZ, layer: 3, pos: 634
type: DSZ, layer: 3, pos: 2250
type: DSZ, layer: 3, pos: 1779

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 1206

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4149816, upper bound: 0.4152451
time: 3.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4149816, upper bound: 0.4152451
time: 3.79 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 16.23 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 16.23
Output dim: 7, lower bound: -0.4152441, upper bound: 0.4149822
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 16.23
Output dim: 7, lower bound: -0.4152441, upper bound: 0.4149822
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 16.23
Output dim: 7, lower bound: -0.4148694, upper bound: 0.4153575
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 16.23
Output dim: 7, lower bound: -0.4148694, upper bound: 0.4153575
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 16.23
Output dim: 7, lower bound: -0.4153563, upper bound: 0.4148704
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 16.23
Output dim: 7, lower bound: -0.4153563, upper bound: 0.4148704
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 16.23
Output dim: 7, lower bound: -0.4149816, upper bound: 0.4152451
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 16.23
Output dim: 7, lower bound: -0.4149816, upper bound: 0.4152451

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 56.98 + 112.67 = 169.66 seconds
