## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.11490763


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-5.4925499, -5.1446381, -5.4925499, -5.1446381, -0.2869463, 0.2869463)
1: (-1.9263756, -1.6647363, -1.9263756, -1.6647363, -0.1361805, 0.1361805)
2: (7.6529355, 8.0298309, 7.6529355, 8.0298309, -0.3099370, 0.3099370)
3: (-3.1874378, -2.7689991, -3.1874378, -2.7689991, -0.1867815, 0.1867815)
4: (-10.6161385, -10.1756039, -10.6161385, -10.1756039, -0.1977763, 0.1977763)
5: (-8.4546127, -7.8914390, -8.4546127, -7.8914390, -0.3067017, 0.3067017)
6: (-7.3124766, -6.8582253, -7.3124766, -6.8582253, -0.2110730, 0.2110729)
7: (-5.2929306, -4.8406897, -5.2929306, -4.8406897, -0.2082946, 0.2082946)
8: (-0.9866056, -0.7281651, -0.9866056, -0.7281651, -0.1505284, 0.1505284)
9: (-11.4172773, -11.0448074, -11.4172773, -11.0448074, -0.1758047, 0.1758049)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.36 + 33.41 = 55.77 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.1209554, upper bound: 0.1209554

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 214
type: B, layer: 3, pos: 214
type: A, layer: 3, pos: 1110
type: B, layer: 3, pos: 1110
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 60
type: A, layer: 3, pos: 2494
type: B, layer: 3, pos: 2494
type: A, layer: 3, pos: 753
type: B, layer: 3, pos: 753
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1262
type: B, layer: 3, pos: 1262
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 2322
type: B, layer: 3, pos: 2322
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2466
type: B, layer: 3, pos: 2466
type: A, layer: 3, pos: 2928
type: B, layer: 3, pos: 2928

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 214

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1184842, upper bound: 0.1190632
time: 3.00 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1190629, upper bound: 0.1190629
time: 3.48 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 6.76 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 6.76
Output dim: 2, lower bound: -0.1184842, upper bound: 0.1190632
NS_A2, status: Status.UNKNOWN, split count: 1, time: 6.76
Output dim: 2, lower bound: -0.1190629, upper bound: 0.1190629

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -5.4869790, -5.1501546, -5.4925399, -5.1465549, -0.2827702, 0.2822032
1: -1.9258332, -1.6651736, -1.9263654, -1.6648858, -0.1348236, 0.1353929
2: 7.6579580, 8.0296555, 7.6548862, 8.0298271, -0.3018851, 0.3010042
3: -3.1885664, -2.7705901, -3.1874204, -2.7695522, -0.1850231, 0.1848717
4: -10.6142282, -10.1757317, -10.6156120, -10.1756039, -0.1970332, 0.1997924
5: -8.4579487, -7.8962078, -8.4546089, -7.8930473, -0.3039894, 0.3017385
6: -7.3161612, -6.8633852, -7.3124762, -6.8599772, -0.1967151, 0.1986390
7: -5.2884808, -4.8389292, -5.2913971, -4.8406897, -0.2032285, 0.2054286
8: -0.9864128, -0.7281065, -0.9865413, -0.7281625, -0.1500758, 0.1499943
9: -11.4136963, -11.0473328, -11.4160032, -11.0448151, -0.1722695, 0.1711693

Time for backsubstitution: 7.81 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 214
type: B, layer: 3, pos: 1110
type: A, layer: 3, pos: 1110
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 60
type: A, layer: 3, pos: 2494
type: B, layer: 3, pos: 2494
type: A, layer: 3, pos: 753
type: B, layer: 3, pos: 753
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1262
type: B, layer: 3, pos: 1262
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 2322
type: B, layer: 3, pos: 2322
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2466
type: B, layer: 3, pos: 2466
type: B, layer: 3, pos: 2928
type: A, layer: 3, pos: 2928

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 214

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1182947, upper bound: 0.1182947
time: 3.04 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1182947, upper bound: 0.1190632
time: 3.04 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -5.4925475, -5.1458297, -5.4925480, -5.1450992, -0.2861762, 0.2834275
1: -1.9263735, -1.6650403, -1.9263753, -1.6648309, -0.1361756, 0.1348423
2: 7.6561766, 8.0298290, 7.6540060, 8.0298290, -0.2973373, 0.3097968
3: -3.1874356, -2.7713082, -3.1874385, -2.7697256, -0.1864038, 0.1839920
4: -10.6151552, -10.1756029, -10.6158333, -10.1756020, -0.2003489, 0.1971273
5: -8.4546108, -7.8977880, -8.4546118, -7.8934216, -0.3054247, 0.3008952
6: -7.3124762, -6.8651123, -7.3124762, -6.8603725, -0.2109499, 0.1874121
7: -5.2883997, -4.8406901, -5.2912922, -4.8406897, -0.2031903, 0.2072399
8: -0.9863527, -0.7281637, -0.9865258, -0.7281651, -0.1496838, 0.1505207
9: -11.4168367, -11.0448074, -11.4171391, -11.0448074, -0.1699249, 0.1757841

Time for backsubstitution: 7.75 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 214
type: A, layer: 3, pos: 1110
type: B, layer: 3, pos: 1110
type: B, layer: 3, pos: 60
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 2494
type: B, layer: 3, pos: 753
type: A, layer: 3, pos: 753
type: A, layer: 3, pos: 2494
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 1262
type: A, layer: 3, pos: 1262
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 2322
type: A, layer: 3, pos: 2322
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2466
type: A, layer: 3, pos: 2466
type: A, layer: 3, pos: 2928
type: B, layer: 3, pos: 2928

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 214

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1190633, upper bound: 0.1182947
time: 3.16 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1190633, upper bound: 0.1190632
time: 3.02 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 14.09 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 14.09
Output dim: 2, lower bound: -0.1182947, upper bound: 0.1182947
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 14.09
Output dim: 2, lower bound: -0.1182947, upper bound: 0.1190632
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 14.09
Output dim: 2, lower bound: -0.1190633, upper bound: 0.1182947
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 14.09
Output dim: 2, lower bound: -0.1190633, upper bound: 0.1190632

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -5.4869790, -5.1501546, -5.4869790, -5.1501546, -0.2796993, 0.2796993
1: -1.9258332, -1.6651736, -1.9258332, -1.6651736, -0.1343160, 0.1343160
2: 7.6579580, 8.0296555, 7.6579580, 8.0296555, -0.2959852, 0.2959855
3: -3.1885664, -2.7705901, -3.1885664, -2.7705901, -0.1839373, 0.1839373
4: -10.6142282, -10.1757317, -10.6142282, -10.1757317, -0.1992061, 0.1992064
5: -8.4579487, -7.8962078, -8.4579487, -7.8962078, -0.3015563, 0.3015563
6: -7.3161612, -6.8633852, -7.3161612, -6.8633852, -0.1887400, 0.1887400
7: -5.2884808, -4.8389292, -5.2884808, -4.8389292, -0.2026403, 0.2026403
8: -0.9864128, -0.7281065, -0.9864128, -0.7281065, -0.1496990, 0.1496990
9: -11.4136963, -11.0473328, -11.4136963, -11.0473328, -0.1689340, 0.1689340

Time for backsubstitution: 8.03 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1110
type: B, layer: 3, pos: 1110
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 60
type: A, layer: 3, pos: 2494
type: B, layer: 3, pos: 2494
type: A, layer: 3, pos: 753
type: B, layer: 3, pos: 753
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1262
type: B, layer: 3, pos: 1262
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 2322
type: B, layer: 3, pos: 2322
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2466
type: B, layer: 3, pos: 2466
type: A, layer: 3, pos: 2928
type: B, layer: 3, pos: 2928

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 1110

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1173682, upper bound: 0.1181368
time: 3.30 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1178253, upper bound: 0.1176394
time: 3.07 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -5.4869790, -5.1501546, -5.4925475, -5.1458297, -0.2820148, 0.2822018
1: -1.9258332, -1.6651736, -1.9263735, -1.6650403, -0.1350815, 0.1354029
2: 7.6579580, 8.0296555, 7.6561766, 8.0298290, -0.3018875, 0.3035901
3: -3.1885664, -2.7705901, -3.1874356, -2.7713082, -0.1848620, 0.1848786
4: -10.6142282, -10.1757317, -10.6151552, -10.1756029, -0.1970367, 0.1980662
5: -8.4579487, -7.8962078, -8.4546108, -7.8977880, -0.3026481, 0.3016834
6: -7.3161612, -6.8633852, -7.3124762, -6.8651123, -0.2009621, 0.1986126
7: -5.2884808, -4.8389292, -5.2883997, -4.8406901, -0.2031972, 0.2048326
8: -0.9864128, -0.7281065, -0.9863527, -0.7281637, -0.1500762, 0.1501273
9: -11.4136963, -11.0473328, -11.4168367, -11.0448074, -0.1722747, 0.1724058

Time for backsubstitution: 8.58 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1110
type: B, layer: 3, pos: 1110
type: B, layer: 3, pos: 60
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 2494
type: A, layer: 3, pos: 2494
type: B, layer: 3, pos: 753
type: A, layer: 3, pos: 753
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 1262
type: A, layer: 3, pos: 1262
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 2322
type: A, layer: 3, pos: 2322
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2466
type: A, layer: 3, pos: 2466
type: A, layer: 3, pos: 2928
type: B, layer: 3, pos: 2928

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 1110

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1173682, upper bound: 0.1189053
time: 2.94 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1178253, upper bound: 0.1184042
time: 3.34 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -5.4925475, -5.1458297, -5.4869790, -5.1501546, -0.2822018, 0.2820148
1: -1.9263735, -1.6650403, -1.9258332, -1.6651736, -0.1354028, 0.1350815
2: 7.6561766, 8.0298290, 7.6579580, 8.0296555, -0.3035898, 0.3018880
3: -3.1874356, -2.7713082, -3.1885664, -2.7705901, -0.1848786, 0.1848620
4: -10.6151552, -10.1756029, -10.6142282, -10.1757317, -0.1980665, 0.1970367
5: -8.4546108, -7.8977880, -8.4579487, -7.8962078, -0.3016834, 0.3026481
6: -7.3124762, -6.8651123, -7.3161612, -6.8633852, -0.1986126, 0.2009621
7: -5.2883997, -4.8406901, -5.2884808, -4.8389292, -0.2048326, 0.2031972
8: -0.9863527, -0.7281637, -0.9864128, -0.7281065, -0.1501273, 0.1500762
9: -11.4168367, -11.0448074, -11.4136963, -11.0473328, -0.1724060, 0.1722747

Time for backsubstitution: 8.57 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1110
type: A, layer: 3, pos: 1110
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 60
type: A, layer: 3, pos: 2494
type: B, layer: 3, pos: 2494
type: A, layer: 3, pos: 753
type: B, layer: 3, pos: 753
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1262
type: B, layer: 3, pos: 1262
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 2322
type: B, layer: 3, pos: 2322
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2466
type: B, layer: 3, pos: 2466
type: B, layer: 3, pos: 2928
type: A, layer: 3, pos: 2928

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 1110

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1189049, upper bound: 0.1171783
time: 3.28 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1184042, upper bound: 0.1176396
time: 3.23 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -5.4925475, -5.1458297, -5.4925475, -5.1458297, -0.2834167, 0.2834167
1: -1.9263735, -1.6650403, -1.9263735, -1.6650403, -0.1348411, 0.1348412
2: 7.6561766, 8.0298290, 7.6561766, 8.0298290, -0.2973371, 0.2973371
3: -3.1874356, -2.7713082, -3.1874356, -2.7713082, -0.1839904, 0.1839904
4: -10.6151552, -10.1756029, -10.6151552, -10.1756029, -0.2003474, 0.2003474
5: -8.4546108, -7.8977880, -8.4546108, -7.8977880, -0.3008776, 0.3008776
6: -7.3124762, -6.8651123, -7.3124762, -6.8651123, -0.1873971, 0.1873971
7: -5.2883997, -4.8406901, -5.2883997, -4.8406901, -0.2031817, 0.2031817
8: -0.9863527, -0.7281637, -0.9863527, -0.7281637, -0.1496838, 0.1496838
9: -11.4168367, -11.0448074, -11.4168367, -11.0448074, -0.1699240, 0.1699240

Time for backsubstitution: 7.94 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1110
type: B, layer: 3, pos: 1110
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 60
type: A, layer: 3, pos: 2494
type: B, layer: 3, pos: 2494
type: A, layer: 3, pos: 753
type: B, layer: 3, pos: 753
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1262
type: B, layer: 3, pos: 1262
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 2322
type: B, layer: 3, pos: 2322
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2466
type: B, layer: 3, pos: 2466
type: A, layer: 3, pos: 2928
type: B, layer: 3, pos: 2928

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 1110

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1179466, upper bound: 0.1181366
time: 2.97 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1184043, upper bound: 0.1176396
time: 3.06 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 14.12 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 14.12
Output dim: 2, lower bound: -0.1173682, upper bound: 0.1181368
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 14.12
Output dim: 2, lower bound: -0.1178253, upper bound: 0.1176394
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 14.12
Output dim: 2, lower bound: -0.1173682, upper bound: 0.1189053
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 14.12
Output dim: 2, lower bound: -0.1178253, upper bound: 0.1184042
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 14.12
Output dim: 2, lower bound: -0.1189049, upper bound: 0.1171783
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 14.12
Output dim: 2, lower bound: -0.1184042, upper bound: 0.1176396
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 14.12
Output dim: 2, lower bound: -0.1179466, upper bound: 0.1181366
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 14.12
Output dim: 2, lower bound: -0.1184043, upper bound: 0.1176396

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -5.4786925, -5.1492414, -5.4843216, -5.1501546, -0.2673397, 0.2707384
1: -1.9241651, -1.6635351, -1.9252852, -1.6651884, -0.1330577, 0.1360909
2: 7.6575255, 8.0292883, 7.6581030, 8.0295353, -0.2968788, 0.2920990
3: -3.1857047, -2.7699382, -3.1876724, -2.7705901, -0.1797338, 0.1808977
4: -10.6110401, -10.1850424, -10.6142292, -10.1788139, -0.1925893, 0.1900306
5: -8.4504519, -7.8972983, -8.4555511, -7.8962088, -0.2922692, 0.2949052
6: -7.3081980, -6.8665380, -7.3135910, -6.8633871, -0.1815015, 0.1834463
7: -5.2862792, -4.8471990, -5.2884808, -4.8415880, -0.1962130, 0.1938777
8: -0.9848564, -0.7274609, -0.9859872, -0.7281079, -0.1474696, 0.1479646
9: -11.4134865, -11.0449066, -11.4136286, -11.0473804, -0.1673026, 0.1704726

Time for backsubstitution: 8.02 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1110
type: B, layer: 3, pos: 2494
type: B, layer: 3, pos: 60
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 753
type: B, layer: 3, pos: 753
type: A, layer: 3, pos: 2494
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1262
type: A, layer: 3, pos: 1262
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2322
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 2322
type: A, layer: 3, pos: 2466
type: B, layer: 3, pos: 2466
type: B, layer: 3, pos: 2928
type: A, layer: 3, pos: 2928

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 1110

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1175073, upper bound: 0.1175072
time: 3.48 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1175073, upper bound: 0.1179685
time: 3.21 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -5.4855371, -5.1501546, -5.4868965, -5.1501546, -0.2686570, 0.2796936
1: -1.9249485, -1.6651742, -1.9257889, -1.6651739, -0.1365710, 0.1340789
2: 7.6579580, 8.0291843, 7.6579571, 8.0296345, -0.2955215, 0.3033652
3: -3.1879721, -2.7705901, -3.1885355, -2.7705901, -0.1807437, 0.1839352
4: -10.6142292, -10.1759052, -10.6142282, -10.1757421, -0.1992006, 0.1906245
5: -8.4573936, -7.8962078, -8.4579201, -7.8962069, -0.2950089, 0.3015549
6: -7.3160810, -6.8633862, -7.3161559, -6.8633871, -0.1825522, 0.1887355
7: -5.2884812, -4.8394351, -5.2884812, -4.8389540, -0.2026358, 0.1948638
8: -0.9859877, -0.7281075, -0.9863915, -0.7281070, -0.1487010, 0.1496973
9: -11.4126415, -11.0473337, -11.4136429, -11.0473337, -0.1701334, 0.1686741

Time for backsubstitution: 8.51 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1110
type: B, layer: 3, pos: 2494
type: B, layer: 3, pos: 60
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 753
type: B, layer: 3, pos: 753
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1262
type: A, layer: 3, pos: 1262
type: A, layer: 3, pos: 2494
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2322
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 2466
type: B, layer: 3, pos: 2466
type: A, layer: 3, pos: 2322
type: B, layer: 3, pos: 2928
type: A, layer: 3, pos: 2928

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1110

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1179688, upper bound: 0.1175074
time: 3.73 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1179688, upper bound: 0.1179687
time: 3.09 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -5.4786925, -5.1492414, -5.4898939, -5.1458297, -0.2696552, 0.2732444
1: -1.9241651, -1.6635351, -1.9258264, -1.6650553, -0.1338230, 0.1371777
2: 7.6575255, 8.0292883, 7.6563234, 8.0297079, -0.3027811, 0.2997038
3: -3.1857047, -2.7699382, -3.1865416, -2.7713082, -0.1806586, 0.1818421
4: -10.6110401, -10.1850424, -10.6151562, -10.1786833, -0.1904199, 0.1888909
5: -8.4504519, -7.8972983, -8.4522171, -7.8977861, -0.2933614, 0.2950344
6: -7.3081980, -6.8665380, -7.3099060, -6.8651099, -0.1937234, 0.1933187
7: -5.2862792, -4.8471990, -5.2884002, -4.8433495, -0.1967723, 0.1960695
8: -0.9848564, -0.7274609, -0.9859271, -0.7281635, -0.1478466, 0.1483928
9: -11.4134865, -11.0449066, -11.4167681, -11.0448570, -0.1705887, 0.1739445

Time for backsubstitution: 7.93 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1110
type: B, layer: 3, pos: 2494
type: B, layer: 3, pos: 60
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 753
type: A, layer: 3, pos: 753
type: A, layer: 3, pos: 2494
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 1262
type: A, layer: 3, pos: 1262
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2322
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 2322
type: A, layer: 3, pos: 2466
type: B, layer: 3, pos: 2466
type: A, layer: 3, pos: 2928
type: B, layer: 3, pos: 2928

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 1110

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1173681, upper bound: 0.1179469
time: 3.29 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1173681, upper bound: 0.1184042
time: 3.07 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -5.4855371, -5.1501546, -5.4924669, -5.1458297, -0.2709742, 0.2821960
1: -1.9249485, -1.6651742, -1.9263299, -1.6650410, -0.1372917, 0.1351684
2: 7.6579580, 8.0291843, 7.6561775, 8.0298061, -0.3014240, 0.3109689
3: -3.1879721, -2.7705901, -3.1874058, -2.7713082, -0.1816686, 0.1848766
4: -10.6142292, -10.1759052, -10.6151552, -10.1756105, -0.1970310, 0.1894851
5: -8.4573936, -7.8962078, -8.4545841, -7.8977880, -0.2961011, 0.3016820
6: -7.3160810, -6.8633862, -7.3124709, -6.8651123, -0.1947737, 0.1986078
7: -5.2884812, -4.8394351, -5.2883997, -4.8407145, -0.2031932, 0.1970561
8: -0.9859877, -0.7281075, -0.9863322, -0.7281637, -0.1490781, 0.1501257
9: -11.4126415, -11.0473337, -11.4167833, -11.0448084, -0.1735258, 0.1721460

Time for backsubstitution: 7.99 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1110
type: B, layer: 3, pos: 2494
type: B, layer: 3, pos: 60
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 753
type: B, layer: 3, pos: 753
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1262
type: A, layer: 3, pos: 1262
type: A, layer: 3, pos: 2494
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2322
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 2466
type: B, layer: 3, pos: 2466
type: A, layer: 3, pos: 2322
type: B, layer: 3, pos: 2928
type: A, layer: 3, pos: 2928

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 1110

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1178253, upper bound: 0.1179468
time: 3.21 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1178253, upper bound: 0.1184043
time: 3.37 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -5.4898939, -5.1458297, -5.4786925, -5.1492414, -0.2732444, 0.2696552
1: -1.9258264, -1.6650553, -1.9241651, -1.6635351, -0.1371778, 0.1338230
2: 7.6563234, 8.0297079, 7.6575255, 8.0292883, -0.2997041, 0.3027809
3: -3.1865416, -2.7713082, -3.1857047, -2.7699382, -0.1818421, 0.1806586
4: -10.6151562, -10.1786833, -10.6110401, -10.1850424, -0.1888909, 0.1904197
5: -8.4522171, -7.8977861, -8.4504519, -7.8972983, -0.2950344, 0.2933614
6: -7.3099060, -6.8651099, -7.3081980, -6.8665380, -0.1933186, 0.1937234
7: -5.2884002, -4.8433495, -5.2862792, -4.8471990, -0.1960695, 0.1967723
8: -0.9859271, -0.7281635, -0.9848564, -0.7274609, -0.1483928, 0.1478466
9: -11.4167681, -11.0448570, -11.4134865, -11.0449066, -0.1739445, 0.1705887

Time for backsubstitution: 8.09 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1110
type: A, layer: 3, pos: 2494
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 60
type: A, layer: 3, pos: 753
type: B, layer: 3, pos: 753
type: B, layer: 3, pos: 2494
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1262
type: B, layer: 3, pos: 1262
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 2322
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 2322
type: B, layer: 3, pos: 2466
type: A, layer: 3, pos: 2466
type: B, layer: 3, pos: 2928
type: A, layer: 3, pos: 2928

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 1110

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1179470, upper bound: 0.1173681
time: 3.33 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1179470, upper bound: 0.1173680
time: 3.69 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -5.4924669, -5.1458297, -5.4855371, -5.1501546, -0.2821960, 0.2709739
1: -1.9263299, -1.6650410, -1.9249485, -1.6651742, -0.1351684, 0.1372917
2: 7.6561775, 8.0298061, 7.6579580, 8.0291843, -0.3109684, 0.3014243
3: -3.1874058, -2.7713082, -3.1879721, -2.7705901, -0.1848764, 0.1816686
4: -10.6151552, -10.1756105, -10.6142292, -10.1759052, -0.1894853, 0.1970310
5: -8.4545841, -7.8977880, -8.4573936, -7.8962078, -0.3016822, 0.2961011
6: -7.3124709, -6.8651123, -7.3160810, -6.8633862, -0.1986078, 0.1947736
7: -5.2883997, -4.8407145, -5.2884812, -4.8394351, -0.1970561, 0.2031932
8: -0.9863322, -0.7281637, -0.9859877, -0.7281075, -0.1501257, 0.1490781
9: -11.4167833, -11.0448084, -11.4126415, -11.0473337, -0.1721458, 0.1735259

Time for backsubstitution: 8.93 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1110
type: A, layer: 3, pos: 2494
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 753
type: A, layer: 3, pos: 753
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1262
type: B, layer: 3, pos: 1262
type: B, layer: 3, pos: 2494
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 2322
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 2466
type: A, layer: 3, pos: 2466
type: B, layer: 3, pos: 2322
type: A, layer: 3, pos: 2928
type: B, layer: 3, pos: 2928

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 1110

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1179465, upper bound: 0.1178253
time: 3.31 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1179470, upper bound: 0.1178253
time: 3.55 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5.4842277, -5.1451063, -5.4898939, -5.1458297, -0.2710457, 0.2744570
1: -1.9247060, -1.6634014, -1.9258264, -1.6650553, -0.1335634, 0.1366144
2: 7.6557417, 8.0294628, 7.6563234, 8.0297079, -0.2982290, 0.2934504
3: -3.1845701, -2.7706549, -3.1865416, -2.7713082, -0.1797864, 0.1809506
4: -10.6119690, -10.1849117, -10.6151562, -10.1786833, -0.1937418, 0.1911716
5: -8.4471102, -7.8988752, -8.4522171, -7.8977861, -0.2915905, 0.2942266
6: -7.3045149, -6.8682666, -7.3099060, -6.8651099, -0.1801581, 0.1821082
7: -5.2861967, -4.8489618, -5.2884002, -4.8433495, -0.1967556, 0.1944191
8: -0.9847944, -0.7275162, -0.9859271, -0.7281635, -0.1474551, 0.1479498
9: -11.4166288, -11.0423851, -11.4167681, -11.0448570, -0.1682912, 0.1714501

Time for backsubstitution: 8.19 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1110
type: B, layer: 3, pos: 2494
type: B, layer: 3, pos: 60
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 753
type: B, layer: 3, pos: 753
type: A, layer: 3, pos: 2494
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1262
type: A, layer: 3, pos: 1262
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2322
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 2322
type: A, layer: 3, pos: 2466
type: B, layer: 3, pos: 2466
type: B, layer: 3, pos: 2928
type: A, layer: 3, pos: 2928

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 3, pos: 1110

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1179470, upper bound: 0.1171786
time: 3.75 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1179470, upper bound: 0.1176396
time: 3.73 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -5.4911098, -5.1458297, -5.4924669, -5.1458297, -0.2723761, 0.2834115
1: -1.9254895, -1.6650400, -1.9263299, -1.6650410, -0.1370708, 0.1346068
2: 7.6561766, 8.0293598, 7.6561775, 8.0298061, -0.2968736, 0.3047156
3: -3.1868405, -2.7713082, -3.1874058, -2.7713082, -0.1807967, 0.1839882
4: -10.6151552, -10.1757765, -10.6151552, -10.1756105, -0.2003422, 0.1917667
5: -8.4540558, -7.8977876, -8.4545841, -7.8977880, -0.2943301, 0.3008759
6: -7.3123951, -6.8651137, -7.3124709, -6.8651123, -0.1812094, 0.1873926
7: -5.2884007, -4.8411946, -5.2883997, -4.8407145, -0.2031772, 0.1954057
8: -0.9859285, -0.7281630, -0.9863322, -0.7281637, -0.1486861, 0.1496828
9: -11.4157810, -11.0448084, -11.4167833, -11.0448084, -0.1711124, 0.1696500

Time for backsubstitution: 8.26 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1110
type: B, layer: 3, pos: 2494
type: B, layer: 3, pos: 60
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 753
type: B, layer: 3, pos: 753
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1262
type: A, layer: 3, pos: 1262
type: A, layer: 3, pos: 2494
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2322
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 2466
type: B, layer: 3, pos: 2466
type: A, layer: 3, pos: 2322
type: B, layer: 3, pos: 2928
type: A, layer: 3, pos: 2928

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 3, pos: 1110

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1184043, upper bound: 0.1171786
time: 3.26 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1184043, upper bound: 0.1176396
time: 3.66 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 15.38 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.38
Output dim: 2, lower bound: -0.1175073, upper bound: 0.1175072
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.38
Output dim: 2, lower bound: -0.1175073, upper bound: 0.1179685
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.38
Output dim: 2, lower bound: -0.1179688, upper bound: 0.1175074
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.38
Output dim: 2, lower bound: -0.1179688, upper bound: 0.1179687
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.38
Output dim: 2, lower bound: -0.1173681, upper bound: 0.1179469
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.38
Output dim: 2, lower bound: -0.1173681, upper bound: 0.1184042
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.38
Output dim: 2, lower bound: -0.1178253, upper bound: 0.1179468
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.38
Output dim: 2, lower bound: -0.1178253, upper bound: 0.1184043
NS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 15.38
Output dim: 2, lower bound: -0.1179470, upper bound: 0.1173681
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 15.38
Output dim: 2, lower bound: -0.1179470, upper bound: 0.1173680
NS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 15.38
Output dim: 2, lower bound: -0.1179465, upper bound: 0.1178253
NS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 15.38
Output dim: 2, lower bound: -0.1179470, upper bound: 0.1178253
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.38
Output dim: 2, lower bound: -0.1179470, upper bound: 0.1171786
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.38
Output dim: 2, lower bound: -0.1179470, upper bound: 0.1176396
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.38
Output dim: 2, lower bound: -0.1184043, upper bound: 0.1171786
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.38
Output dim: 2, lower bound: -0.1184043, upper bound: 0.1176396

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -5.4786925, -5.1492414, -5.4786925, -5.1492414, -0.2624037, 0.2624037
1: -1.9241651, -1.6635351, -1.9241651, -1.6635351, -0.1353905, 0.1353905
2: 7.6575255, 8.0292883, 7.6575255, 8.0292883, -0.2946148, 0.2946148
3: -3.1857047, -2.7699382, -3.1857047, -2.7699382, -0.1780591, 0.1780591
4: -10.6110401, -10.1850424, -10.6110401, -10.1850424, -0.1865590, 0.1865590
5: -8.4504519, -7.8972983, -8.4504519, -7.8972983, -0.2886412, 0.2886415
6: -7.3081980, -6.8665380, -7.3081980, -6.8665380, -0.1787412, 0.1787413
7: -5.2862792, -4.8471990, -5.2862792, -4.8471990, -0.1904581, 0.1904578
8: -0.9848564, -0.7274609, -0.9848564, -0.7274609, -0.1465142, 0.1465142
9: -11.4134865, -11.0449066, -11.4134865, -11.0449066, -0.1696012, 0.1696013

Time for backsubstitution: 8.76 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 60
type: A, layer: 3, pos: 2494
type: B, layer: 3, pos: 2494
type: A, layer: 3, pos: 753
type: B, layer: 3, pos: 753
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1262
type: B, layer: 3, pos: 1262
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 2322
type: B, layer: 3, pos: 2322
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2466
type: B, layer: 3, pos: 2466
type: A, layer: 3, pos: 2928
type: B, layer: 3, pos: 2928

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 60

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1163859, upper bound: 0.1166263
time: 2.88 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1170359, upper bound: 0.1179622
time: 3.43 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -5.4786925, -5.1492414, -5.4855371, -5.1501546, -0.2673397, 0.2746406
1: -1.9241651, -1.6635351, -1.9249485, -1.6651742, -0.1330773, 0.1352253
2: 7.6575255, 8.0292883, 7.6579580, 8.0291843, -0.2942214, 0.2933888
3: -3.1857047, -2.7699382, -3.1879721, -2.7705901, -0.1797338, 0.1822211
4: -10.6110401, -10.1850424, -10.6142292, -10.1759052, -0.1956291, 0.1900306
5: -8.4504519, -7.8972983, -8.4573936, -7.8962078, -0.2922688, 0.2979093
6: -7.3081980, -6.8665380, -7.3160810, -6.8633862, -0.1815013, 0.1858940
7: -5.2862792, -4.8471990, -5.2884812, -4.8394351, -0.1991282, 0.1938775
8: -0.9848564, -0.7274609, -0.9859877, -0.7281075, -0.1474694, 0.1487206
9: -11.4134865, -11.0449066, -11.4126415, -11.0473337, -0.1675674, 0.1692598

Time for backsubstitution: 7.87 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2494
type: B, layer: 3, pos: 60
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 753
type: B, layer: 3, pos: 753
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2494
type: B, layer: 3, pos: 1262
type: A, layer: 3, pos: 1262
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2322
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 2322
type: A, layer: 3, pos: 2466
type: B, layer: 3, pos: 2466
type: B, layer: 3, pos: 2928
type: A, layer: 3, pos: 2928

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 2494

## Relational analysis of NS_A1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1141543, upper bound: 0.1151787
time: 2.89 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1141543, upper bound: 0.1151484
time: 2.82 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -5.4855371, -5.1501546, -5.4786925, -5.1492414, -0.2746406, 0.2673392
1: -1.9249485, -1.6651742, -1.9241651, -1.6635351, -0.1352253, 0.1330773
2: 7.6579580, 8.0291843, 7.6575255, 8.0292883, -0.2933888, 0.2942214
3: -3.1879721, -2.7705901, -3.1857047, -2.7699382, -0.1822211, 0.1797338
4: -10.6142292, -10.1759052, -10.6110401, -10.1850424, -0.1900303, 0.1956291
5: -8.4573936, -7.8962078, -8.4504519, -7.8972983, -0.2979095, 0.2922688
6: -7.3160810, -6.8633862, -7.3081980, -6.8665380, -0.1858940, 0.1815013
7: -5.2884812, -4.8394351, -5.2862792, -4.8471990, -0.1938775, 0.1991282
8: -0.9859877, -0.7281075, -0.9848564, -0.7274609, -0.1487206, 0.1474694
9: -11.4126415, -11.0473337, -11.4134865, -11.0449066, -0.1692598, 0.1675675

Time for backsubstitution: 7.85 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2494
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 753
type: A, layer: 3, pos: 753
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2494
type: A, layer: 3, pos: 1262
type: B, layer: 3, pos: 1262
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 2322
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 2322
type: B, layer: 3, pos: 2466
type: A, layer: 3, pos: 2466
type: A, layer: 3, pos: 2928
type: B, layer: 3, pos: 2928

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 2494

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1146314, upper bound: 0.1141542
time: 3.09 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1146011, upper bound: 0.1141541
time: 3.15 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -5.4855371, -5.1501546, -5.4855371, -5.1501546, -0.2686570, 0.2686570
1: -1.9249485, -1.6651742, -1.9249485, -1.6651742, -0.1365216, 0.1365216
2: 7.6579580, 8.0291843, 7.6579580, 8.0291843, -0.3033543, 0.3033547
3: -3.1879721, -2.7705901, -3.1879721, -2.7705901, -0.1807437, 0.1807437
4: -10.6142292, -10.1759052, -10.6142292, -10.1759052, -0.1906247, 0.1906247
5: -8.4573936, -7.8962078, -8.4573936, -7.8962078, -0.2950084, 0.2950084
6: -7.3160810, -6.8633862, -7.3160810, -6.8633862, -0.1825521, 0.1825521
7: -5.2884812, -4.8394351, -5.2884812, -4.8394351, -0.1948638, 0.1948640
8: -0.9859877, -0.7281075, -0.9859877, -0.7281075, -0.1487010, 0.1487010
9: -11.4126415, -11.0473337, -11.4126415, -11.0473337, -0.1701292, 0.1701292

Time for backsubstitution: 7.90 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 2494
type: A, layer: 3, pos: 2494
type: A, layer: 3, pos: 753
type: B, layer: 3, pos: 753
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1262
type: B, layer: 3, pos: 1262
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 2322
type: B, layer: 3, pos: 2322
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2466
type: B, layer: 3, pos: 2466
type: A, layer: 3, pos: 2928
type: B, layer: 3, pos: 2928

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 60

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1168331, upper bound: 0.1156321
time: 2.86 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1175103, upper bound: 0.1170359
time: 3.18 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -5.4786925, -5.1492414, -5.4842277, -5.1451063, -0.2647200, 0.2648711
1: -1.9241651, -1.6635351, -1.9247060, -1.6634014, -0.1361115, 0.1364577
2: 7.6575255, 8.0292883, 7.6557417, 8.0294628, -0.3005166, 0.3022227
3: -3.1857047, -2.7699382, -3.1845701, -2.7706549, -0.1789840, 0.1789787
4: -10.6110401, -10.1850424, -10.6119690, -10.1849117, -0.1843865, 0.1854305
5: -8.4504519, -7.8972983, -8.4471102, -7.8988752, -0.2897325, 0.2887595
6: -7.3081980, -6.8665380, -7.3045149, -6.8682666, -0.1909673, 0.1886109
7: -5.2862792, -4.8471990, -5.2861967, -4.8489618, -0.1910069, 0.1926498
8: -0.9848564, -0.7274609, -0.9847944, -0.7275162, -0.1468918, 0.1469427
9: -11.4134865, -11.0449066, -11.4166288, -11.0423851, -0.1729879, 0.1730735

Time for backsubstitution: 7.95 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 60
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 2494
type: A, layer: 3, pos: 2494
type: B, layer: 3, pos: 753
type: A, layer: 3, pos: 753
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 1262
type: A, layer: 3, pos: 1262
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 2322
type: A, layer: 3, pos: 2322
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2466
type: A, layer: 3, pos: 2466
type: A, layer: 3, pos: 2928
type: B, layer: 3, pos: 2928

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 60

## Relational analysis of NS_A1_B2_A1_B1_B1

### Relational analysis result of NS_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1155746, upper bound: 0.1177322
time: 2.88 seconds

## Relational analysis of NS_A1_B2_A1_B1_B2

### Relational analysis result of NS_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1169400, upper bound: 0.1184502
time: 3.13 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -5.4786925, -5.1492414, -5.4911098, -5.1458297, -0.2696552, 0.2771432
1: -1.9241651, -1.6635351, -1.9254895, -1.6650400, -0.1338426, 0.1363182
2: 7.6575255, 8.0292883, 7.6561766, 8.0293598, -0.3001242, 0.3009939
3: -3.1857047, -2.7699382, -3.1868405, -2.7713082, -0.1806586, 0.1831625
4: -10.6110401, -10.1850424, -10.6151552, -10.1757765, -0.1934597, 0.1888909
5: -8.4504519, -7.8972983, -8.4540558, -7.8977876, -0.2933607, 0.2980375
6: -7.3081980, -6.8665380, -7.3123951, -6.8651137, -0.1937232, 0.1957664
7: -5.2862792, -4.8471990, -5.2884007, -4.8411946, -0.1996851, 0.1960695
8: -0.9848564, -0.7274609, -0.9859285, -0.7281630, -0.1478467, 0.1491485
9: -11.4134865, -11.0449066, -11.4157810, -11.0448084, -0.1709569, 0.1727315

Time for backsubstitution: 7.86 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2494
type: B, layer: 3, pos: 60
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 753
type: B, layer: 3, pos: 753
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2494
type: B, layer: 3, pos: 1262
type: A, layer: 3, pos: 1262
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2322
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 2322
type: A, layer: 3, pos: 2466
type: B, layer: 3, pos: 2466
type: A, layer: 3, pos: 2928
type: B, layer: 3, pos: 2928

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 2494

## Relational analysis of NS_A1_B2_A1_B2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1139802, upper bound: 0.1155821
time: 2.95 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1139802, upper bound: 0.1155518
time: 2.73 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -5.4855371, -5.1501546, -5.4842277, -5.1451063, -0.2769570, 0.2698069
1: -1.9249485, -1.6651742, -1.9247060, -1.6634014, -0.1359463, 0.1341444
2: 7.6579580, 8.0291843, 7.6557417, 8.0294628, -0.2992907, 0.3018293
3: -3.1879721, -2.7705901, -3.1845701, -2.7706549, -0.1831460, 0.1806535
4: -10.6142292, -10.1759052, -10.6119690, -10.1849117, -0.1878581, 0.1945007
5: -8.4573936, -7.8962078, -8.4471102, -7.8988752, -0.2990003, 0.2923868
6: -7.3160810, -6.8633862, -7.3045149, -6.8682666, -0.1981201, 0.1913708
7: -5.2884812, -4.8394351, -5.2861967, -4.8489618, -0.1944265, 0.2013199
8: -0.9859877, -0.7281075, -0.9847944, -0.7275162, -0.1490982, 0.1478978
9: -11.4126415, -11.0473337, -11.4166288, -11.0423851, -0.1726465, 0.1710396

Time for backsubstitution: 8.61 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2494
type: B, layer: 3, pos: 60
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 753
type: A, layer: 3, pos: 753
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2494
type: A, layer: 3, pos: 1262
type: B, layer: 3, pos: 1262
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 2322
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 2322
type: B, layer: 3, pos: 2466
type: A, layer: 3, pos: 2466
type: A, layer: 3, pos: 2928
type: B, layer: 3, pos: 2928

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 2494

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1144575, upper bound: 0.1145577
time: 2.81 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1144270, upper bound: 0.1145575
time: 3.13 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -5.4855371, -5.1501546, -5.4911098, -5.1458297, -0.2709742, 0.2711594
1: -1.9249485, -1.6651742, -1.9254895, -1.6650400, -0.1372422, 0.1375849
2: 7.6579580, 8.0291843, 7.6561766, 8.0293598, -0.3092561, 0.3109589
3: -3.1879721, -2.7705901, -3.1868405, -2.7713082, -0.1816686, 0.1816846
4: -10.6142292, -10.1759052, -10.6151552, -10.1757765, -0.1884553, 0.1894848
5: -8.4573936, -7.8962078, -8.4540558, -7.8977876, -0.2961004, 0.2951369
6: -7.3160810, -6.8633862, -7.3123951, -6.8651137, -0.1947736, 0.1924242
7: -5.2884812, -4.8394351, -5.2884007, -4.8411946, -0.1954215, 0.1970558
8: -0.9859877, -0.7281075, -0.9859285, -0.7281630, -0.1490782, 0.1491290
9: -11.4126415, -11.0473337, -11.4157810, -11.0448084, -0.1735225, 0.1736014

Time for backsubstitution: 8.56 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 60
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 2494
type: A, layer: 3, pos: 2494
type: B, layer: 3, pos: 753
type: A, layer: 3, pos: 753
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 1262
type: A, layer: 3, pos: 1262
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 2322
type: A, layer: 3, pos: 2322
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2466
type: A, layer: 3, pos: 2466
type: A, layer: 3, pos: 2928
type: B, layer: 3, pos: 2928

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 60

## Relational analysis of NS_A1_B2_A2_B2_B1

### Relational analysis result of NS_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1160218, upper bound: 0.1167380
time: 3.04 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2

### Relational analysis result of NS_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1174001, upper bound: 0.1175090
time: 3.00 seconds

## BFS NS instance: NS_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -5.4842277, -5.1451063, -5.4786925, -5.1492414, -0.2648711, 0.2647200
1: -1.9247060, -1.6634014, -1.9241651, -1.6635351, -0.1364577, 0.1361115
2: 7.6557417, 8.0294628, 7.6575255, 8.0292883, -0.3022227, 0.3005168
3: -3.1845701, -2.7706549, -3.1857047, -2.7699382, -0.1789787, 0.1789840
4: -10.6119690, -10.1849117, -10.6110401, -10.1850424, -0.1854305, 0.1843865
5: -8.4471102, -7.8988752, -8.4504519, -7.8972983, -0.2887595, 0.2897325
6: -7.3045149, -6.8682666, -7.3081980, -6.8665380, -0.1886108, 0.1909673
7: -5.2861967, -4.8489618, -5.2862792, -4.8471990, -0.1926496, 0.1910069
8: -0.9847944, -0.7275162, -0.9848564, -0.7274609, -0.1469427, 0.1468918
9: -11.4166288, -11.0423851, -11.4134865, -11.0449066, -0.1730733, 0.1729881

Time for backsubstitution: 8.71 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 60
type: A, layer: 3, pos: 2494
type: B, layer: 3, pos: 2494
type: A, layer: 3, pos: 753
type: B, layer: 3, pos: 753
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1262
type: B, layer: 3, pos: 1262
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 2322
type: B, layer: 3, pos: 2322
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2466
type: B, layer: 3, pos: 2466
type: B, layer: 3, pos: 2928
type: A, layer: 3, pos: 2928

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 60

## Relational analysis of NS_A2_B1_B1_A1_A1

### Relational analysis result of NS_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1177323, upper bound: 0.1155747
time: 3.06 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2

### Relational analysis result of NS_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1184502, upper bound: 0.1169400
time: 3.00 seconds

## BFS NS instance: NS_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -5.4911098, -5.1458297, -5.4786925, -5.1492414, -0.2771432, 0.2696552
1: -1.9254895, -1.6650400, -1.9241651, -1.6635351, -0.1363182, 0.1338426
2: 7.6561766, 8.0293598, 7.6575255, 8.0292883, -0.3009939, 0.3001244
3: -3.1868405, -2.7713082, -3.1857047, -2.7699382, -0.1831625, 0.1806586
4: -10.6151552, -10.1757765, -10.6110401, -10.1850424, -0.1888909, 0.1934597
5: -8.4540558, -7.8977876, -8.4504519, -7.8972983, -0.2980375, 0.2933607
6: -7.3123951, -6.8651137, -7.3081980, -6.8665380, -0.1957664, 0.1937232
7: -5.2884007, -4.8411946, -5.2862792, -4.8471990, -0.1960695, 0.1996851
8: -0.9859285, -0.7281630, -0.9848564, -0.7274609, -0.1491485, 0.1478467
9: -11.4157810, -11.0448084, -11.4134865, -11.0449066, -0.1727314, 0.1709569

Time for backsubstitution: 8.77 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2494
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 753
type: A, layer: 3, pos: 753
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2494
type: A, layer: 3, pos: 1262
type: B, layer: 3, pos: 1262
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 2322
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 2322
type: B, layer: 3, pos: 2466
type: A, layer: 3, pos: 2466
type: B, layer: 3, pos: 2928
type: A, layer: 3, pos: 2928

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 2494

## Relational analysis of NS_A2_B1_B1_A2_A1

### Relational analysis result of NS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1155820, upper bound: 0.1139802
time: 3.16 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2

### Relational analysis result of NS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1155518, upper bound: 0.1139803
time: 3.10 seconds

## BFS NS instance: NS_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -5.4842277, -5.1451063, -5.4855371, -5.1501546, -0.2698069, 0.2769570
1: -1.9247060, -1.6634014, -1.9249485, -1.6651742, -0.1341444, 0.1359463
2: 7.6557417, 8.0294628, 7.6579580, 8.0291843, -0.3018293, 0.2992909
3: -3.1845701, -2.7706549, -3.1879721, -2.7705901, -0.1806533, 0.1831460
4: -10.6119690, -10.1849117, -10.6142292, -10.1759052, -0.1945007, 0.1878581
5: -8.4471102, -7.8988752, -8.4573936, -7.8962078, -0.2923868, 0.2990003
6: -7.3045149, -6.8682666, -7.3160810, -6.8633862, -0.1913708, 0.1981200
7: -5.2861967, -4.8489618, -5.2884812, -4.8394351, -0.2013199, 0.1944265
8: -0.9847944, -0.7275162, -0.9859877, -0.7281075, -0.1478978, 0.1490982
9: -11.4166288, -11.0423851, -11.4126415, -11.0473337, -0.1710396, 0.1726465

Time for backsubstitution: 8.77 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2494
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 60
type: A, layer: 3, pos: 753
type: B, layer: 3, pos: 753
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2494
type: B, layer: 3, pos: 1262
type: A, layer: 3, pos: 1262
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2322
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 2322
type: A, layer: 3, pos: 2466
type: B, layer: 3, pos: 2466
type: B, layer: 3, pos: 2928
type: A, layer: 3, pos: 2928

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 2494

## Relational analysis of NS_A2_B1_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1145576, upper bound: 0.1144576
time: 3.01 seconds

## Relational analysis of NS_A2_B1_B2_A1_B2

### Relational analysis result of NS_A2_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1145576, upper bound: 0.1144273
time: 3.07 seconds

## BFS NS instance: NS_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -5.4911098, -5.1458297, -5.4855371, -5.1501546, -0.2711594, 0.2709739
1: -1.9254895, -1.6650400, -1.9249485, -1.6651742, -0.1375849, 0.1372422
2: 7.6561766, 8.0293598, 7.6579580, 8.0291843, -0.3109589, 0.3092561
3: -3.1868405, -2.7713082, -3.1879721, -2.7705901, -0.1816846, 0.1816686
4: -10.6151552, -10.1757765, -10.6142292, -10.1759052, -0.1894848, 0.1884553
5: -8.4540558, -7.8977876, -8.4573936, -7.8962078, -0.2951369, 0.2961004
6: -7.3123951, -6.8651137, -7.3160810, -6.8633862, -0.1924242, 0.1947736
7: -5.2884007, -4.8411946, -5.2884812, -4.8394351, -0.1970561, 0.1954215
8: -0.9859285, -0.7281630, -0.9859877, -0.7281075, -0.1491290, 0.1490782
9: -11.4157810, -11.0448084, -11.4126415, -11.0473337, -0.1736013, 0.1735222

Time for backsubstitution: 8.80 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 60
type: A, layer: 3, pos: 2494
type: B, layer: 3, pos: 2494
type: A, layer: 3, pos: 753
type: B, layer: 3, pos: 753
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1262
type: B, layer: 3, pos: 1262
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 2322
type: B, layer: 3, pos: 2322
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2466
type: B, layer: 3, pos: 2466
type: B, layer: 3, pos: 2928
type: A, layer: 3, pos: 2928

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 60

## Relational analysis of NS_A2_B1_B2_A2_A1

### Relational analysis result of NS_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1167378, upper bound: 0.1155745
time: 3.33 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2

### Relational analysis result of NS_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1175091, upper bound: 0.1169400
time: 3.27 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -5.4842277, -5.1451063, -5.4842277, -5.1451063, -0.2661109, 0.2661109
1: -1.9247060, -1.6634014, -1.9247060, -1.6634014, -0.1358945, 0.1358945
2: 7.6557417, 8.0294628, 7.6557417, 8.0294628, -0.2959652, 0.2959650
3: -3.1845701, -2.7706549, -3.1845701, -2.7706549, -0.1781118, 0.1781118
4: -10.6119690, -10.1849117, -10.6119690, -10.1849117, -0.1877117, 0.1877117
5: -8.4471102, -7.8988752, -8.4471102, -7.8988752, -0.2879627, 0.2879627
6: -7.3045149, -6.8682666, -7.3045149, -6.8682666, -0.1774029, 0.1774029
7: -5.2861967, -4.8489618, -5.2861967, -4.8489618, -0.1910005, 0.1910005
8: -0.9847944, -0.7275162, -0.9847944, -0.7275162, -0.1464998, 0.1464998
9: -11.4166288, -11.0423851, -11.4166288, -11.0423851, -0.1705790, 0.1705790

Time for backsubstitution: 8.73 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 60
type: A, layer: 3, pos: 2494
type: B, layer: 3, pos: 2494
type: A, layer: 3, pos: 753
type: B, layer: 3, pos: 753
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1262
type: B, layer: 3, pos: 1262
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 2322
type: B, layer: 3, pos: 2322
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2466
type: B, layer: 3, pos: 2466
type: A, layer: 3, pos: 2928
type: B, layer: 3, pos: 2928

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 60

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1167379, upper bound: 0.1165687
time: 2.96 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1175092, upper bound: 0.1176944
time: 3.45 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -5.4842277, -5.1451063, -5.4911098, -5.1458297, -0.2710457, 0.2783594
1: -1.9247060, -1.6634014, -1.9254895, -1.6650400, -0.1335829, 0.1357547
2: 7.6557417, 8.0294628, 7.6561766, 8.0293598, -0.2955718, 0.2947402
3: -3.1845701, -2.7706549, -3.1868405, -2.7713082, -0.1797864, 0.1822741
4: -10.6119690, -10.1849117, -10.6151552, -10.1757765, -0.1967821, 0.1911719
5: -8.4471102, -7.8988752, -8.4540558, -7.8977876, -0.2915902, 0.2972307
6: -7.3045149, -6.8682666, -7.3123951, -6.8651137, -0.1801578, 0.1845561
7: -5.2861967, -4.8489618, -5.2884007, -4.8411946, -0.1996701, 0.1944191
8: -0.9847944, -0.7275162, -0.9859285, -0.7281630, -0.1474549, 0.1487056
9: -11.4166288, -11.0423851, -11.4157810, -11.0448084, -0.1685435, 0.1702373

Time for backsubstitution: 8.75 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2494
type: B, layer: 3, pos: 60
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 753
type: B, layer: 3, pos: 753
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2494
type: B, layer: 3, pos: 1262
type: A, layer: 3, pos: 1262
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2322
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 2322
type: A, layer: 3, pos: 2466
type: B, layer: 3, pos: 2466
type: B, layer: 3, pos: 2928
type: A, layer: 3, pos: 2928

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 2494

## Relational analysis of NS_A2_B2_A1_B2_B1

### Relational analysis result of NS_A2_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1145577, upper bound: 0.1148190
time: 3.36 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2

### Relational analysis result of NS_A2_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1145578, upper bound: 0.1147888
time: 3.54 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -5.4911098, -5.1458297, -5.4842277, -5.1451063, -0.2783594, 0.2710457
1: -1.9254895, -1.6650400, -1.9247060, -1.6634014, -0.1357547, 0.1335829
2: 7.6561766, 8.0293598, 7.6557417, 8.0294628, -0.2947402, 0.2955718
3: -3.1868405, -2.7713082, -3.1845701, -2.7706549, -0.1822741, 0.1797864
4: -10.6151552, -10.1757765, -10.6119690, -10.1849117, -0.1911719, 0.1967821
5: -8.4540558, -7.8977876, -8.4471102, -7.8988752, -0.2972307, 0.2915902
6: -7.3123951, -6.8651137, -7.3045149, -6.8682666, -0.1845561, 0.1801578
7: -5.2884007, -4.8411946, -5.2861967, -4.8489618, -0.1944191, 0.1996703
8: -0.9859285, -0.7281630, -0.9847944, -0.7275162, -0.1487056, 0.1474549
9: -11.4157810, -11.0448084, -11.4166288, -11.0423851, -0.1702375, 0.1685436

Time for backsubstitution: 8.73 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2494
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 753
type: A, layer: 3, pos: 753
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2494
type: A, layer: 3, pos: 1262
type: B, layer: 3, pos: 1262
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 2322
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 2322
type: B, layer: 3, pos: 2466
type: A, layer: 3, pos: 2466
type: A, layer: 3, pos: 2928
type: B, layer: 3, pos: 2928

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 2494

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1150352, upper bound: 0.1137947
time: 2.99 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1150049, upper bound: 0.1137947
time: 3.20 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -5.4911098, -5.1458297, -5.4911098, -5.1458297, -0.2723761, 0.2723761
1: -1.9254895, -1.6650400, -1.9254895, -1.6650400, -0.1370215, 0.1370215
2: 7.6561766, 8.0293598, 7.6561766, 8.0293598, -0.3047042, 0.3047044
3: -3.1868405, -2.7713082, -3.1868405, -2.7713082, -0.1807967, 0.1807967
4: -10.6151552, -10.1757765, -10.6151552, -10.1757765, -0.1917663, 0.1917663
5: -8.4540558, -7.8977876, -8.4540558, -7.8977876, -0.2943299, 0.2943299
6: -7.3123951, -6.8651137, -7.3123951, -6.8651137, -0.1812093, 0.1812093
7: -5.2884007, -4.8411946, -5.2884007, -4.8411946, -0.1954057, 0.1954057
8: -0.9859285, -0.7281630, -0.9859285, -0.7281630, -0.1486861, 0.1486861
9: -11.4157810, -11.0448084, -11.4157810, -11.0448084, -0.1711087, 0.1711087

Time for backsubstitution: 8.77 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 2494
type: A, layer: 3, pos: 2494
type: A, layer: 3, pos: 753
type: B, layer: 3, pos: 753
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1262
type: B, layer: 3, pos: 1262
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 2322
type: B, layer: 3, pos: 2322
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2466
type: B, layer: 3, pos: 2466
type: A, layer: 3, pos: 2928
type: B, layer: 3, pos: 2928

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 60

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1171851, upper bound: 0.1155745
time: 2.99 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1179695, upper bound: 0.1167432
time: 3.31 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 15.23 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 2, lower bound: -0.1163859, upper bound: 0.1166263
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 2, lower bound: -0.1170359, upper bound: 0.1179622
NS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 2, lower bound: -0.1141543, upper bound: 0.1151787
NS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 2, lower bound: -0.1141543, upper bound: 0.1151484
NS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 15.23
Output dim: 2, lower bound: -0.1146314, upper bound: 0.1141542
NS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 15.23
Output dim: 2, lower bound: -0.1146011, upper bound: 0.1141541
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 2, lower bound: -0.1168331, upper bound: 0.1156321
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 2, lower bound: -0.1175103, upper bound: 0.1170359
NS_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 2, lower bound: -0.1155746, upper bound: 0.1177322
NS_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 2, lower bound: -0.1169400, upper bound: 0.1184502
NS_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 2, lower bound: -0.1139802, upper bound: 0.1155821
NS_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 2, lower bound: -0.1139802, upper bound: 0.1155518
NS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 15.23
Output dim: 2, lower bound: -0.1144575, upper bound: 0.1145577
NS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 15.23
Output dim: 2, lower bound: -0.1144270, upper bound: 0.1145575
NS_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 2, lower bound: -0.1160218, upper bound: 0.1167380
NS_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 2, lower bound: -0.1174001, upper bound: 0.1175090
NS_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 2, lower bound: -0.1177323, upper bound: 0.1155747
NS_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 2, lower bound: -0.1184502, upper bound: 0.1169400
NS_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 2, lower bound: -0.1155820, upper bound: 0.1139802
NS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 2, lower bound: -0.1155518, upper bound: 0.1139803
NS_A2_B1_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 15.23
Output dim: 2, lower bound: -0.1145576, upper bound: 0.1144576
NS_A2_B1_B2_A1_B2, status: Status.VERIFIED, split count: 5, time: 15.23
Output dim: 2, lower bound: -0.1145576, upper bound: 0.1144273
NS_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 2, lower bound: -0.1167378, upper bound: 0.1155745
NS_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 2, lower bound: -0.1175091, upper bound: 0.1169400
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 2, lower bound: -0.1167379, upper bound: 0.1165687
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 2, lower bound: -0.1175092, upper bound: 0.1176944
NS_A2_B2_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 15.23
Output dim: 2, lower bound: -0.1145577, upper bound: 0.1148190
NS_A2_B2_A1_B2_B2, status: Status.VERIFIED, split count: 5, time: 15.23
Output dim: 2, lower bound: -0.1145578, upper bound: 0.1147888
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 2, lower bound: -0.1150352, upper bound: 0.1137947
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 2, lower bound: -0.1150049, upper bound: 0.1137947
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 2, lower bound: -0.1171851, upper bound: 0.1155745
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 2, lower bound: -0.1179695, upper bound: 0.1167432

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -5.4780483, -5.1489267, -5.4785309, -5.1492414, -0.2619851, 0.2628658
1: -1.9243892, -1.6644975, -1.9241319, -1.6637769, -0.1359086, 0.1347493
2: 7.6553617, 8.0258799, 7.6575236, 8.0283279, -0.2939329, 0.2898118
3: -3.1845384, -2.7691550, -3.1854053, -2.7699382, -0.1766341, 0.1784540
4: -10.6112452, -10.1852798, -10.6110153, -10.1851034, -0.1867309, 0.1862979
5: -8.4486256, -7.8929443, -8.4499950, -7.8973069, -0.2863889, 0.2919524
6: -7.3069963, -6.8654909, -7.3078966, -6.8665466, -0.1774753, 0.1794740
7: -5.2831249, -4.8458271, -5.2853780, -4.8471994, -0.1858215, 0.1889403
8: -0.9848723, -0.7275245, -0.9848549, -0.7274780, -0.1465335, 0.1464583
9: -11.4133892, -11.0448751, -11.4134626, -11.0449238, -0.1694614, 0.1695698

Time for backsubstitution: 8.74 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2494
type: B, layer: 3, pos: 2494
type: A, layer: 3, pos: 753
type: B, layer: 3, pos: 753
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 60
type: A, layer: 3, pos: 1262
type: B, layer: 3, pos: 1262
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 2322
type: B, layer: 3, pos: 2322
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2466
type: B, layer: 3, pos: 2466
type: B, layer: 3, pos: 2928
type: A, layer: 3, pos: 2928

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 2494

## Relational analysis of NS_A1_B1_A1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1140133, upper bound: 0.1132293
time: 2.79 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1139832, upper bound: 0.1132294
time: 2.86 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -5.4786024, -5.1492414, -5.4786925, -5.1492414, -0.2627349, 0.2624032
1: -1.9241436, -1.6636258, -1.9241651, -1.6635351, -0.1353718, 0.1357086
2: 7.6575232, 8.0289173, 7.6575255, 8.0292883, -0.2946153, 0.2906680
3: -3.1855316, -2.7699382, -3.1857047, -2.7699382, -0.1775296, 0.1780589
4: -10.6110249, -10.1851025, -10.6110401, -10.1850424, -0.1865442, 0.1866622
5: -8.4496441, -7.8973055, -8.4504519, -7.8972983, -0.2893975, 0.2886348
6: -7.3079748, -6.8665452, -7.3081980, -6.8665380, -0.1785498, 0.1787356
7: -5.2860146, -4.8471999, -5.2862792, -4.8471990, -0.1860430, 0.1904578
8: -0.9848573, -0.7274668, -0.9848564, -0.7274609, -0.1465142, 0.1465005
9: -11.4134674, -11.0449181, -11.4134865, -11.0449066, -0.1695668, 0.1695879

Time for backsubstitution: 8.50 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2494
type: B, layer: 3, pos: 2494
type: A, layer: 3, pos: 753
type: B, layer: 3, pos: 753
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1262
type: B, layer: 3, pos: 1262
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 2322
type: B, layer: 3, pos: 2322
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2466
type: B, layer: 3, pos: 2466
type: B, layer: 3, pos: 2928
type: A, layer: 3, pos: 2928

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 2494

## Relational analysis of NS_A1_B1_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1147500, upper bound: 0.1147197
time: 2.90 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1147197, upper bound: 0.1147197
time: 3.36 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -5.4786110, -5.1492414, -5.4821062, -5.1501546, -0.2672510, 0.2709682
1: -1.9241654, -1.6635371, -1.9249485, -1.6652410, -0.1324551, 0.1349199
2: 7.6575818, 8.0292892, 7.6604815, 8.0291853, -0.2941198, 0.2893543
3: -3.1856327, -2.7699382, -3.1850200, -2.7705901, -0.1796627, 0.1792978
4: -10.6110392, -10.1851759, -10.6142292, -10.1815186, -0.1904588, 0.1898954
5: -8.4503632, -7.8972988, -8.4537678, -7.8962078, -0.2921953, 0.2948217
6: -7.3081651, -6.8665380, -7.3146825, -6.8633833, -0.1814630, 0.1843022
7: -5.2862787, -4.8473454, -5.2884822, -4.8455009, -0.1930778, 0.1937189
8: -0.9847782, -0.7274623, -0.9827099, -0.7281079, -0.1473870, 0.1452558
9: -11.4134865, -11.0449619, -11.4126415, -11.0496073, -0.1655319, 0.1692113

Time for backsubstitution: 8.62 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2494
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 753
type: A, layer: 3, pos: 753
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1262
type: B, layer: 3, pos: 1262
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 2322
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 2322
type: B, layer: 3, pos: 2466
type: A, layer: 3, pos: 2466
type: A, layer: 3, pos: 2928
type: B, layer: 3, pos: 2928

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 2494

## Relational analysis of NS_A1_B1_A1_B2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1139178, upper bound: 0.1149119
time: 2.93 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1139178, upper bound: 0.1149119
time: 2.83 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -5.4776635, -5.1492414, -5.4844751, -5.1417789, -0.2757802, 0.2724581
1: -1.9241650, -1.6635765, -1.9250891, -1.6643735, -0.1324250, 0.1362252
2: 7.6597309, 8.0292883, 7.6615901, 8.0347681, -0.3039727, 0.2912655
3: -3.1852846, -2.7699382, -3.1879916, -2.7623458, -0.1867819, 0.1810694
4: -10.6110401, -10.1853580, -10.6293354, -10.1762829, -0.1952734, 0.2043808
5: -8.4502096, -7.8972998, -8.4575615, -7.8862691, -0.2993932, 0.2962477
6: -7.3077745, -6.8665380, -7.3152027, -6.8600202, -0.1855481, 0.1854426
7: -5.2862782, -4.8480430, -5.3044415, -4.8412218, -0.1975648, 0.2106152
8: -0.9840045, -0.7274599, -0.9845510, -0.7199869, -0.1559820, 0.1472894
9: -11.4134865, -11.0452538, -11.4186068, -11.0475111, -0.1671737, 0.1750784

Time for backsubstitution: 8.64 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2494
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 753
type: A, layer: 3, pos: 753
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1262
type: B, layer: 3, pos: 1262
type: A, layer: 3, pos: 2322
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 2466
type: A, layer: 3, pos: 2466
type: B, layer: 3, pos: 2322
type: A, layer: 3, pos: 2928
type: B, layer: 3, pos: 2928

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 2494

## Relational analysis of NS_A1_B1_A1_B2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1139178, upper bound: 0.1151485
time: 3.19 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1139178, upper bound: 0.1151485
time: 3.13 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5.4848890, -5.1498408, -5.4853735, -5.1501546, -0.2682357, 0.2691193
1: -1.9251730, -1.6661363, -1.9249161, -1.6654155, -0.1370370, 0.1358775
2: 7.6557913, 8.0257778, 7.6579580, 8.0282249, -0.3026690, 0.2985528
3: -3.1868062, -2.7698069, -3.1876724, -2.7705901, -0.1793189, 0.1811389
4: -10.6144323, -10.1761427, -10.6142035, -10.1759644, -0.1908004, 0.1903644
5: -8.4555683, -7.8918505, -8.4569387, -7.8962183, -0.2927556, 0.2983201
6: -7.3148832, -6.8623371, -7.3157797, -6.8633900, -0.1812874, 0.1832868
7: -5.2853270, -4.8380632, -5.2875795, -4.8394351, -0.1902275, 0.1933460
8: -0.9860058, -0.7281721, -0.9859874, -0.7281241, -0.1487203, 0.1486447
9: -11.4125423, -11.0473003, -11.4126167, -11.0473499, -0.1699897, 0.1700969

Time for backsubstitution: 8.62 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2494
type: B, layer: 3, pos: 2494
type: A, layer: 3, pos: 753
type: B, layer: 3, pos: 753
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 60
type: A, layer: 3, pos: 1262
type: B, layer: 3, pos: 1262
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 2322
type: B, layer: 3, pos: 2322
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2466
type: B, layer: 3, pos: 2466
type: B, layer: 3, pos: 2928
type: A, layer: 3, pos: 2928

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 2494

## Relational analysis of NS_A1_B1_A2_B2_A1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1134641, upper bound: 0.1122329
time: 2.86 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1134338, upper bound: 0.1122329
time: 3.33 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -5.4854445, -5.1501546, -5.4855371, -5.1501546, -0.2689872, 0.2686567
1: -1.9249281, -1.6652633, -1.9249485, -1.6651742, -0.1365031, 0.1368372
2: 7.6579590, 8.0288143, 7.6579580, 8.0291843, -0.3033552, 0.2994053
3: -3.1877997, -2.7705901, -3.1879721, -2.7705901, -0.1802143, 0.1807435
4: -10.6142120, -10.1759653, -10.6142292, -10.1759052, -0.1906104, 0.1907279
5: -8.4565878, -7.8962150, -8.4573936, -7.8962078, -0.2957649, 0.2950017
6: -7.3158603, -6.8633900, -7.3160810, -6.8633862, -0.1823616, 0.1825467
7: -5.2882166, -4.8394341, -5.2884812, -4.8394351, -0.1904488, 0.1948638
8: -0.9859867, -0.7281134, -0.9859877, -0.7281075, -0.1487010, 0.1486872
9: -11.4126196, -11.0473423, -11.4126415, -11.0473337, -0.1700947, 0.1701158

Time for backsubstitution: 9.07 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2494
type: B, layer: 3, pos: 2494
type: A, layer: 3, pos: 753
type: B, layer: 3, pos: 753
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1262
type: B, layer: 3, pos: 1262
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 2322
type: B, layer: 3, pos: 2322
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2466
type: B, layer: 3, pos: 2466
type: B, layer: 3, pos: 2928
type: A, layer: 3, pos: 2928

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 2494

## Relational analysis of NS_A1_B1_A2_B2_A2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1142006, upper bound: 0.1137231
time: 3.39 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1141703, upper bound: 0.1137231
time: 3.67 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -5.4785309, -5.1492414, -5.4835672, -5.1447935, -0.2651825, 0.2643881
1: -1.9241319, -1.6637769, -1.9249313, -1.6643646, -0.1354700, 0.1369457
2: 7.6575236, 8.0283279, 7.6532879, 8.0262928, -0.2960272, 0.3016906
3: -3.1854053, -2.7699382, -3.1833930, -2.7698827, -0.1794136, 0.1775438
4: -10.6110153, -10.1851034, -10.6121740, -10.1851501, -0.1841254, 0.1856052
5: -8.4499950, -7.8973069, -8.4451294, -7.8945208, -0.2923114, 0.2858701
6: -7.3078966, -6.8665466, -7.3032274, -6.8672204, -0.1917678, 0.1874702
7: -5.2853780, -4.8471994, -5.2835112, -4.8470469, -0.1901479, 0.1885922
8: -0.9848549, -0.7274780, -0.9848132, -0.7275810, -0.1468359, 0.1469617
9: -11.4134626, -11.0449238, -11.4165163, -11.0423679, -0.1729529, 0.1729331

Time for backsubstitution: 9.05 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 55.77 + 551.88 = 607.65 seconds
