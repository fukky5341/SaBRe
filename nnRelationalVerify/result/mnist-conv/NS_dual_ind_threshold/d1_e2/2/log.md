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
execution time: IAR + RelationalAnalysis = 22.71 + 34.27 = 56.98 seconds
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
type: A, layer: 3, pos: 2494
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 753
type: A, layer: 3, pos: 1110
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1262
type: A, layer: 3, pos: 2322
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 2466
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 2928

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 214

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1184842, upper bound: 0.1190632
time: 3.10 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1190629, upper bound: 0.1190629
time: 3.68 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.06 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 7.06
Output dim: 2, lower bound: -0.1184842, upper bound: 0.1190632
NS_A2, status: Status.UNKNOWN, split count: 1, time: 7.06
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

Time for backsubstitution: 7.80 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2494
type: B, layer: 3, pos: 214
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 753
type: B, layer: 3, pos: 1110
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 1262
type: B, layer: 3, pos: 2322
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2466
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 2928

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 2494

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1151140, upper bound: 0.1157219
time: 2.78 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1151141, upper bound: 0.1156916
time: 2.93 seconds

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

Time for backsubstitution: 7.89 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2494
type: B, layer: 3, pos: 214
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 753
type: B, layer: 3, pos: 1110
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 1262
type: B, layer: 3, pos: 2322
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2466
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 2928

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 2494

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1156917, upper bound: 0.1157219
time: 2.85 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1156917, upper bound: 0.1156916
time: 2.88 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 13.92 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 13.92
Output dim: 2, lower bound: -0.1151140, upper bound: 0.1157219
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 13.92
Output dim: 2, lower bound: -0.1151141, upper bound: 0.1156916
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 13.92
Output dim: 2, lower bound: -0.1156917, upper bound: 0.1157219
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 13.92
Output dim: 2, lower bound: -0.1156917, upper bound: 0.1156916

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -5.4868960, -5.1501546, -5.4891157, -5.1465549, -0.2826819, 0.2785373
1: -1.9258331, -1.6651752, -1.9263667, -1.6649526, -0.1342009, 0.1350875
2: 7.6580186, 8.0296555, 7.6574106, 8.0298271, -0.3017850, 0.2969673
3: -3.1884954, -2.7705901, -3.1844687, -2.7695522, -0.1849523, 0.1819578
4: -10.6142282, -10.1758671, -10.6156120, -10.1812153, -0.1918638, 0.1996582
5: -8.4578590, -7.8962088, -8.4509830, -7.8930469, -0.3039150, 0.2986536
6: -7.3161268, -6.8633857, -7.3110771, -6.8599772, -0.1966771, 0.1970476
7: -5.2884812, -4.8390732, -5.2913980, -4.8467565, -0.1971819, 0.2052703
8: -0.9863331, -0.7281075, -0.9832644, -0.7281630, -0.1499928, 0.1465293
9: -11.4136963, -11.0473890, -11.4160032, -11.0470905, -0.1702358, 0.1711208

Time for backsubstitution: 8.44 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2494
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 753
type: A, layer: 3, pos: 1110
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1262
type: A, layer: 3, pos: 2322
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 2466
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 2928

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 2494

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1148775, upper bound: 0.1154552
time: 2.99 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1148775, upper bound: 0.1154551
time: 3.35 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -5.4859514, -5.1501546, -5.4916697, -5.1381798, -0.2912102, 0.2802074
1: -1.9258330, -1.6652117, -1.9265063, -1.6640851, -0.1341629, 0.1363835
2: 7.6601515, 8.0296545, 7.6585398, 8.0354118, -0.3116474, 0.2988472
3: -3.1881452, -2.7705901, -3.1874235, -2.7613077, -0.1920713, 0.1838661
4: -10.6142282, -10.1760492, -10.6307173, -10.1759806, -0.1966980, 0.2141433
5: -8.4577065, -7.8962078, -8.4548206, -7.8831081, -0.3111143, 0.3001397
6: -7.3157387, -6.8633833, -7.3116012, -6.8566141, -0.2007653, 0.1982002
7: -5.2884808, -4.8397751, -5.3073578, -4.8424640, -0.2017505, 0.2221646
8: -0.9855638, -0.7281075, -0.9850926, -0.7200425, -0.1585888, 0.1485623
9: -11.4136963, -11.0476809, -11.4219694, -11.0449848, -0.1718733, 0.1769886

Time for backsubstitution: 7.92 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2494
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 753
type: A, layer: 3, pos: 1110
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1262
type: A, layer: 3, pos: 2322
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 2466
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 2928

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 2494

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1148775, upper bound: 0.1156917
time: 2.72 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1148775, upper bound: 0.1156917
time: 2.85 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -5.4924655, -5.1458297, -5.4891276, -5.1450992, -0.2860889, 0.2797546
1: -1.9263742, -1.6650416, -1.9263744, -1.6648983, -0.1355534, 0.1345372
2: 7.6562366, 8.0298290, 7.6565309, 8.0298309, -0.2972372, 0.3057613
3: -3.1873651, -2.7713082, -3.1844876, -2.7697256, -0.1863332, 0.1810704
4: -10.6151562, -10.1757364, -10.6158323, -10.1812143, -0.1951778, 0.1969924
5: -8.4545240, -7.8977861, -8.4509850, -7.8934221, -0.3053513, 0.2978075
6: -7.3124428, -6.8651099, -7.3110771, -6.8603754, -0.2109120, 0.1858202
7: -5.2884002, -4.8408332, -5.2912922, -4.8467541, -0.1971395, 0.2070823
8: -0.9862752, -0.7281635, -0.9832497, -0.7281651, -0.1496007, 0.1470557
9: -11.4168367, -11.0448618, -11.4171391, -11.0470829, -0.1678914, 0.1757354

Time for backsubstitution: 8.12 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2494
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 753
type: A, layer: 3, pos: 1110
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1262
type: A, layer: 3, pos: 2322
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 2466
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 2928

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 2494

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1154551, upper bound: 0.1154552
time: 2.82 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1154551, upper bound: 0.1154552
time: 2.94 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -5.4915237, -5.1458297, -5.4916821, -5.1367240, -0.2946186, 0.2812428
1: -1.9263735, -1.6650782, -1.9265153, -1.6640310, -0.1355149, 0.1358334
2: 7.6583695, 8.0298290, 7.6576481, 8.0354128, -0.3070998, 0.3076353
3: -3.1870158, -2.7713082, -3.1874430, -2.7614830, -0.1934533, 0.1828396
4: -10.6151571, -10.1759214, -10.6309395, -10.1759787, -0.1999917, 0.2114780
5: -8.4543676, -7.8977880, -8.4548244, -7.8834820, -0.3125505, 0.2992349
6: -7.3120542, -6.8651123, -7.3116045, -6.8570137, -0.2150002, 0.1869611
7: -5.2883987, -4.8415341, -5.3072519, -4.8424621, -0.2016335, 0.2239773
8: -0.9855037, -0.7281637, -0.9850771, -0.7200427, -0.1581964, 0.1490884
9: -11.4168367, -11.0451546, -11.4231071, -11.0449753, -0.1695358, 0.1815991

Time for backsubstitution: 8.10 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2494
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 753
type: A, layer: 3, pos: 1110
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1262
type: A, layer: 3, pos: 2322
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 2466
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 2928

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 2494

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1154551, upper bound: 0.1156917
time: 2.94 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1154551, upper bound: 0.1156917
time: 3.08 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 14.37 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 14.37
Output dim: 2, lower bound: -0.1148775, upper bound: 0.1154552
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 14.37
Output dim: 2, lower bound: -0.1148775, upper bound: 0.1154551
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 14.37
Output dim: 2, lower bound: -0.1148775, upper bound: 0.1156917
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 14.37
Output dim: 2, lower bound: -0.1148775, upper bound: 0.1156917
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 14.37
Output dim: 2, lower bound: -0.1154551, upper bound: 0.1154552
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 14.37
Output dim: 2, lower bound: -0.1154551, upper bound: 0.1154552
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 14.37
Output dim: 2, lower bound: -0.1154551, upper bound: 0.1156917
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 14.37
Output dim: 2, lower bound: -0.1154551, upper bound: 0.1156917

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -5.4835501, -5.1501546, -5.4891157, -5.1465549, -0.2790971, 0.2785373
1: -1.9258330, -1.6652412, -1.9263667, -1.6649526, -0.1339183, 0.1344873
2: 7.6604800, 8.0296545, 7.6574106, 8.0298271, -0.2978454, 0.2969632
3: -3.1856155, -2.7705901, -3.1844687, -2.7695522, -0.1820998, 0.1819578
4: -10.6142282, -10.1813469, -10.6156120, -10.1812153, -0.1918638, 0.1946220
5: -8.4543200, -7.8962078, -8.4509830, -7.8930469, -0.3009009, 0.2986536
6: -7.3147612, -6.8633838, -7.3110771, -6.8599772, -0.1951231, 0.1970478
7: -5.2884808, -4.8449969, -5.2913980, -4.8467565, -0.1971822, 0.1993780
8: -0.9831343, -0.7281075, -0.9832644, -0.7281630, -0.1466113, 0.1465293
9: -11.4136963, -11.0496063, -11.4160032, -11.0470905, -0.1702358, 0.1691353

Time for backsubstitution: 8.69 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 214
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 753
type: B, layer: 3, pos: 1110
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 1262
type: B, layer: 3, pos: 2322
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2466
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 2928

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 214

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1148775, upper bound: 0.1149587
time: 2.80 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1148775, upper bound: 0.1157219
time: 2.75 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -5.4859552, -5.1417789, -5.4891157, -5.1465549, -0.2807326, 0.2878909
1: -1.9259741, -1.6643723, -1.9263667, -1.6649526, -0.1341739, 0.1348308
2: 7.6615896, 8.0352373, 7.6574106, 8.0298271, -0.2997575, 0.3075078
3: -3.1885753, -2.7623458, -3.1844687, -2.7695522, -0.1842117, 0.1894587
4: -10.6293364, -10.1761131, -10.6156120, -10.1812153, -0.2065353, 0.1990578
5: -8.4581470, -7.8862681, -8.4509830, -7.8930469, -0.3022132, 0.3065445
6: -7.3152833, -6.8600216, -7.3110771, -6.8599772, -0.1964673, 0.2013406
7: -5.3044419, -4.8407202, -5.2913980, -4.8467565, -0.2142990, 0.2044890
8: -0.9849634, -0.7199860, -0.9832644, -0.7281630, -0.1487594, 0.1557833
9: -11.4196634, -11.0475101, -11.4160032, -11.0470905, -0.1763099, 0.1708680

Time for backsubstitution: 8.00 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 214
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 753
type: B, layer: 3, pos: 1110
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 1262
type: B, layer: 3, pos: 2322
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2466
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 2928

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 214

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1148775, upper bound: 0.1149587
time: 3.42 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1148775, upper bound: 0.1157219
time: 2.76 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -5.4835501, -5.1501546, -5.4916697, -5.1381798, -0.2884502, 0.2803540
1: -1.9258330, -1.6652412, -1.9265063, -1.6640851, -0.1342615, 0.1347326
2: 7.6604800, 8.0296545, 7.6585398, 8.0354118, -0.3083897, 0.2988572
3: -3.1856155, -2.7705901, -3.1874235, -2.7613077, -0.1896007, 0.1842020
4: -10.6142282, -10.1813469, -10.6307173, -10.1759806, -0.1963189, 0.2092957
5: -8.4543200, -7.8962078, -8.4548206, -7.8831081, -0.3087926, 0.3000252
6: -7.3147612, -6.8633838, -7.3116012, -6.8566141, -0.1994191, 0.1984046
7: -5.2884808, -4.8449969, -5.3073578, -4.8424640, -0.2023683, 0.2164955
8: -0.9831343, -0.7281075, -0.9850926, -0.7200425, -0.1558658, 0.1486772
9: -11.4136963, -11.0496063, -11.4219694, -11.0449848, -0.1719594, 0.1752093

Time for backsubstitution: 8.06 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 214
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 753
type: B, layer: 3, pos: 1110
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 1262
type: B, layer: 3, pos: 2322
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2466
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 2928

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 214

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1148775, upper bound: 0.1149284
time: 2.90 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1148775, upper bound: 0.1156918
time: 3.07 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -5.4859552, -5.1417789, -5.4916697, -5.1381798, -0.2888098, 0.2884309
1: -1.9259741, -1.6643723, -1.9265063, -1.6640851, -0.1357218, 0.1362814
2: 7.6615896, 8.0352373, 7.6585398, 8.0354118, -0.3103781, 0.3094761
3: -3.1885753, -2.7623458, -3.1874235, -2.7613077, -0.1873749, 0.1873701
4: -10.6293364, -10.1761131, -10.6307173, -10.1759806, -0.1995010, 0.2022398
5: -8.4581470, -7.8862681, -8.4548206, -7.8831081, -0.3096638, 0.3074749
6: -7.3152833, -6.8600216, -7.3116012, -6.8566141, -0.1988890, 0.2008252
7: -5.3044419, -4.8407202, -5.3073578, -4.8424640, -0.2054009, 0.2075229
8: -0.9849634, -0.7199860, -0.9850926, -0.7200425, -0.1517564, 0.1516743
9: -11.4196634, -11.0475101, -11.4219694, -11.0449848, -0.1754501, 0.1743590

Time for backsubstitution: 8.73 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 214
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 753
type: B, layer: 3, pos: 1110
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 1262
type: B, layer: 3, pos: 2322
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2466
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 2928

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 214

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1148775, upper bound: 0.1146920
time: 3.00 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1148775, upper bound: 0.1154551
time: 3.34 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -5.4891257, -5.1458297, -5.4891276, -5.1450992, -0.2825108, 0.2797546
1: -1.9263735, -1.6651080, -1.9263744, -1.6648983, -0.1352704, 0.1339370
2: 7.6587009, 8.0298290, 7.6565309, 8.0298309, -0.2932975, 0.3057566
3: -3.1844854, -2.7713082, -3.1844876, -2.7697256, -0.1834898, 0.1810704
4: -10.6151552, -10.1812153, -10.6158323, -10.1812143, -0.1951778, 0.1919575
5: -8.4509830, -7.8977890, -8.4509850, -7.8934221, -0.3023400, 0.2978077
6: -7.3110790, -6.8651123, -7.3110771, -6.8603754, -0.2093589, 0.1858200
7: -5.2883997, -4.8467569, -5.2912922, -4.8467541, -0.1971395, 0.2011938
8: -0.9830761, -0.7281637, -0.9832497, -0.7281651, -0.1462189, 0.1470560
9: -11.4168367, -11.0470829, -11.4171391, -11.0470829, -0.1678914, 0.1737500

Time for backsubstitution: 8.04 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 214
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 753
type: B, layer: 3, pos: 1110
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 1262
type: B, layer: 3, pos: 2322
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2466
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 2928

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 214

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1154550, upper bound: 0.1149587
time: 2.89 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1154552, upper bound: 0.1149587
time: 2.84 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -5.4916801, -5.1374531, -5.4891276, -5.1450992, -0.2843294, 0.2891092
1: -1.9265141, -1.6642396, -1.9263744, -1.6648983, -0.1355163, 0.1342800
2: 7.6598158, 8.0354118, 7.6565309, 8.0298309, -0.2952120, 0.3163023
3: -3.1874406, -2.7630651, -3.1844876, -2.7697256, -0.1857346, 0.1885713
4: -10.6302633, -10.1759806, -10.6158323, -10.1812143, -0.2098515, 0.1964126
5: -8.4548254, -7.8878503, -8.4509850, -7.8934221, -0.3037157, 0.3056993
6: -7.3116012, -6.8617506, -7.3110771, -6.8603754, -0.2107170, 0.1901160
7: -5.3043594, -4.8424635, -5.2912922, -4.8467541, -0.2142570, 0.2063818
8: -0.9849036, -0.7200422, -0.9832497, -0.7281651, -0.1483672, 0.1563102
9: -11.4228039, -11.0449791, -11.4171391, -11.0470829, -0.1739657, 0.1754743

Time for backsubstitution: 8.58 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 214
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 753
type: B, layer: 3, pos: 1110
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 1262
type: B, layer: 3, pos: 2322
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2466
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 2928

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 214

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1154550, upper bound: 0.1149587
time: 2.90 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1154552, upper bound: 0.1149587
time: 2.89 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5.4891257, -5.1458297, -5.4916821, -5.1367240, -0.2918653, 0.2813902
1: -1.9263735, -1.6651080, -1.9265153, -1.6640310, -0.1356132, 0.1341828
2: 7.6587009, 8.0298290, 7.6576481, 8.0354128, -0.3038430, 0.3076458
3: -3.1844854, -2.7713082, -3.1874430, -2.7614830, -0.1909906, 0.1831763
4: -10.6151552, -10.1812153, -10.6309395, -10.1759787, -0.1996126, 0.2066309
5: -8.4509830, -7.8977890, -8.4548244, -7.8834820, -0.3102317, 0.2991207
6: -7.3110790, -6.8651123, -7.3116045, -6.8570137, -0.2136548, 0.1871650
7: -5.2883997, -4.8467569, -5.3072519, -4.8424621, -0.2022507, 0.2183113
8: -0.9830761, -0.7281637, -0.9850771, -0.7200427, -0.1554731, 0.1492039
9: -11.4168367, -11.0470829, -11.4231071, -11.0449753, -0.1696217, 0.1798239

Time for backsubstitution: 8.63 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 214
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 753
type: B, layer: 3, pos: 1110
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 1262
type: B, layer: 3, pos: 2322
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2466
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 2928

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 214

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1154549, upper bound: 0.1149284
time: 3.13 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1154550, upper bound: 0.1149282
time: 2.98 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -5.4916801, -5.1374531, -5.4916821, -5.1367240, -0.2924061, 0.2894669
1: -1.9265141, -1.6642396, -1.9265153, -1.6640310, -0.1370643, 0.1357309
2: 7.6598158, 8.0354118, 7.6576481, 8.0354128, -0.3058319, 0.3182657
3: -3.1874406, -2.7630651, -3.1874430, -2.7614830, -0.1889023, 0.1863438
4: -10.6302633, -10.1759806, -10.6309395, -10.1759787, -0.2027948, 0.1995947
5: -8.4548254, -7.8878503, -8.4548244, -7.8834820, -0.3111660, 0.3065708
6: -7.3116012, -6.8617506, -7.3116045, -6.8570137, -0.2131386, 0.1895868
7: -5.3043594, -4.8424635, -5.3072519, -4.8424621, -0.2052836, 0.2094150
8: -0.9849036, -0.7200422, -0.9850771, -0.7200427, -0.1513637, 0.1522005
9: -11.4228039, -11.0449791, -11.4231071, -11.0449753, -0.1731126, 0.1789649

Time for backsubstitution: 7.97 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 214
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 753
type: B, layer: 3, pos: 1110
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 1262
type: B, layer: 3, pos: 2322
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2466
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 2928

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 214

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1154549, upper bound: 0.1146919
time: 3.20 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1154550, upper bound: 0.1146918
time: 2.96 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 14.37 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.37
Output dim: 2, lower bound: -0.1148775, upper bound: 0.1149587
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.37
Output dim: 2, lower bound: -0.1148775, upper bound: 0.1157219
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.37
Output dim: 2, lower bound: -0.1148775, upper bound: 0.1149587
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.37
Output dim: 2, lower bound: -0.1148775, upper bound: 0.1157219
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.37
Output dim: 2, lower bound: -0.1148775, upper bound: 0.1149284
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.37
Output dim: 2, lower bound: -0.1148775, upper bound: 0.1156918
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 14.37
Output dim: 2, lower bound: -0.1148775, upper bound: 0.1146920
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.37
Output dim: 2, lower bound: -0.1148775, upper bound: 0.1154551
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.37
Output dim: 2, lower bound: -0.1154550, upper bound: 0.1149587
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.37
Output dim: 2, lower bound: -0.1154552, upper bound: 0.1149587
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.37
Output dim: 2, lower bound: -0.1154550, upper bound: 0.1149587
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.37
Output dim: 2, lower bound: -0.1154552, upper bound: 0.1149587
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.37
Output dim: 2, lower bound: -0.1154549, upper bound: 0.1149284
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.37
Output dim: 2, lower bound: -0.1154550, upper bound: 0.1149282
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.37
Output dim: 2, lower bound: -0.1154549, upper bound: 0.1146919
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.37
Output dim: 2, lower bound: -0.1154550, upper bound: 0.1146918

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -5.4835501, -5.1501546, -5.4835501, -5.1501546, -0.2760262, 0.2760265
1: -1.9258330, -1.6652412, -1.9258330, -1.6652412, -0.1334106, 0.1334106
2: 7.6604800, 8.0296545, 7.6604800, 8.0296545, -0.2919459, 0.2919459
3: -3.1856155, -2.7705901, -3.1856155, -2.7705901, -0.1810141, 0.1810141
4: -10.6142282, -10.1813469, -10.6142282, -10.1813469, -0.1940355, 0.1940355
5: -8.4543200, -7.8962078, -8.4543200, -7.8962078, -0.2984679, 0.2984679
6: -7.3147612, -6.8633838, -7.3147612, -6.8633838, -0.1871482, 0.1871482
7: -5.2884808, -4.8449969, -5.2884808, -4.8449969, -0.1965899, 0.1965897
8: -0.9831343, -0.7281075, -0.9831343, -0.7281075, -0.1462345, 0.1462345
9: -11.4136963, -11.0496063, -11.4136963, -11.0496063, -0.1668999, 0.1669000

Time for backsubstitution: 8.05 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 753
type: A, layer: 3, pos: 1110
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1262
type: A, layer: 3, pos: 2322
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 2466
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 2928

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 60

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1156867, upper bound: 0.1149828
time: 3.29 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1160634, upper bound: 0.1158658
time: 3.49 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -5.4835501, -5.1501546, -5.4891257, -5.1458297, -0.2783422, 0.2785368
1: -1.9258330, -1.6652412, -1.9263735, -1.6651080, -0.1341759, 0.1344973
2: 7.6604800, 8.0296545, 7.6587009, 8.0298290, -0.2978477, 0.2995501
3: -3.1856155, -2.7705901, -3.1844854, -2.7713082, -0.1819386, 0.1819646
4: -10.6142282, -10.1813469, -10.6151552, -10.1812153, -0.1918671, 0.1928957
5: -8.4543200, -7.8962078, -8.4509830, -7.8977890, -0.2995596, 0.2985988
6: -7.3147612, -6.8633838, -7.3110790, -6.8651123, -0.1993699, 0.1970211
7: -5.2884808, -4.8449969, -5.2883997, -4.8467569, -0.1971512, 0.1987822
8: -0.9831343, -0.7281075, -0.9830761, -0.7281637, -0.1466117, 0.1466621
9: -11.4136963, -11.0496063, -11.4168367, -11.0470829, -0.1702409, 0.1703720

Time for backsubstitution: 8.07 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 753
type: A, layer: 3, pos: 1110
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1262
type: A, layer: 3, pos: 2322
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 2466
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 2928

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 60

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1156867, upper bound: 0.1156858
time: 3.25 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1160634, upper bound: 0.1166181
time: 3.29 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -5.4859552, -5.1417789, -5.4835501, -5.1501546, -0.2776623, 0.2853796
1: -1.9259741, -1.6643723, -1.9258330, -1.6652412, -0.1336660, 0.1337541
2: 7.6615896, 8.0352373, 7.6604800, 8.0296545, -0.2938581, 0.3024902
3: -3.1885753, -2.7623458, -3.1856155, -2.7705901, -0.1831260, 0.1885149
4: -10.6293364, -10.1761131, -10.6142282, -10.1813469, -0.2087071, 0.1984713
5: -8.4581470, -7.8862681, -8.4543200, -7.8962078, -0.2997799, 0.3063588
6: -7.3152833, -6.8600216, -7.3147612, -6.8633838, -0.1884922, 0.1914409
7: -5.3044419, -4.8407202, -5.2884808, -4.8449969, -0.2137070, 0.2017009
8: -0.9849634, -0.7199860, -0.9831343, -0.7281075, -0.1483825, 0.1554885
9: -11.4196634, -11.0475101, -11.4136963, -11.0496063, -0.1729742, 0.1686326

Time for backsubstitution: 8.71 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 753
type: A, layer: 3, pos: 1110
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1262
type: A, layer: 3, pos: 2322
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 2466
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 2928

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 60

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1140423, upper bound: 0.1133606
time: 2.97 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1146877, upper bound: 0.1145255
time: 2.95 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -5.4859552, -5.1417789, -5.4891257, -5.1458297, -0.2799778, 0.2878900
1: -1.9259741, -1.6643723, -1.9263735, -1.6651080, -0.1344314, 0.1348407
2: 7.6615896, 8.0352373, 7.6587009, 8.0298290, -0.2997599, 0.3100944
3: -3.1885753, -2.7623458, -3.1844854, -2.7713082, -0.1840506, 0.1894655
4: -10.6293364, -10.1761131, -10.6151552, -10.1812153, -0.2065389, 0.1973312
5: -8.4581470, -7.8862681, -8.4509830, -7.8977890, -0.3008716, 0.3064895
6: -7.3152833, -6.8600216, -7.3110790, -6.8651123, -0.2007141, 0.2013139
7: -5.3044419, -4.8407202, -5.2883997, -4.8467569, -0.2142682, 0.2038932
8: -0.9849634, -0.7199860, -0.9830761, -0.7281637, -0.1487598, 0.1559161
9: -11.4196634, -11.0475101, -11.4168367, -11.0470829, -0.1763148, 0.1721046

Time for backsubstitution: 7.96 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 753
type: A, layer: 3, pos: 1110
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1262
type: A, layer: 3, pos: 2322
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 2466
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 2928

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 60

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1140423, upper bound: 0.1140636
time: 2.91 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1146877, upper bound: 0.1152952
time: 2.98 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -5.4835501, -5.1501546, -5.4859552, -5.1417789, -0.2853794, 0.2776618
1: -1.9258330, -1.6652412, -1.9259741, -1.6643723, -0.1337541, 0.1336660
2: 7.6604800, 8.0296545, 7.6615896, 8.0352373, -0.3024902, 0.2938581
3: -3.1856155, -2.7705901, -3.1885753, -2.7623458, -0.1885149, 0.1831259
4: -10.6142282, -10.1813469, -10.6293364, -10.1761131, -0.1984713, 0.2087071
5: -8.4543200, -7.8962078, -8.4581470, -7.8862681, -0.3063588, 0.2997801
6: -7.3147612, -6.8633838, -7.3152833, -6.8600216, -0.1914411, 0.1884922
7: -5.2884808, -4.8449969, -5.3044419, -4.8407202, -0.2017009, 0.2137070
8: -0.9831343, -0.7281075, -0.9849634, -0.7199860, -0.1554885, 0.1483825
9: -11.4136963, -11.0496063, -11.4196634, -11.0475101, -0.1686327, 0.1729743

Time for backsubstitution: 8.01 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 753
type: A, layer: 3, pos: 1110
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1262
type: A, layer: 3, pos: 2322
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 2466
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 2928

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 60

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1140726, upper bound: 0.1133303
time: 2.82 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1147181, upper bound: 0.1144952
time: 2.92 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -5.4835501, -5.1501546, -5.4916801, -5.1374531, -0.2876959, 0.2803550
1: -1.9258330, -1.6652412, -1.9265141, -1.6642396, -0.1345190, 0.1347432
2: 7.6604800, 8.0296545, 7.6598158, 8.0354118, -0.3083935, 0.3014393
3: -3.1856155, -2.7705901, -3.1874406, -2.7630651, -0.1894398, 0.1842095
4: -10.6142282, -10.1813469, -10.6302633, -10.1759806, -0.1963222, 0.2075691
5: -8.4543200, -7.8962078, -8.4548254, -7.8878503, -0.3074512, 0.2999744
6: -7.3147612, -6.8633838, -7.3116012, -6.8617506, -0.2036660, 0.1983793
7: -5.2884808, -4.8449969, -5.3043594, -4.8424635, -0.2023392, 0.2158995
8: -0.9831343, -0.7281075, -0.9849036, -0.7200422, -0.1558659, 0.1488103
9: -11.4136963, -11.0496063, -11.4228039, -11.0449791, -0.1719649, 0.1764462

Time for backsubstitution: 8.05 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 753
type: A, layer: 3, pos: 1110
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1262
type: A, layer: 3, pos: 2322
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 2466
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 2928

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 60

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1140726, upper bound: 0.1140333
time: 2.81 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1147181, upper bound: 0.1152649
time: 3.04 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -5.4859552, -5.1417789, -5.4916801, -5.1374531, -0.2880545, 0.2884316
1: -1.9259741, -1.6643723, -1.9265141, -1.6642396, -0.1359794, 0.1362920
2: 7.6615896, 8.0352373, 7.6598158, 8.0354118, -0.3103800, 0.3120575
3: -3.1885753, -2.7623458, -3.1874406, -2.7630651, -0.1872143, 0.1873771
4: -10.6293364, -10.1761131, -10.6302633, -10.1759806, -0.1995044, 0.2005138
5: -8.4581470, -7.8862681, -8.4548254, -7.8878503, -0.3083220, 0.3074234
6: -7.3152833, -6.8600216, -7.3116012, -6.8617506, -0.2031357, 0.2007997
7: -5.3044419, -4.8407202, -5.3043594, -4.8424635, -0.2053721, 0.2069266
8: -0.9849634, -0.7199860, -0.9849036, -0.7200422, -0.1517565, 0.1518068
9: -11.4196634, -11.0475101, -11.4228039, -11.0449791, -0.1754558, 0.1755958

Time for backsubstitution: 8.07 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 753
type: A, layer: 3, pos: 1110
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1262
type: A, layer: 3, pos: 2322
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 2466
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 2928

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 60

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1140423, upper bound: 0.1137968
time: 3.06 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1146877, upper bound: 0.1150284
time: 3.21 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -5.4891257, -5.1458297, -5.4835501, -5.1501546, -0.2785368, 0.2783420
1: -1.9263735, -1.6651080, -1.9258330, -1.6652412, -0.1344973, 0.1341759
2: 7.6587009, 8.0298290, 7.6604800, 8.0296545, -0.2995501, 0.2978482
3: -3.1844854, -2.7713082, -3.1856155, -2.7705901, -0.1819646, 0.1819386
4: -10.6151552, -10.1812153, -10.6142282, -10.1813469, -0.1928957, 0.1918671
5: -8.4509830, -7.8977890, -8.4543200, -7.8962078, -0.2985988, 0.2995596
6: -7.3110790, -6.8651123, -7.3147612, -6.8633838, -0.1970211, 0.1993700
7: -5.2883997, -4.8467569, -5.2884808, -4.8449969, -0.1987820, 0.1971509
8: -0.9830761, -0.7281637, -0.9831343, -0.7281075, -0.1466621, 0.1466117
9: -11.4168367, -11.0470829, -11.4136963, -11.0496063, -0.1703719, 0.1702406

Time for backsubstitution: 8.05 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 753
type: A, layer: 3, pos: 1110
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1262
type: A, layer: 3, pos: 2322
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 2466
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 2928

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 60

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1161251, upper bound: 0.1149617
time: 2.86 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1166179, upper bound: 0.1158656
time: 3.29 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -5.4891257, -5.1458297, -5.4891257, -5.1458297, -0.2797451, 0.2797449
1: -1.9263735, -1.6651080, -1.9263735, -1.6651080, -0.1339360, 0.1339360
2: 7.6587009, 8.0298290, 7.6587009, 8.0298290, -0.2932973, 0.2932975
3: -3.1844854, -2.7713082, -3.1844854, -2.7713082, -0.1810690, 0.1810690
4: -10.6151552, -10.1812153, -10.6151552, -10.1812153, -0.1951764, 0.1951764
5: -8.4509830, -7.8977890, -8.4509830, -7.8977890, -0.2977896, 0.2977896
6: -7.3110790, -6.8651123, -7.3110790, -6.8651123, -0.1858051, 0.1858051
7: -5.2883997, -4.8467569, -5.2883997, -4.8467569, -0.1971314, 0.1971314
8: -0.9830761, -0.7281637, -0.9830761, -0.7281637, -0.1462190, 0.1462190
9: -11.4168367, -11.0470829, -11.4168367, -11.0470829, -0.1678904, 0.1678903

Time for backsubstitution: 7.95 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 753
type: A, layer: 3, pos: 1110
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1262
type: A, layer: 3, pos: 2322
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 2466
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 2928

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 60

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1161252, upper bound: 0.1149616
time: 3.17 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1166181, upper bound: 0.1158656
time: 3.16 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -5.4916801, -5.1374531, -5.4835501, -5.1501546, -0.2803550, 0.2876961
1: -1.9265141, -1.6642396, -1.9258330, -1.6652412, -0.1347432, 0.1345190
2: 7.6598158, 8.0354118, 7.6604800, 8.0296545, -0.3014393, 0.3083940
3: -3.1874406, -2.7630651, -3.1856155, -2.7705901, -0.1842095, 0.1894398
4: -10.6302633, -10.1759806, -10.6142282, -10.1813469, -0.2075691, 0.1963220
5: -8.4548254, -7.8878503, -8.4543200, -7.8962078, -0.2999744, 0.3074512
6: -7.3116012, -6.8617506, -7.3147612, -6.8633838, -0.1983792, 0.2036661
7: -5.3043594, -4.8424635, -5.2884808, -4.8449969, -0.2158995, 0.2023392
8: -0.9849036, -0.7200422, -0.9831343, -0.7281075, -0.1488101, 0.1558659
9: -11.4228039, -11.0449791, -11.4136963, -11.0496063, -0.1764462, 0.1719649

Time for backsubstitution: 7.95 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 753
type: A, layer: 3, pos: 1110
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1262
type: A, layer: 3, pos: 2322
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 2466
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 2928

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 60

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1144725, upper bound: 0.1130148
time: 3.34 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1152649, upper bound: 0.1145255
time: 3.15 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -5.4916801, -5.1374531, -5.4891257, -5.1458297, -0.2813821, 0.2890995
1: -1.9265141, -1.6642396, -1.9263735, -1.6651080, -0.1341820, 0.1342789
2: 7.6598158, 8.0354118, 7.6587009, 8.0298290, -0.2952123, 0.3038421
3: -3.1874406, -2.7630651, -3.1844854, -2.7713082, -0.1831745, 0.1885699
4: -10.6302633, -10.1759806, -10.6151552, -10.1812153, -0.2098501, 0.1996114
5: -8.4548254, -7.8878503, -8.4509830, -7.8977890, -0.2991030, 0.3056812
6: -7.3116012, -6.8617506, -7.3110790, -6.8651123, -0.1871505, 0.1901011
7: -5.3043594, -4.8424635, -5.2883997, -4.8467569, -0.2142489, 0.2022426
8: -0.9849036, -0.7200422, -0.9830761, -0.7281637, -0.1483673, 0.1554731
9: -11.4228039, -11.0449791, -11.4168367, -11.0470829, -0.1739647, 0.1696205

Time for backsubstitution: 7.98 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 753
type: A, layer: 3, pos: 1110
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1262
type: A, layer: 3, pos: 2322
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 2466
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 2928

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 60

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1144724, upper bound: 0.1130145
time: 3.13 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1152651, upper bound: 0.1145255
time: 2.89 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -5.4891257, -5.1458297, -5.4859552, -5.1417789, -0.2878900, 0.2799778
1: -1.9263735, -1.6651080, -1.9259741, -1.6643723, -0.1348408, 0.1344314
2: 7.6587009, 8.0298290, 7.6615896, 8.0352373, -0.3100939, 0.2997603
3: -3.1844854, -2.7713082, -3.1885753, -2.7623458, -0.1894655, 0.1840507
4: -10.6151552, -10.1812153, -10.6293364, -10.1761131, -0.1973312, 0.2065389
5: -8.4509830, -7.8977890, -8.4581470, -7.8862681, -0.3064895, 0.3008716
6: -7.3110790, -6.8651123, -7.3152833, -6.8600216, -0.2013140, 0.2007140
7: -5.2883997, -4.8467569, -5.3044419, -4.8407202, -0.2038932, 0.2142682
8: -0.9830761, -0.7281637, -0.9849634, -0.7199860, -0.1559161, 0.1487598
9: -11.4168367, -11.0470829, -11.4196634, -11.0475101, -0.1721046, 0.1763151

Time for backsubstitution: 8.01 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 753
type: A, layer: 3, pos: 1110
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1262
type: A, layer: 3, pos: 2322
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 2466
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 2928

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 60

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1145029, upper bound: 0.1133091
time: 2.90 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1152950, upper bound: 0.1144951
time: 2.79 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -5.4891257, -5.1458297, -5.4916801, -5.1374531, -0.2890997, 0.2813821
1: -1.9263735, -1.6651080, -1.9265141, -1.6642396, -0.1342789, 0.1341820
2: 7.6587009, 8.0298290, 7.6598158, 8.0354118, -0.3038421, 0.2952120
3: -3.1844854, -2.7713082, -3.1874406, -2.7630651, -0.1885699, 0.1831745
4: -10.6151552, -10.1812153, -10.6302633, -10.1759806, -0.1996114, 0.2098501
5: -8.4509830, -7.8977890, -8.4548254, -7.8878503, -0.3056815, 0.2991030
6: -7.3110790, -6.8651123, -7.3116012, -6.8617506, -0.1901011, 0.1871505
7: -5.2883997, -4.8467569, -5.3043594, -4.8424635, -0.2022426, 0.2142489
8: -0.9830761, -0.7281637, -0.9849036, -0.7200422, -0.1554731, 0.1483673
9: -11.4168367, -11.0470829, -11.4228039, -11.0449791, -0.1696206, 0.1739650

Time for backsubstitution: 8.71 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 753
type: A, layer: 3, pos: 1110
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1262
type: A, layer: 3, pos: 2322
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 2466
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 2928

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 60

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1145030, upper bound: 0.1133092
time: 3.15 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1152954, upper bound: 0.1144952
time: 2.80 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -5.4916801, -5.1374531, -5.4859552, -5.1417789, -0.2884312, 0.2880545
1: -1.9265141, -1.6642396, -1.9259741, -1.6643723, -0.1362920, 0.1359794
2: 7.6598158, 8.0354118, 7.6615896, 8.0352373, -0.3120580, 0.3103802
3: -3.1874406, -2.7630651, -3.1885753, -2.7623458, -0.1873771, 0.1872143
4: -10.6302633, -10.1759806, -10.6293364, -10.1761131, -0.2005138, 0.1995044
5: -8.4548254, -7.8878503, -8.4581470, -7.8862681, -0.3074231, 0.3083220
6: -7.3116012, -6.8617506, -7.3152833, -6.8600216, -0.2007998, 0.2031358
7: -5.3043594, -4.8424635, -5.3044419, -4.8407202, -0.2069266, 0.2053721
8: -0.9849036, -0.7200422, -0.9849634, -0.7199860, -0.1518068, 0.1517566
9: -11.4228039, -11.0449791, -11.4196634, -11.0475101, -0.1755955, 0.1754558

Time for backsubstitution: 8.81 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 753
type: A, layer: 3, pos: 1110
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1262
type: A, layer: 3, pos: 2322
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 2466
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 2928

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 60

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1144725, upper bound: 0.1130726
time: 3.10 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1152649, upper bound: 0.1142587
time: 2.96 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -5.4916801, -5.1374531, -5.4916801, -5.1374531, -0.2894592, 0.2894597
1: -1.9265141, -1.6642396, -1.9265141, -1.6642396, -0.1357298, 0.1357298
2: 7.6598158, 8.0354118, 7.6598158, 8.0354118, -0.3058319, 0.3058319
3: -3.1874406, -2.7630651, -3.1874406, -2.7630651, -0.1863424, 0.1863424
4: -10.6302633, -10.1759806, -10.6302633, -10.1759806, -0.2027938, 0.2027938
5: -8.4548254, -7.8878503, -8.4548254, -7.8878503, -0.3065531, 0.3065534
6: -7.3116012, -6.8617506, -7.3116012, -6.8617506, -0.1895723, 0.1895723
7: -5.3043594, -4.8424635, -5.3043594, -4.8424635, -0.2052760, 0.2052760
8: -0.9849036, -0.7200422, -0.9849036, -0.7200422, -0.1513636, 0.1513636
9: -11.4228039, -11.0449791, -11.4228039, -11.0449791, -0.1731118, 0.1731119

Time for backsubstitution: 8.74 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 753
type: A, layer: 3, pos: 1110
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1262
type: A, layer: 3, pos: 2322
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 2466
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 2928

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 60

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1144727, upper bound: 0.1129496
time: 3.08 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1152651, upper bound: 0.1142587
time: 2.85 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 14.95 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.95
Output dim: 2, lower bound: -0.1156867, upper bound: 0.1149828
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.95
Output dim: 2, lower bound: -0.1160634, upper bound: 0.1158658
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.95
Output dim: 2, lower bound: -0.1156867, upper bound: 0.1156858
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.95
Output dim: 2, lower bound: -0.1160634, upper bound: 0.1166181
NS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 14.95
Output dim: 2, lower bound: -0.1140423, upper bound: 0.1133606
NS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 14.95
Output dim: 2, lower bound: -0.1146877, upper bound: 0.1145255
NS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 14.95
Output dim: 2, lower bound: -0.1140423, upper bound: 0.1140636
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.95
Output dim: 2, lower bound: -0.1146877, upper bound: 0.1152952
NS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 14.95
Output dim: 2, lower bound: -0.1140726, upper bound: 0.1133303
NS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 14.95
Output dim: 2, lower bound: -0.1147181, upper bound: 0.1144952
NS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 14.95
Output dim: 2, lower bound: -0.1140726, upper bound: 0.1140333
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.95
Output dim: 2, lower bound: -0.1147181, upper bound: 0.1152649
NS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 14.95
Output dim: 2, lower bound: -0.1140423, upper bound: 0.1137968
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.95
Output dim: 2, lower bound: -0.1146877, upper bound: 0.1150284
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.95
Output dim: 2, lower bound: -0.1161251, upper bound: 0.1149617
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.95
Output dim: 2, lower bound: -0.1166179, upper bound: 0.1158656
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.95
Output dim: 2, lower bound: -0.1161252, upper bound: 0.1149616
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.95
Output dim: 2, lower bound: -0.1166181, upper bound: 0.1158656
NS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 14.95
Output dim: 2, lower bound: -0.1144725, upper bound: 0.1130148
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.95
Output dim: 2, lower bound: -0.1152649, upper bound: 0.1145255
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 14.95
Output dim: 2, lower bound: -0.1144724, upper bound: 0.1130145
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.95
Output dim: 2, lower bound: -0.1152651, upper bound: 0.1145255
NS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 14.95
Output dim: 2, lower bound: -0.1145029, upper bound: 0.1133091
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.95
Output dim: 2, lower bound: -0.1152950, upper bound: 0.1144951
NS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 14.95
Output dim: 2, lower bound: -0.1145030, upper bound: 0.1133092
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.95
Output dim: 2, lower bound: -0.1152954, upper bound: 0.1144952
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 14.95
Output dim: 2, lower bound: -0.1144725, upper bound: 0.1130726
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.95
Output dim: 2, lower bound: -0.1152649, upper bound: 0.1142587
NS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 14.95
Output dim: 2, lower bound: -0.1144727, upper bound: 0.1129496
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.95
Output dim: 2, lower bound: -0.1152651, upper bound: 0.1142587

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -5.4829016, -5.1498408, -5.4833865, -5.1501546, -0.2756042, 0.2764878
1: -1.9260569, -1.6662039, -1.9257997, -1.6654820, -0.1339166, 0.1327637
2: 7.6583147, 8.0262480, 7.6604834, 8.0286942, -0.2912605, 0.2871437
3: -3.1844504, -2.7698069, -3.1853158, -2.7705901, -0.1795887, 0.1814086
4: -10.6144342, -10.1815834, -10.6142035, -10.1814060, -0.1942117, 0.1937752
5: -8.4524937, -7.8918533, -8.4538631, -7.8962169, -0.2962146, 0.3017793
6: -7.3135624, -6.8623362, -7.3144612, -6.8633900, -0.1858832, 0.1878825
7: -5.2853251, -4.8436260, -5.2875786, -4.8449965, -0.1919532, 0.1950712
8: -0.9831507, -0.7281718, -0.9831345, -0.7281241, -0.1462528, 0.1461781
9: -11.4135990, -11.0495739, -11.4136715, -11.0496225, -0.1667597, 0.1668643

Time for backsubstitution: 8.73 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 753
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 1110
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 1262
type: B, layer: 3, pos: 2322
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2466
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 2928

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 753

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1135917, upper bound: 0.0981086
time: 2.99 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1151287, upper bound: 0.1145789
time: 3.24 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -5.4834557, -5.1501546, -5.4835501, -5.1501546, -0.2763557, 0.2760255
1: -1.9258116, -1.6653305, -1.9258330, -1.6652412, -0.1333903, 0.1337254
2: 7.6604815, 8.0292835, 7.6604800, 8.0296545, -0.2919459, 0.2879972
3: -3.1854429, -2.7705901, -3.1856155, -2.7705901, -0.1804844, 0.1810136
4: -10.6142120, -10.1814070, -10.6142282, -10.1813469, -0.1940210, 0.1941390
5: -8.4535122, -7.8962135, -8.4543200, -7.8962078, -0.2992227, 0.2984610
6: -7.3145413, -6.8633890, -7.3147612, -6.8633838, -0.1869576, 0.1871427
7: -5.2882185, -4.8449974, -5.2884808, -4.8449969, -0.1921740, 0.1965897
8: -0.9831352, -0.7281139, -0.9831343, -0.7281075, -0.1462340, 0.1462198
9: -11.4136763, -11.0496187, -11.4136963, -11.0496063, -0.1668659, 0.1668862

Time for backsubstitution: 8.65 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 753
type: B, layer: 3, pos: 1110
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 1262
type: B, layer: 3, pos: 2322
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2466
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 2928

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 60

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1150191, upper bound: 0.1157468
time: 2.91 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1150191, upper bound: 0.1161205
time: 2.92 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -5.4829016, -5.1498408, -5.4889584, -5.1458297, -0.2779207, 0.2789819
1: -1.9260569, -1.6662039, -1.9263418, -1.6653483, -0.1346819, 0.1338476
2: 7.6583147, 8.0262480, 7.6587019, 8.0289173, -0.2973380, 0.2947471
3: -3.1844504, -2.7698069, -3.1841850, -2.7713082, -0.1805134, 0.1823434
4: -10.6144342, -10.1815834, -10.6151304, -10.1812744, -0.1920431, 0.1926355
5: -8.4524937, -7.8918533, -8.4503670, -7.8977966, -0.2973094, 0.3017507
6: -7.3135624, -6.8623362, -7.3106918, -6.8651185, -0.1981051, 0.1978010
7: -5.2853251, -4.8436260, -5.2876086, -4.8467565, -0.1925144, 0.1973991
8: -0.9831507, -0.7281718, -0.9830744, -0.7281806, -0.1466304, 0.1466061
9: -11.4135990, -11.0495739, -11.4168081, -11.0470991, -0.1701007, 0.1703361

Time for backsubstitution: 8.60 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 753
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 1110
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 1262
type: B, layer: 3, pos: 2322
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2466
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 2928

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 753

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1131710, upper bound: 0.0876176
time: 2.83 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1150956, upper bound: 0.1152335
time: 3.18 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -5.4834557, -5.1501546, -5.4891257, -5.1458297, -0.2786727, 0.2785358
1: -1.9258116, -1.6653305, -1.9263735, -1.6651080, -0.1341558, 0.1347942
2: 7.6604815, 8.0292835, 7.6587009, 8.0298290, -0.2978482, 0.2957509
3: -3.1854429, -2.7705901, -3.1844854, -2.7713082, -0.1814437, 0.1819642
4: -10.6142120, -10.1814070, -10.6151552, -10.1812153, -0.1918526, 0.1929991
5: -8.4535122, -7.8962135, -8.4509830, -7.8977890, -0.2995691, 0.2985919
6: -7.3145413, -6.8633890, -7.3110790, -6.8651123, -0.1992440, 0.1970156
7: -5.2882185, -4.8449974, -5.2883997, -4.8467569, -0.1933978, 0.1987822
8: -0.9831352, -0.7281139, -0.9830761, -0.7281637, -0.1466113, 0.1466480
9: -11.4136763, -11.0496187, -11.4168367, -11.0470829, -0.1702064, 0.1703581

Time for backsubstitution: 8.56 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 753
type: B, layer: 3, pos: 1110
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 1262
type: B, layer: 3, pos: 2322
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2466
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 2928

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 60

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1149616, upper bound: 0.1161251
time: 2.89 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1149616, upper bound: 0.1161251
time: 2.98 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -5.4858627, -5.1417789, -5.4891257, -5.1458297, -0.2803087, 0.2878890
1: -1.9259524, -1.6644635, -1.9263735, -1.6651080, -0.1344118, 0.1351206
2: 7.6615887, 8.0348644, 7.6587009, 8.0298290, -0.2997599, 0.3062952
3: -3.1884019, -2.7623458, -3.1844854, -2.7713082, -0.1835555, 0.1894650
4: -10.6293192, -10.1761713, -10.6151552, -10.1812153, -0.2065246, 0.1974347
5: -8.4573383, -7.8862748, -8.4509830, -7.8977890, -0.3008821, 0.3064823
6: -7.3150625, -6.8600273, -7.3110790, -6.8651123, -0.2005916, 0.2013080
7: -5.3041763, -4.8407202, -5.2883997, -4.8467569, -0.2105148, 0.2038929
8: -0.9849627, -0.7199924, -0.9830761, -0.7281637, -0.1487598, 0.1559020
9: -11.4196434, -11.0475197, -11.4168367, -11.0470829, -0.1762806, 0.1720908

Time for backsubstitution: 8.54 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 753
type: B, layer: 3, pos: 1110
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 1262
type: B, layer: 3, pos: 2322
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2466
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 2928

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 60

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1133091, upper bound: 0.1145030
time: 2.99 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1133091, upper bound: 0.1152954
time: 3.10 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -5.4834557, -5.1501546, -5.4916801, -5.1374531, -0.2880259, 0.2803540
1: -1.9258116, -1.6653305, -1.9265141, -1.6642396, -0.1344988, 0.1350400
2: 7.6604815, 8.0292835, 7.6598158, 8.0354118, -0.3083935, 0.2976234
3: -3.1854429, -2.7705901, -3.1874406, -2.7630651, -0.1889447, 0.1842091
4: -10.6142120, -10.1814070, -10.6302633, -10.1759806, -0.1963074, 0.2076726
5: -8.4535122, -7.8962135, -8.4548254, -7.8878503, -0.3074605, 0.2999675
6: -7.3145413, -6.8633890, -7.3116012, -6.8617506, -0.2035401, 0.1983736
7: -5.2882185, -4.8449974, -5.3043594, -4.8424635, -0.1985846, 0.2158995
8: -0.9831352, -0.7281139, -0.9849036, -0.7200422, -0.1558656, 0.1487962
9: -11.4136763, -11.0496187, -11.4228039, -11.0449791, -0.1719303, 0.1764324

Time for backsubstitution: 8.61 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 753
type: B, layer: 3, pos: 1110
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 1262
type: B, layer: 3, pos: 2322
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2466
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 2928

Time for candidate selection: 0.35 seconds

### Candidate
type: B, layer: 3, pos: 60

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1133394, upper bound: 0.1144725
time: 3.18 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1133394, upper bound: 0.1152649
time: 2.95 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -5.4858627, -5.1417789, -5.4916801, -5.1374531, -0.2883849, 0.2884307
1: -1.9259524, -1.6644635, -1.9265141, -1.6642396, -0.1359597, 0.1365718
2: 7.6615887, 8.0348644, 7.6598158, 8.0354118, -0.3103790, 0.3082418
3: -3.1884019, -2.7623458, -3.1874406, -2.7630651, -0.1867185, 0.1873766
4: -10.6293192, -10.1761713, -10.6302633, -10.1759806, -0.1994903, 0.2006176
5: -8.4573383, -7.8862748, -8.4548254, -7.8878503, -0.3083329, 0.3074160
6: -7.3150625, -6.8600273, -7.3116012, -6.8617506, -0.2030137, 0.2007940
7: -5.3041763, -4.8407202, -5.3043594, -4.8424635, -0.2016177, 0.2069259
8: -0.9849627, -0.7199924, -0.9849036, -0.7200422, -0.1517565, 0.1517932
9: -11.4196434, -11.0475197, -11.4228039, -11.0449791, -0.1754215, 0.1755817

Time for backsubstitution: 8.09 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 753
type: B, layer: 3, pos: 1110
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 1262
type: B, layer: 3, pos: 2322
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2466
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 2928

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 60

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1133091, upper bound: 0.1142360
time: 3.40 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1133091, upper bound: 0.1150284
time: 3.39 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -5.4884610, -5.1455159, -5.4833865, -5.1501546, -0.2780499, 0.2788038
1: -1.9265993, -1.6660705, -1.9257997, -1.6654820, -0.1349769, 0.1335287
2: 7.6562433, 8.0266609, 7.6604834, 8.0286942, -0.2990139, 0.2933593
3: -3.1833081, -2.7705367, -3.1853158, -2.7705901, -0.1805170, 0.1823688
4: -10.6153612, -10.1814537, -10.6142035, -10.1814060, -0.1930745, 0.1916070
5: -8.4489994, -7.8934317, -8.4538631, -7.8962169, -0.2957096, 0.3021393
6: -7.3097897, -6.8640652, -7.3144612, -6.8633900, -0.1958810, 0.2001719
7: -5.2857137, -4.8448420, -5.2875786, -4.8449965, -0.1947246, 0.1962955
8: -0.9830925, -0.7282274, -0.9831345, -0.7281241, -0.1466806, 0.1465555
9: -11.4167252, -11.0470657, -11.4136715, -11.0496225, -0.1702316, 0.1702058

Time for backsubstitution: 8.67 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 753
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 1110
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 1262
type: B, layer: 3, pos: 2322
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2466
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 2928

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 753

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1135766, upper bound: 0.0957722
time: 2.96 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1155672, upper bound: 0.1145065
time: 3.22 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.98 + 547.20 = 604.18 seconds
