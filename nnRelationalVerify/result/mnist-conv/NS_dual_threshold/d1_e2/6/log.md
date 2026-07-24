## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.107825006


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2641456, 0.2641456)
1: (-15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.2365746, 0.2365746)
2: (-8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1989318, 0.1989318)
3: (-7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.2008582, 0.2008582)
4: (-6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2459675, 0.2459674)
5: (-1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1788921, 0.1788921)
6: (-15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.2207536, 0.2207536)
7: (-5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2624643, 0.2624643)
8: (-2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1839114, 0.1839114)
9: (2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1294789, 0.1294789)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.45 + 33.31 = 56.76 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.1111598, upper bound: 0.1111597

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1195
type: B, layer: 3, pos: 1195
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 416
type: B, layer: 3, pos: 416
type: A, layer: 3, pos: 228
type: B, layer: 3, pos: 228
type: A, layer: 3, pos: 1808
type: B, layer: 3, pos: 1808
type: A, layer: 3, pos: 659
type: B, layer: 3, pos: 659
type: A, layer: 3, pos: 2229
type: B, layer: 3, pos: 2229

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1095381, upper bound: 0.1101964
time: 2.92 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1101965, upper bound: 0.1101964
time: 3.32 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 6.52 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 6.52
Output dim: 9, lower bound: -0.1095381, upper bound: 0.1101964
NS_A2, status: Status.UNKNOWN, split count: 1, time: 6.52
Output dim: 9, lower bound: -0.1101965, upper bound: 0.1101964

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -14.1044760, -13.4014311, -14.1147242, -13.4014311, -0.2479776, 0.2580040
1: -15.8250790, -15.2642174, -15.8250790, -15.2625113, -0.2352421, 0.2326288
2: -8.7455387, -8.2218332, -8.7464924, -8.2218304, -0.1976731, 0.1983783
3: -7.1936679, -6.6151047, -7.1936679, -6.6129360, -0.1992239, 0.1958808
4: -6.7397442, -6.1860542, -6.7397547, -6.1855960, -0.2457808, 0.2454202
5: -1.6634974, -1.0764687, -1.6634974, -1.0734663, -0.1768541, 0.1733942
6: -15.9410648, -15.3196239, -15.9493217, -15.3196201, -0.2028114, 0.2148820
7: -5.4759989, -4.9604001, -5.4760113, -4.9572453, -0.2609068, 0.2573967
8: -2.3088870, -1.8050098, -2.3089485, -1.8050084, -0.1838062, 0.1838732
9: 2.5448291, 2.8984504, 2.5430226, 2.8984499, -0.1268311, 0.1285330

Time for backsubstitution: 8.30 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1195
type: B, layer: 3, pos: 1195
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 416
type: B, layer: 3, pos: 416
type: A, layer: 3, pos: 228
type: B, layer: 3, pos: 228
type: A, layer: 3, pos: 1808
type: B, layer: 3, pos: 1808
type: A, layer: 3, pos: 659
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 1836

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1095381, upper bound: 0.1095380
time: 4.30 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1095381, upper bound: 0.1101965
time: 3.09 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -14.1119051, -13.3889809, -14.1163921, -13.4014282, -0.2529613, 0.2821805
1: -15.8253345, -15.2650700, -15.8250790, -15.2629147, -0.2411702, 0.2324684
2: -8.7451077, -8.2213078, -8.7462196, -8.2218313, -0.1977832, 0.2014550
3: -7.1935987, -6.6163902, -7.1936679, -6.6135387, -0.2044026, 0.1958444
4: -6.7403646, -6.1856775, -6.7397494, -6.1854916, -0.2461416, 0.2456565
5: -1.6660295, -1.0754595, -1.6634974, -1.0732968, -0.1857916, 0.1743458
6: -15.9449377, -15.3116083, -15.9500713, -15.3196220, -0.2057784, 0.2435850
7: -5.4805431, -4.9569149, -5.4760089, -4.9563427, -0.2675211, 0.2594428
8: -2.3089042, -1.8049545, -2.3089528, -1.8050084, -0.1838245, 0.1840364
9: 2.5442443, 2.9001536, 2.5429401, 2.8984499, -0.1274616, 0.1336396

Time for backsubstitution: 7.89 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1195
type: B, layer: 3, pos: 1195
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 416
type: B, layer: 3, pos: 416
type: A, layer: 3, pos: 228
type: B, layer: 3, pos: 228
type: A, layer: 3, pos: 1808
type: B, layer: 3, pos: 1808
type: A, layer: 3, pos: 659
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 1195

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1094568, upper bound: 0.1096598
time: 2.97 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1096598, upper bound: 0.1096598
time: 2.91 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 13.97 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 13.97
Output dim: 9, lower bound: -0.1095381, upper bound: 0.1095380
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 13.97
Output dim: 9, lower bound: -0.1095381, upper bound: 0.1101965
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 13.97
Output dim: 9, lower bound: -0.1094568, upper bound: 0.1096598
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 13.97
Output dim: 9, lower bound: -0.1096598, upper bound: 0.1096598

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -14.1044760, -13.4014311, -14.1044760, -13.4014311, -0.2479761, 0.2479761
1: -15.8250790, -15.2642174, -15.8250790, -15.2642174, -0.2326288, 0.2326288
2: -8.7455387, -8.2218332, -8.7455387, -8.2218332, -0.1976726, 0.1976726
3: -7.1936679, -6.6151047, -7.1936679, -6.6151047, -0.1958807, 0.1958808
4: -6.7397442, -6.1860542, -6.7397442, -6.1860542, -0.2454193, 0.2454193
5: -1.6634974, -1.0764687, -1.6634974, -1.0764687, -0.1733942, 0.1733942
6: -15.9410648, -15.3196239, -15.9410648, -15.3196239, -0.2028107, 0.2028108
7: -5.4759989, -4.9604001, -5.4759989, -4.9604001, -0.2573944, 0.2573946
8: -2.3088870, -1.8050098, -2.3088870, -1.8050098, -0.1838049, 0.1838049
9: 2.5448291, 2.8984504, 2.5448291, 2.8984504, -0.1268311, 0.1268310

Time for backsubstitution: 7.86 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1195
type: B, layer: 3, pos: 1195
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 416
type: B, layer: 3, pos: 416
type: A, layer: 3, pos: 228
type: B, layer: 3, pos: 228
type: A, layer: 3, pos: 1808
type: B, layer: 3, pos: 1808
type: A, layer: 3, pos: 659
type: B, layer: 3, pos: 659
type: A, layer: 3, pos: 2229
type: B, layer: 3, pos: 2229

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 1195

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1087984, upper bound: 0.1092353
time: 3.24 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1090014, upper bound: 0.1092353
time: 3.13 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -14.1044760, -13.4014311, -14.1119051, -13.3889809, -0.2666469, 0.2633084
1: -15.8250790, -15.2642174, -15.8253345, -15.2650700, -0.2357388, 0.2375339
2: -8.7455387, -8.2218332, -8.7451077, -8.2213078, -0.2005808, 0.1981854
3: -7.1936679, -6.6151047, -7.1935987, -6.6163902, -0.1997706, 0.1998817
4: -6.7397442, -6.1860542, -6.7403646, -6.1856775, -0.2456653, 0.2457001
5: -1.6634974, -1.0764687, -1.6660295, -1.0754595, -0.1787900, 0.1803270
6: -15.9410648, -15.3196239, -15.9449377, -15.3116083, -0.2256439, 0.2207507
7: -5.4759989, -4.9604001, -5.4805431, -4.9569149, -0.2616777, 0.2627287
8: -2.3088870, -1.8050098, -2.3089042, -1.8049545, -0.1839429, 0.1838824
9: 2.5448291, 2.8984504, 2.5442443, 2.9001536, -0.1313666, 0.1285340

Time for backsubstitution: 8.11 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1195
type: B, layer: 3, pos: 1195
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1109
type: B, layer: 3, pos: 416
type: A, layer: 3, pos: 416
type: A, layer: 3, pos: 228
type: B, layer: 3, pos: 228
type: B, layer: 3, pos: 1808
type: A, layer: 3, pos: 1808
type: B, layer: 3, pos: 659
type: A, layer: 3, pos: 659
type: A, layer: 3, pos: 2229
type: B, layer: 3, pos: 2229

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 1195

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1087984, upper bound: 0.1096598
time: 3.12 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1090014, upper bound: 0.1096598
time: 3.38 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -14.1119022, -13.3910112, -14.1163921, -13.4020567, -0.2521237, 0.2803206
1: -15.8253345, -15.2665348, -15.8250790, -15.2633667, -0.2402308, 0.2303944
2: -8.7451086, -8.2220554, -8.7462206, -8.2220631, -0.1970134, 0.1995193
3: -7.1935987, -6.6204462, -7.1936679, -6.6148024, -0.2025356, 0.1901934
4: -6.7403626, -6.1878090, -6.7397509, -6.1861501, -0.2454302, 0.2433624
5: -1.6597555, -1.0754588, -1.6612520, -1.0732965, -0.1761854, 0.1713754
6: -15.9428062, -15.3116093, -15.9494133, -15.3196220, -0.2029862, 0.2427217
7: -5.4805412, -4.9616933, -5.4760094, -4.9578476, -0.2657106, 0.2535942
8: -2.3089070, -1.8054209, -2.3089542, -1.8051529, -0.1836262, 0.1835307
9: 2.5455329, 2.9001541, 2.5433383, 2.8984499, -0.1265537, 0.1332803

Time for backsubstitution: 7.98 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 416
type: B, layer: 3, pos: 416
type: A, layer: 3, pos: 228
type: B, layer: 3, pos: 228
type: A, layer: 3, pos: 1808
type: B, layer: 3, pos: 1808
type: B, layer: 3, pos: 1195
type: A, layer: 3, pos: 659
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 1836

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1094568, upper bound: 0.1090012
time: 3.81 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1094568, upper bound: 0.1090013
time: 3.08 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -14.1137276, -13.3898735, -14.1163931, -13.4017906, -0.2563034, 0.2810855
1: -15.8250771, -15.2674665, -15.8250790, -15.2637501, -0.2452331, 0.2300951
2: -8.7441711, -8.2234879, -8.7462215, -8.2225924, -0.1999456, 0.1989272
3: -7.1977587, -6.6174426, -7.1936679, -6.6139793, -0.2116377, 0.1919789
4: -6.7428951, -6.1858354, -6.7397513, -6.1855593, -0.2480412, 0.2444584
5: -1.6632323, -1.0701327, -1.6625223, -1.0732970, -0.1783913, 0.1856314
6: -15.9441481, -15.3096170, -15.9497948, -15.3196201, -0.2037911, 0.2466493
7: -5.4853096, -4.9580870, -5.4760084, -4.9567504, -0.2729943, 0.2556796
8: -2.3088336, -1.8058491, -2.3089547, -1.8053203, -0.1850371, 0.1834308
9: 2.5454135, 2.9009118, 2.5433483, 2.8984492, -0.1268104, 0.1358747

Time for backsubstitution: 8.64 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 416
type: B, layer: 3, pos: 416
type: A, layer: 3, pos: 228
type: B, layer: 3, pos: 228
type: A, layer: 3, pos: 1808
type: B, layer: 3, pos: 1808
type: B, layer: 3, pos: 1195
type: A, layer: 3, pos: 659
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 1836

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1096598, upper bound: 0.1090013
time: 3.17 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1096598, upper bound: 0.1090013
time: 3.07 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 15.03 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 15.03
Output dim: 9, lower bound: -0.1087984, upper bound: 0.1092353
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 15.03
Output dim: 9, lower bound: -0.1090014, upper bound: 0.1092353
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 15.03
Output dim: 9, lower bound: -0.1087984, upper bound: 0.1096598
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 15.03
Output dim: 9, lower bound: -0.1090014, upper bound: 0.1096598
NS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 15.03
Output dim: 9, lower bound: -0.1094568, upper bound: 0.1090012
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 15.03
Output dim: 9, lower bound: -0.1094568, upper bound: 0.1090013
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 15.03
Output dim: 9, lower bound: -0.1096598, upper bound: 0.1090013
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 15.03
Output dim: 9, lower bound: -0.1096598, upper bound: 0.1090013

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -14.1044769, -13.4034624, -14.1044750, -13.4020596, -0.2471386, 0.2461164
1: -15.8250790, -15.2656851, -15.8250790, -15.2646723, -0.2316897, 0.2305541
2: -8.7455387, -8.2225819, -8.7455378, -8.2220631, -0.1969030, 0.1957370
3: -7.1936679, -6.6191616, -7.1936679, -6.6163678, -0.1940139, 0.1902300
4: -6.7397451, -6.1881866, -6.7397447, -6.1867127, -0.2447083, 0.2431250
5: -1.6572220, -1.0764701, -1.6612520, -1.0764689, -0.1637880, 0.1704239
6: -15.9389334, -15.3196220, -15.9404049, -15.3196239, -0.2000188, 0.2019473
7: -5.4759998, -4.9651780, -5.4760003, -4.9619083, -0.2555850, 0.2515457
8: -2.3088851, -1.8054771, -2.3088851, -1.8051548, -0.1836065, 0.1832992
9: 2.5461173, 2.8984494, 2.5452268, 2.8984501, -0.1259233, 0.1264721

Time for backsubstitution: 8.10 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1109
type: B, layer: 3, pos: 416
type: A, layer: 3, pos: 416
type: A, layer: 3, pos: 228
type: B, layer: 3, pos: 228
type: B, layer: 3, pos: 1808
type: A, layer: 3, pos: 1808
type: B, layer: 3, pos: 1195
type: B, layer: 3, pos: 659
type: A, layer: 3, pos: 659
type: A, layer: 3, pos: 2229
type: B, layer: 3, pos: 2229

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 1109

## Relational analysis of NS_A1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1064829, upper bound: 0.1089662
time: 3.43 seconds

## Relational analysis of NS_A1_B1_A1_A2

### Relational analysis result of NS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1087632, upper bound: 0.1089663
time: 3.19 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -14.1062946, -13.4023247, -14.1044769, -13.4017897, -0.2513182, 0.2468809
1: -15.8248234, -15.2666159, -15.8250790, -15.2650557, -0.2366917, 0.2302554
2: -8.7446022, -8.2240133, -8.7455378, -8.2225933, -0.1998351, 0.1951449
3: -7.1978269, -6.6161561, -7.1936679, -6.6155453, -0.2031157, 0.1920154
4: -6.7422748, -6.1862125, -6.7397437, -6.1861219, -0.2473193, 0.2442210
5: -1.6607018, -1.0711398, -1.6625223, -1.0764701, -0.1659939, 0.1846799
6: -15.9402733, -15.3176336, -15.9407883, -15.3196239, -0.2008237, 0.2058748
7: -5.4807663, -4.9615726, -5.4759998, -4.9608097, -0.2628680, 0.2536316
8: -2.3088112, -1.8059034, -2.3088851, -1.8053212, -0.1850177, 0.1831995
9: 2.5459991, 2.8992074, 2.5452363, 2.8984497, -0.1261802, 0.1290661

Time for backsubstitution: 8.64 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1109
type: B, layer: 3, pos: 416
type: A, layer: 3, pos: 416
type: B, layer: 3, pos: 228
type: A, layer: 3, pos: 228
type: B, layer: 3, pos: 1808
type: A, layer: 3, pos: 1808
type: B, layer: 3, pos: 1195
type: B, layer: 3, pos: 659
type: A, layer: 3, pos: 659
type: A, layer: 3, pos: 2229
type: B, layer: 3, pos: 2229

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1109

## Relational analysis of NS_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1066859, upper bound: 0.1089662
time: 3.25 seconds

## Relational analysis of NS_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1089662, upper bound: 0.1089661
time: 3.10 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -14.1044769, -13.4034624, -14.1119061, -13.3896084, -0.2658098, 0.2614486
1: -15.8250790, -15.2656851, -15.8253345, -15.2655239, -0.2347994, 0.2354592
2: -8.7455387, -8.2225819, -8.7451096, -8.2215405, -0.1998113, 0.1962498
3: -7.1936679, -6.6191616, -7.1935987, -6.6176529, -0.1979034, 0.1942309
4: -6.7397451, -6.1881866, -6.7403631, -6.1863365, -0.2449545, 0.2434053
5: -1.6572220, -1.0764701, -1.6637826, -1.0754604, -0.1691837, 0.1773566
6: -15.9389334, -15.3196220, -15.9442797, -15.3116064, -0.2228518, 0.2198873
7: -5.4759998, -4.9651780, -5.4805422, -4.9584217, -0.2598681, 0.2568796
8: -2.3088851, -1.8054771, -2.3089056, -1.8050985, -0.1837443, 0.1833768
9: 2.5461173, 2.8984494, 2.5446434, 2.9001536, -0.1304588, 0.1281746

Time for backsubstitution: 8.59 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1109
type: B, layer: 3, pos: 416
type: A, layer: 3, pos: 416
type: B, layer: 3, pos: 228
type: A, layer: 3, pos: 228
type: B, layer: 3, pos: 1808
type: A, layer: 3, pos: 1808
type: B, layer: 3, pos: 1195
type: B, layer: 3, pos: 659
type: A, layer: 3, pos: 659
type: A, layer: 3, pos: 2229
type: B, layer: 3, pos: 2229

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 1109

## Relational analysis of NS_A1_B2_A1_A1

### Relational analysis result of NS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1062096, upper bound: 0.1093907
time: 3.56 seconds

## Relational analysis of NS_A1_B2_A1_A2

### Relational analysis result of NS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1085293, upper bound: 0.1093907
time: 3.10 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -14.1062946, -13.4023247, -14.1119051, -13.3893414, -0.2699888, 0.2622137
1: -15.8248234, -15.2666159, -15.8253345, -15.2659063, -0.2398012, 0.2351605
2: -8.7446022, -8.2240133, -8.7451086, -8.2220688, -0.2027432, 0.1956577
3: -7.1978269, -6.6161561, -7.1935987, -6.6168308, -0.2070053, 0.1960162
4: -6.7422748, -6.1862125, -6.7403636, -6.1857443, -0.2475654, 0.2445009
5: -1.6607018, -1.0711398, -1.6650543, -1.0754614, -0.1713896, 0.1916126
6: -15.9402733, -15.3176336, -15.9446602, -15.3116074, -0.2236569, 0.2238147
7: -5.4807663, -4.9615726, -5.4805417, -4.9573245, -0.2671511, 0.2589645
8: -2.3088112, -1.8059034, -2.3089056, -1.8052678, -0.1851550, 0.1832771
9: 2.5459991, 2.8992074, 2.5446529, 2.9001541, -0.1307157, 0.1307685

Time for backsubstitution: 8.68 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1109
type: B, layer: 3, pos: 416
type: A, layer: 3, pos: 416
type: B, layer: 3, pos: 228
type: A, layer: 3, pos: 228
type: B, layer: 3, pos: 1808
type: A, layer: 3, pos: 1808
type: B, layer: 3, pos: 1195
type: B, layer: 3, pos: 659
type: A, layer: 3, pos: 659
type: A, layer: 3, pos: 2229
type: B, layer: 3, pos: 2229

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1109

## Relational analysis of NS_A1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1064125, upper bound: 0.1093907
time: 3.63 seconds

## Relational analysis of NS_A1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1087322, upper bound: 0.1093907
time: 3.16 seconds

## BFS NS instance: NS_A2_A1_B1

### Backsubstitution after applying NS history:
0: -14.1119022, -13.3910112, -14.1044750, -13.4020596, -0.2624713, 0.2647871
1: -15.8253345, -15.2665348, -15.8250790, -15.2646723, -0.2365949, 0.2336640
2: -8.7451086, -8.2220554, -8.7455378, -8.2220631, -0.1974158, 0.1986454
3: -7.1935987, -6.6204462, -7.1936679, -6.6163678, -0.1980147, 0.1941196
4: -6.7403626, -6.1878090, -6.7397447, -6.1867127, -0.2449887, 0.2433710
5: -1.6597555, -1.0754588, -1.6612520, -1.0764689, -0.1707208, 0.1758196
6: -15.9428062, -15.3116093, -15.9404049, -15.3196239, -0.2179584, 0.2247804
7: -5.4805412, -4.9616933, -5.4760003, -4.9619083, -0.2609181, 0.2558289
8: -2.3089070, -1.8054209, -2.3088851, -1.8051548, -0.1836843, 0.1834368
9: 2.5455329, 2.9001541, 2.5452268, 2.8984501, -0.1276262, 0.1310076

Time for backsubstitution: 8.71 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 416
type: B, layer: 3, pos: 416
type: A, layer: 3, pos: 228
type: B, layer: 3, pos: 228
type: A, layer: 3, pos: 1808
type: B, layer: 3, pos: 1808
type: B, layer: 3, pos: 1195
type: A, layer: 3, pos: 659
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 3, pos: 1109

## Relational analysis of NS_A2_A1_B1_B1

### Relational analysis result of NS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1091877, upper bound: 0.1064126
time: 3.44 seconds

## Relational analysis of NS_A2_A1_B1_B2

### Relational analysis result of NS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1091877, upper bound: 0.1087322
time: 3.44 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: -14.1119022, -13.3910112, -14.1119061, -13.3896084, -0.2521148, 0.2510924
1: -15.8253345, -15.2665348, -15.8253345, -15.2655239, -0.2315295, 0.2303944
2: -8.7451086, -8.2220554, -8.7451096, -8.2215405, -0.1970168, 0.1958508
3: -7.1935987, -6.6204462, -7.1935987, -6.6176529, -0.1939774, 0.1901934
4: -6.7403626, -6.1878090, -6.7403631, -6.1863365, -0.2449672, 0.2433839
5: -1.6597555, -1.0754588, -1.6637826, -1.0754604, -0.1647395, 0.1713754
6: -15.9428062, -15.3116093, -15.9442797, -15.3116064, -0.2029949, 0.2049233
7: -5.4805412, -4.9616933, -5.4805422, -4.9584217, -0.2577090, 0.2536700
8: -2.3089070, -1.8054209, -2.3089056, -1.8050985, -0.1836419, 0.1833344
9: 2.5455329, 2.9001541, 2.5446434, 2.9001536, -0.1265537, 0.1271023

Time for backsubstitution: 8.16 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1109
type: B, layer: 3, pos: 416
type: A, layer: 3, pos: 416
type: A, layer: 3, pos: 228
type: B, layer: 3, pos: 228
type: B, layer: 3, pos: 1808
type: A, layer: 3, pos: 1808
type: B, layer: 3, pos: 1195
type: B, layer: 3, pos: 659
type: A, layer: 3, pos: 659
type: A, layer: 3, pos: 2229
type: B, layer: 3, pos: 2229

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 1109

## Relational analysis of NS_A2_A1_B2_A1

### Relational analysis result of NS_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1069460, upper bound: 0.1087323
time: 3.51 seconds

## Relational analysis of NS_A2_A1_B2_A2

### Relational analysis result of NS_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1091877, upper bound: 0.1087322
time: 3.11 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: -14.1137276, -13.3898735, -14.1044769, -13.4017897, -0.2666509, 0.2655518
1: -15.8250771, -15.2674665, -15.8250790, -15.2650557, -0.2415969, 0.2333655
2: -8.7441711, -8.2234879, -8.7455378, -8.2225933, -0.2003478, 0.1980532
3: -7.1977587, -6.6174426, -7.1936679, -6.6155453, -0.2071167, 0.1959052
4: -6.7428951, -6.1858354, -6.7397437, -6.1861219, -0.2475995, 0.2444668
5: -1.6632323, -1.0701327, -1.6625223, -1.0764701, -0.1729267, 0.1900756
6: -15.9441481, -15.3096170, -15.9407883, -15.3196239, -0.2187635, 0.2287081
7: -5.4853096, -4.9580870, -5.4759998, -4.9608097, -0.2682017, 0.2579148
8: -2.3088336, -1.8058491, -2.3088851, -1.8053212, -0.1850955, 0.1833371
9: 2.5454135, 2.9009118, 2.5452363, 2.8984497, -0.1278828, 0.1336016

Time for backsubstitution: 8.73 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 416
type: B, layer: 3, pos: 416
type: B, layer: 3, pos: 228
type: A, layer: 3, pos: 228
type: A, layer: 3, pos: 1808
type: B, layer: 3, pos: 1808
type: B, layer: 3, pos: 1195
type: A, layer: 3, pos: 659
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1109

## Relational analysis of NS_A2_A2_B1_B1

### Relational analysis result of NS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1093907, upper bound: 0.1064124
time: 3.73 seconds

## Relational analysis of NS_A2_A2_B1_B2

### Relational analysis result of NS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1093907, upper bound: 0.1087322
time: 3.06 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: -14.1137276, -13.3898735, -14.1119051, -13.3893414, -0.2562941, 0.2518568
1: -15.8250771, -15.2674665, -15.8253345, -15.2659063, -0.2365310, 0.2300951
2: -8.7441711, -8.2234879, -8.7451086, -8.2220688, -0.1999490, 0.1952589
3: -7.1977587, -6.6174426, -7.1935987, -6.6168308, -0.2030791, 0.1919789
4: -6.7428951, -6.1858354, -6.7403636, -6.1857443, -0.2475783, 0.2444794
5: -1.6632323, -1.0701327, -1.6650543, -1.0754614, -0.1669454, 0.1856314
6: -15.9441481, -15.3096170, -15.9446602, -15.3116074, -0.2037998, 0.2088510
7: -5.4853096, -4.9580870, -5.4805417, -4.9573245, -0.2649921, 0.2557552
8: -2.3088336, -1.8058491, -2.3089056, -1.8052678, -0.1850526, 0.1832347
9: 2.5454135, 2.9009118, 2.5446529, 2.9001541, -0.1268105, 0.1296961

Time for backsubstitution: 8.73 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1109
type: B, layer: 3, pos: 416
type: A, layer: 3, pos: 416
type: B, layer: 3, pos: 228
type: A, layer: 3, pos: 228
type: B, layer: 3, pos: 1808
type: A, layer: 3, pos: 1808
type: B, layer: 3, pos: 1195
type: B, layer: 3, pos: 659
type: A, layer: 3, pos: 659
type: A, layer: 3, pos: 2229
type: B, layer: 3, pos: 2229

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 1109

## Relational analysis of NS_A2_A2_B2_A1

### Relational analysis result of NS_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1071490, upper bound: 0.1087322
time: 3.30 seconds

## Relational analysis of NS_A2_A2_B2_A2

### Relational analysis result of NS_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1093908, upper bound: 0.1087323
time: 3.13 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 15.35 seconds
NS_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 15.35
Output dim: 9, lower bound: -0.1064829, upper bound: 0.1089662
NS_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 15.35
Output dim: 9, lower bound: -0.1087632, upper bound: 0.1089663
NS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 15.35
Output dim: 9, lower bound: -0.1066859, upper bound: 0.1089662
NS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 15.35
Output dim: 9, lower bound: -0.1089662, upper bound: 0.1089661
NS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 15.35
Output dim: 9, lower bound: -0.1062096, upper bound: 0.1093907
NS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 15.35
Output dim: 9, lower bound: -0.1085293, upper bound: 0.1093907
NS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 15.35
Output dim: 9, lower bound: -0.1064125, upper bound: 0.1093907
NS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 15.35
Output dim: 9, lower bound: -0.1087322, upper bound: 0.1093907
NS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 15.35
Output dim: 9, lower bound: -0.1091877, upper bound: 0.1064126
NS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 15.35
Output dim: 9, lower bound: -0.1091877, upper bound: 0.1087322
NS_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 15.35
Output dim: 9, lower bound: -0.1069460, upper bound: 0.1087323
NS_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 15.35
Output dim: 9, lower bound: -0.1091877, upper bound: 0.1087322
NS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 15.35
Output dim: 9, lower bound: -0.1093907, upper bound: 0.1064124
NS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 15.35
Output dim: 9, lower bound: -0.1093907, upper bound: 0.1087322
NS_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 15.35
Output dim: 9, lower bound: -0.1071490, upper bound: 0.1087322
NS_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 15.35
Output dim: 9, lower bound: -0.1093908, upper bound: 0.1087323

## BFS NS instance: NS_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -14.0842171, -13.4058619, -14.0971928, -13.4020634, -0.2254453, 0.2281592
1: -15.8034105, -15.2628765, -15.8169365, -15.2646875, -0.1999631, 0.2067490
2: -8.7457695, -8.2256660, -8.7455368, -8.2231951, -0.1967191, 0.1923891
3: -7.1938410, -6.6340303, -7.1936679, -6.6219988, -0.1757548, 0.1698150
4: -6.7380743, -6.1873045, -6.7389445, -6.1867127, -0.2425919, 0.2415884
5: -1.6572695, -1.0792501, -1.6612520, -1.0774856, -0.1606038, 0.1661671
6: -15.9398899, -15.3249369, -15.9404049, -15.3215618, -0.1946548, 0.1944951
7: -5.4543276, -4.9687834, -5.4679637, -4.9619083, -0.2303073, 0.2301719
8: -2.3013415, -1.8278961, -2.3088851, -1.8136778, -0.1606088, 0.1593883
9: 2.5477068, 2.8983154, 2.5459177, 2.8984489, -0.1242456, 0.1252636

Time for backsubstitution: 8.80 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 416
type: B, layer: 3, pos: 416
type: B, layer: 3, pos: 228
type: A, layer: 3, pos: 228
type: A, layer: 3, pos: 1808
type: B, layer: 3, pos: 1808
type: B, layer: 3, pos: 1195
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 659
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 416

## Relational analysis of NS_A1_B1_A1_A1_A1

### Relational analysis result of NS_A1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1051631, upper bound: 0.1082878
time: 3.41 seconds

## Relational analysis of NS_A1_B1_A1_A1_A2

### Relational analysis result of NS_A1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1053710, upper bound: 0.1079255
time: 3.91 seconds

## BFS NS instance: NS_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -14.0995674, -13.4034786, -14.1034098, -13.4020624, -0.2211426, 0.2459350
1: -15.8179665, -15.2657318, -15.8235226, -15.2646818, -0.1920235, 0.2304071
2: -8.7455359, -8.2234735, -8.7455378, -8.2222767, -0.1967326, 0.1938927
3: -7.1936679, -6.6232281, -7.1936679, -6.6172504, -0.1939111, 0.1617860
4: -6.7387486, -6.1881866, -6.7395301, -6.1867127, -0.2411610, 0.2429845
5: -1.6572220, -1.0773938, -1.6612520, -1.0766702, -0.1637772, 0.1649625
6: -15.9389334, -15.3218231, -15.9404049, -15.3201008, -0.1999274, 0.1923222
7: -5.4719687, -4.9651780, -5.4751248, -4.9619083, -0.2230948, 0.2513666
8: -2.3088856, -1.8078070, -2.3088851, -1.8056612, -0.1835191, 0.1529337
9: 2.5465608, 2.8984489, 2.5453238, 2.8984492, -0.1237235, 0.1264447

Time for backsubstitution: 8.12 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 416
type: B, layer: 3, pos: 416
type: A, layer: 3, pos: 228
type: B, layer: 3, pos: 228
type: A, layer: 3, pos: 1808
type: B, layer: 3, pos: 1109
type: B, layer: 3, pos: 1808
type: B, layer: 3, pos: 1195
type: A, layer: 3, pos: 659
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 416

## Relational analysis of NS_A1_B1_A1_A2_A1

### Relational analysis result of NS_A1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1070844, upper bound: 0.1082878
time: 3.22 seconds

## Relational analysis of NS_A1_B1_A1_A2_A2

### Relational analysis result of NS_A1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1077225, upper bound: 0.1079256
time: 3.06 seconds

## BFS NS instance: NS_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -14.0860348, -13.4047241, -14.0971928, -13.4017963, -0.2296242, 0.2289234
1: -15.8031511, -15.2638092, -15.8169365, -15.2650681, -0.2049644, 0.2064496
2: -8.7448339, -8.2270994, -8.7455378, -8.2237263, -0.1996514, 0.1917970
3: -7.1979990, -6.6310248, -7.1936679, -6.6211762, -0.1848568, 0.1716006
4: -6.7406063, -6.1853323, -6.7389450, -6.1861219, -0.2452022, 0.2426848
5: -1.6607504, -1.0739193, -1.6625223, -1.0774865, -0.1628097, 0.1804221
6: -15.9412308, -15.3229475, -15.9407883, -15.3215599, -0.1954600, 0.1984210
7: -5.4590960, -4.9651761, -5.4679642, -4.9608097, -0.2375735, 0.2322578
8: -2.3012681, -1.8283234, -2.3088851, -1.8138447, -0.1620191, 0.1592889
9: 2.5475879, 2.8990731, 2.5459273, 2.8984492, -0.1245025, 0.1278575

Time for backsubstitution: 8.07 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 416
type: B, layer: 3, pos: 416
type: A, layer: 3, pos: 228
type: B, layer: 3, pos: 228
type: A, layer: 3, pos: 1808
type: B, layer: 3, pos: 1808
type: B, layer: 3, pos: 1195
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 659
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 416

## Relational analysis of NS_A1_B1_A2_A1_A1

### Relational analysis result of NS_A1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1053661, upper bound: 0.1082877
time: 3.21 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2

### Relational analysis result of NS_A1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1055736, upper bound: 0.1079255
time: 3.93 seconds

## BFS NS instance: NS_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -14.1013870, -13.4023447, -14.1034117, -13.4017954, -0.2253216, 0.2466993
1: -15.8177099, -15.2666626, -15.8235226, -15.2650661, -0.1970251, 0.2301081
2: -8.7446012, -8.2249050, -8.7455368, -8.2228060, -0.1996642, 0.1933006
3: -7.1978269, -6.6202240, -7.1936679, -6.6164274, -0.2030130, 0.1635715
4: -6.7412801, -6.1862125, -6.7395282, -6.1861219, -0.2437714, 0.2440808
5: -1.6607018, -1.0720634, -1.6625223, -1.0766697, -0.1659832, 0.1792182
6: -15.9402733, -15.3198299, -15.9407883, -15.3200989, -0.2007324, 0.1962494
7: -5.4767365, -4.9615726, -5.4751239, -4.9608097, -0.2303741, 0.2534525
8: -2.3088093, -1.8082314, -2.3088846, -1.8058276, -0.1849297, 0.1528344
9: 2.5464408, 2.8992076, 2.5453334, 2.8984494, -0.1239804, 0.1290387

Time for backsubstitution: 8.68 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 416
type: B, layer: 3, pos: 416
type: A, layer: 3, pos: 228
type: B, layer: 3, pos: 228
type: A, layer: 3, pos: 1808
type: B, layer: 3, pos: 1109
type: B, layer: 3, pos: 1808
type: B, layer: 3, pos: 1195
type: A, layer: 3, pos: 659
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 416

## Relational analysis of NS_A1_B1_A2_A2_A1

### Relational analysis result of NS_A1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1072874, upper bound: 0.1082878
time: 3.24 seconds

## Relational analysis of NS_A1_B1_A2_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1079255, upper bound: 0.1079256
time: 3.18 seconds

## BFS NS instance: NS_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -14.0842171, -13.4058619, -14.1046190, -13.3896132, -0.2441161, 0.2452455
1: -15.8034105, -15.2628765, -15.8171911, -15.2655354, -0.2030723, 0.2115682
2: -8.7457695, -8.2256660, -8.7451086, -8.2226715, -0.1996273, 0.1929020
3: -7.1938410, -6.6340303, -7.1935987, -6.6229353, -0.1803317, 0.1738158
4: -6.7380743, -6.1873045, -6.7395668, -6.1863365, -0.2428378, 0.2418690
5: -1.6572695, -1.0792501, -1.6637826, -1.0764790, -0.1663040, 0.1730998
6: -15.9398899, -15.3249369, -15.9442797, -15.3135481, -0.2173986, 0.2124351
7: -5.4543276, -4.9687834, -5.4725084, -4.9584217, -0.2345908, 0.2355046
8: -2.3013415, -1.8278961, -2.3089056, -1.8136220, -0.1607466, 0.1594660
9: 2.5477068, 2.8983154, 2.5453334, 2.9001541, -0.1287811, 0.1270157

Time for backsubstitution: 8.69 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 416
type: B, layer: 3, pos: 416
type: A, layer: 3, pos: 228
type: B, layer: 3, pos: 228
type: A, layer: 3, pos: 1808
type: B, layer: 3, pos: 1808
type: B, layer: 3, pos: 1195
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 659
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 416

## Relational analysis of NS_A1_B2_A1_A1_A1

### Relational analysis result of NS_A1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1049091, upper bound: 0.1087126
time: 3.36 seconds

## Relational analysis of NS_A1_B2_A1_A1_A2

### Relational analysis result of NS_A1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1051182, upper bound: 0.1083496
time: 3.84 seconds

## BFS NS instance: NS_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -14.0995674, -13.4034786, -14.1108379, -13.3896112, -0.2455506, 0.2606766
1: -15.8179665, -15.2657318, -15.8237753, -15.2655334, -0.1938149, 0.2353126
2: -8.7455359, -8.2234735, -8.7451057, -8.2217531, -0.1996410, 0.1941918
3: -7.1936679, -6.6232281, -7.1935987, -6.6187162, -0.1975908, 0.1694517
4: -6.7387486, -6.1881866, -6.7401485, -6.1863365, -0.2413822, 0.2432652
5: -1.6572220, -1.0773938, -1.6637826, -1.0756593, -0.1690683, 0.1731973
6: -15.9389334, -15.3218231, -15.9442797, -15.3120832, -0.2227935, 0.2098138
7: -5.4719687, -4.9651780, -5.4796686, -4.9584217, -0.2261037, 0.2567000
8: -2.3088856, -1.8078070, -2.3089080, -1.8056030, -0.1836567, 0.1530070
9: 2.5465608, 2.8984489, 2.5447388, 2.9001541, -0.1286422, 0.1280967

Time for backsubstitution: 8.66 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 416
type: B, layer: 3, pos: 416
type: A, layer: 3, pos: 228
type: B, layer: 3, pos: 228
type: A, layer: 3, pos: 1808
type: B, layer: 3, pos: 1109
type: B, layer: 3, pos: 1808
type: B, layer: 3, pos: 1195
type: A, layer: 3, pos: 659
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 416

## Relational analysis of NS_A1_B2_A1_A2_A1

### Relational analysis result of NS_A1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1069126, upper bound: 0.1087127
time: 3.58 seconds

## Relational analysis of NS_A1_B2_A1_A2_A2

### Relational analysis result of NS_A1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1074966, upper bound: 0.1083497
time: 3.74 seconds

## BFS NS instance: NS_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -14.0860348, -13.4047241, -14.1046200, -13.3893471, -0.2482946, 0.2460101
1: -15.8031511, -15.2638092, -15.8171911, -15.2659216, -0.2080736, 0.2112689
2: -8.7448339, -8.2270994, -8.7451086, -8.2232027, -0.2025595, 0.1923097
3: -7.1979990, -6.6310248, -7.1935987, -6.6221137, -0.1894336, 0.1756015
4: -6.7406063, -6.1853323, -6.7395654, -6.1857443, -0.2454482, 0.2429645
5: -1.6607504, -1.0739193, -1.6650543, -1.0764778, -0.1685100, 0.1873548
6: -15.9412308, -15.3229475, -15.9446602, -15.3135481, -0.2182038, 0.2163608
7: -5.4590960, -4.9651761, -5.4725065, -4.9573245, -0.2418568, 0.2375898
8: -2.3012681, -1.8283234, -2.3089080, -1.8137908, -0.1621567, 0.1593667
9: 2.5475879, 2.8990731, 2.5453444, 2.9001541, -0.1290381, 0.1296096

Time for backsubstitution: 8.60 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 416
type: B, layer: 3, pos: 416
type: A, layer: 3, pos: 228
type: B, layer: 3, pos: 228
type: A, layer: 3, pos: 1808
type: B, layer: 3, pos: 1808
type: B, layer: 3, pos: 1195
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 659
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 416

## Relational analysis of NS_A1_B2_A2_A1_A1

### Relational analysis result of NS_A1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1051121, upper bound: 0.1087127
time: 3.53 seconds

## Relational analysis of NS_A1_B2_A2_A1_A2

### Relational analysis result of NS_A1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1053209, upper bound: 0.1083496
time: 3.50 seconds

## BFS NS instance: NS_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -14.1013870, -13.4023447, -14.1108379, -13.3893471, -0.2497287, 0.2614410
1: -15.8177099, -15.2666626, -15.8237753, -15.2659178, -0.1988165, 0.2350134
2: -8.7446012, -8.2249050, -8.7451077, -8.2222805, -0.2025722, 0.1935998
3: -7.1978269, -6.6202240, -7.1935987, -6.6178951, -0.2066926, 0.1712372
4: -6.7412801, -6.1862125, -6.7401471, -6.1857443, -0.2439927, 0.2443607
5: -1.6607018, -1.0720634, -1.6650543, -1.0756621, -0.1712743, 0.1874528
6: -15.9402733, -15.3198299, -15.9446602, -15.3120842, -0.2235982, 0.2137411
7: -5.4767365, -4.9615726, -5.4796658, -4.9573245, -0.2333829, 0.2587850
8: -2.3088093, -1.8082314, -2.3089066, -1.8057714, -0.1850673, 0.1529077
9: 2.5464408, 2.8992076, 2.5447478, 2.9001539, -0.1288992, 0.1306903

Time for backsubstitution: 8.60 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 416
type: B, layer: 3, pos: 416
type: A, layer: 3, pos: 228
type: B, layer: 3, pos: 228
type: A, layer: 3, pos: 1808
type: B, layer: 3, pos: 1109
type: B, layer: 3, pos: 1808
type: B, layer: 3, pos: 1195
type: A, layer: 3, pos: 659
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 416

## Relational analysis of NS_A1_B2_A2_A2_A1

### Relational analysis result of NS_A1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1071156, upper bound: 0.1087127
time: 3.31 seconds

## Relational analysis of NS_A1_B2_A2_A2_A2

### Relational analysis result of NS_A1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1076996, upper bound: 0.1083496
time: 3.17 seconds

## BFS NS instance: NS_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -14.1046200, -13.3910179, -14.0842171, -13.4044590, -0.2462603, 0.2430940
1: -15.8171911, -15.2665482, -15.8034105, -15.2618656, -0.2126929, 0.2019378
2: -8.7451077, -8.2231884, -8.7457695, -8.2251492, -0.1940599, 0.1984618
3: -7.1935987, -6.6257315, -7.1938410, -6.6312366, -0.1775985, 0.1765477
4: -6.7395654, -6.1878090, -6.7380743, -6.1858320, -0.2434530, 0.2412529
5: -1.6597555, -1.0764794, -1.6612995, -1.0792508, -0.1664639, 0.1729398
6: -15.9428062, -15.3135471, -15.9413576, -15.3249369, -0.2105063, 0.2193276
7: -5.4725075, -4.9616933, -5.4543290, -4.9655113, -0.2395438, 0.2305510
8: -2.3089061, -1.8139458, -2.3013406, -1.8275743, -0.1597720, 0.1604389
9: 2.5462236, 2.9001539, 2.5468166, 2.8983159, -0.1264672, 0.1293262

Time for backsubstitution: 8.74 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 416
type: A, layer: 3, pos: 416
type: B, layer: 3, pos: 228
type: A, layer: 3, pos: 228
type: B, layer: 3, pos: 1808
type: A, layer: 3, pos: 1808
type: B, layer: 3, pos: 1195
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 659
type: A, layer: 3, pos: 659
type: A, layer: 3, pos: 2229
type: B, layer: 3, pos: 2229

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 416

## Relational analysis of NS_A2_A1_B1_B1_B1

### Relational analysis result of NS_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1085097, upper bound: 0.1051122
time: 3.75 seconds

## Relational analysis of NS_A2_A1_B1_B1_B2

### Relational analysis result of NS_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1081467, upper bound: 0.1053208
time: 3.26 seconds

## BFS NS instance: NS_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -14.1108398, -13.3910141, -14.0995684, -13.4020767, -0.2616971, 0.2445283
1: -15.8237753, -15.2665424, -15.8179665, -15.2647181, -0.2364457, 0.1926802
2: -8.7451067, -8.2222672, -8.7455359, -8.2229557, -0.1953565, 0.1984751
3: -7.1935987, -6.6215129, -7.1936679, -6.6204348, -0.1732355, 0.1938071
4: -6.7401476, -6.1878090, -6.7387490, -6.1867127, -0.2448487, 0.2397983
5: -1.6597555, -1.0756607, -1.6612520, -1.0773935, -0.1665614, 0.1757042
6: -15.9428062, -15.3120852, -15.9404049, -15.3218212, -0.2078849, 0.2247218
7: -5.4796658, -4.9616933, -5.4719677, -4.9619083, -0.2607388, 0.2220645
8: -2.3089080, -1.8059278, -2.3088856, -1.8074822, -0.1533144, 0.1833494
9: 2.5456293, 2.9001529, 2.5456691, 2.8984499, -0.1275482, 0.1291901

Time for backsubstitution: 8.65 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 416
type: A, layer: 3, pos: 416
type: B, layer: 3, pos: 228
type: A, layer: 3, pos: 228
type: B, layer: 3, pos: 1808
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1808
type: B, layer: 3, pos: 1195
type: B, layer: 3, pos: 659
type: A, layer: 3, pos: 659
type: A, layer: 3, pos: 2229
type: B, layer: 3, pos: 2229

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 416

## Relational analysis of NS_A2_A1_B1_B2_B1

### Relational analysis result of NS_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1085097, upper bound: 0.1071156
time: 2.94 seconds

## Relational analysis of NS_A2_A1_B1_B2_B2

### Relational analysis result of NS_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1081466, upper bound: 0.1076994
time: 3.03 seconds

## BFS NS instance: NS_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -14.0916433, -13.3934002, -14.1046190, -13.3896132, -0.2304211, 0.2331524
1: -15.8036671, -15.2637215, -15.8171911, -15.2655354, -0.1998026, 0.2066126
2: -8.7453365, -8.2251377, -8.7451086, -8.2226715, -0.1968346, 0.1925023
3: -7.1945143, -6.6349659, -7.1935987, -6.6229353, -0.1757179, 0.1697787
4: -6.7387218, -6.1869268, -6.7395668, -6.1863365, -0.2428825, 0.2418475
5: -1.6599398, -1.0781817, -1.6637826, -1.0764790, -0.1615558, 0.1671191
6: -15.9437637, -15.3169212, -15.9442797, -15.3135481, -0.1976310, 0.1974751
7: -5.4588962, -4.9652972, -5.4725084, -4.9584217, -0.2324190, 0.2322960
8: -2.3013635, -1.8278332, -2.3089056, -1.8136220, -0.1606442, 0.1594249
9: 2.5471113, 2.9000194, 2.5453334, 2.9001541, -0.1248812, 0.1258830

Time for backsubstitution: 8.60 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 416
type: B, layer: 3, pos: 416
type: B, layer: 3, pos: 228
type: A, layer: 3, pos: 228
type: A, layer: 3, pos: 1808
type: B, layer: 3, pos: 1808
type: B, layer: 3, pos: 1195
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 659
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 416

## Relational analysis of NS_A2_A1_B2_A1_A1

### Relational analysis result of NS_A2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1056216, upper bound: 0.1080458
time: 3.35 seconds

## Relational analysis of NS_A2_A1_B2_A1_A2

### Relational analysis result of NS_A2_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1058249, upper bound: 0.1076996
time: 3.72 seconds

## BFS NS instance: NS_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -14.1069946, -13.3910294, -14.1108379, -13.3896112, -0.2261286, 0.2509111
1: -15.8182230, -15.2665815, -15.8237753, -15.2655334, -0.1918766, 0.2302471
2: -8.7451057, -8.2229509, -8.7451057, -8.2217531, -0.1968464, 0.1940022
3: -7.1935987, -6.6248760, -7.1935987, -6.6187162, -0.1938747, 0.1613379
4: -6.7393680, -6.1878090, -6.7401485, -6.1863365, -0.2414193, 0.2432437
5: -1.6597555, -1.0763860, -1.6637826, -1.0756593, -0.1647286, 0.1658556
6: -15.9428062, -15.3138075, -15.9442797, -15.3120832, -0.2029035, 0.1952981
7: -5.4765086, -4.9616933, -5.4796686, -4.9584217, -0.2251703, 0.2534909
8: -2.3089075, -1.8077512, -2.3089080, -1.8056030, -0.1835543, 0.1529635
9: 2.5459754, 2.9001532, 2.5447388, 2.9001541, -0.1243429, 0.1270753

Time for backsubstitution: 8.90 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 416
type: B, layer: 3, pos: 416
type: A, layer: 3, pos: 228
type: B, layer: 3, pos: 228
type: A, layer: 3, pos: 1808
type: B, layer: 3, pos: 1109
type: B, layer: 3, pos: 1808
type: B, layer: 3, pos: 1195
type: A, layer: 3, pos: 659
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 416

## Relational analysis of NS_A2_A1_B2_A2_A1

### Relational analysis result of NS_A2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1075161, upper bound: 0.1080459
time: 3.52 seconds

## Relational analysis of NS_A2_A1_B2_A2_A2

### Relational analysis result of NS_A2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1081466, upper bound: 0.1076997
time: 3.69 seconds

## BFS NS instance: NS_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -14.1064386, -13.3898811, -14.0842171, -13.4041901, -0.2504311, 0.2438586
1: -15.8169346, -15.2674809, -15.8034105, -15.2622452, -0.2176872, 0.2016388
2: -8.7441721, -8.2246227, -8.7457695, -8.2256794, -0.1969904, 0.1978695
3: -7.1977587, -6.6227241, -7.1938410, -6.6304135, -0.1867004, 0.1783333
4: -6.7420969, -6.1858354, -6.7380719, -6.1852403, -0.2460631, 0.2423494
5: -1.6632323, -1.0711489, -1.6625695, -1.0792499, -0.1686700, 0.1871957
6: -15.9441481, -15.3115568, -15.9417419, -15.3249359, -0.2113112, 0.2232551
7: -5.4772739, -4.9580870, -5.4543285, -4.9644151, -0.2468252, 0.2326374
8: -2.3088331, -1.8143711, -2.3013411, -1.8277392, -0.1611769, 0.1603393
9: 2.5461059, 2.9009113, 2.5468259, 2.8983166, -0.1267239, 0.1319192

Time for backsubstitution: 8.72 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 416
type: A, layer: 3, pos: 416
type: B, layer: 3, pos: 228
type: A, layer: 3, pos: 228
type: B, layer: 3, pos: 1808
type: A, layer: 3, pos: 1808
type: B, layer: 3, pos: 1195
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 659
type: A, layer: 3, pos: 659
type: A, layer: 3, pos: 2229
type: B, layer: 3, pos: 2229

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 416

## Relational analysis of NS_A2_A2_B1_B1_B1

### Relational analysis result of NS_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1087127, upper bound: 0.1051120
time: 3.26 seconds

## Relational analysis of NS_A2_A2_B1_B1_B2

### Relational analysis result of NS_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1083496, upper bound: 0.1053208
time: 3.52 seconds

## BFS NS instance: NS_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -14.1126575, -13.3898764, -14.0995674, -13.4018087, -0.2658749, 0.2452924
1: -15.8235168, -15.2674780, -15.8179665, -15.2651043, -0.2414460, 0.1923811
2: -8.7441711, -8.2237015, -8.7455359, -8.2234859, -0.1982880, 0.1978832
3: -7.1977587, -6.6185064, -7.1936679, -6.6196108, -0.1823373, 0.1955925
4: -6.7426786, -6.1858354, -6.7387481, -6.1861219, -0.2474595, 0.2408948
5: -1.6632323, -1.0703311, -1.6625223, -1.0773940, -0.1687673, 0.1899601
6: -15.9441481, -15.3100958, -15.9407883, -15.3218231, -0.2086899, 0.2286495
7: -5.4844337, -4.9580870, -5.4719672, -4.9608097, -0.2680212, 0.2241509
8: -2.3088341, -1.8063540, -2.3088851, -1.8076496, -0.1547238, 0.1832496
9: 2.5455103, 2.9009116, 2.5456793, 2.8984497, -0.1278049, 0.1317840

Time for backsubstitution: 8.65 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 416
type: A, layer: 3, pos: 416
type: B, layer: 3, pos: 228
type: A, layer: 3, pos: 228
type: B, layer: 3, pos: 1808
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1808
type: B, layer: 3, pos: 1195
type: B, layer: 3, pos: 659
type: A, layer: 3, pos: 659
type: A, layer: 3, pos: 2229
type: B, layer: 3, pos: 2229

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 416

## Relational analysis of NS_A2_A2_B1_B2_B1

### Relational analysis result of NS_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1087127, upper bound: 0.1071155
time: 2.97 seconds

## Relational analysis of NS_A2_A2_B1_B2_B2

### Relational analysis result of NS_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1083497, upper bound: 0.1076995
time: 3.24 seconds

## BFS NS instance: NS_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -14.0934620, -13.3922653, -14.1046200, -13.3893471, -0.2345998, 0.2339167
1: -15.8034077, -15.2646551, -15.8171911, -15.2659216, -0.2048039, 0.2063137
2: -8.7444019, -8.2265720, -8.7451086, -8.2232027, -0.1997662, 0.1919103
3: -7.1986742, -6.6319613, -7.1935987, -6.6221137, -0.1848198, 0.1715643
4: -6.7412519, -6.1849546, -6.7395654, -6.1857443, -0.2454935, 0.2429430
5: -1.6634197, -1.0728521, -1.6650543, -1.0764778, -0.1637617, 0.1813738
6: -15.9451036, -15.3149300, -15.9446602, -15.3135481, -0.1984360, 0.2014009
7: -5.4636650, -4.9616899, -5.4725065, -4.9573245, -0.2396849, 0.2343814
8: -2.3012877, -1.8282614, -2.3089080, -1.8137908, -0.1620541, 0.1593254
9: 2.5469940, 2.9007776, 2.5453444, 2.9001541, -0.1251382, 0.1284769

Time for backsubstitution: 8.72 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 416
type: B, layer: 3, pos: 416
type: A, layer: 3, pos: 228
type: B, layer: 3, pos: 228
type: A, layer: 3, pos: 1808
type: B, layer: 3, pos: 1808
type: B, layer: 3, pos: 1195
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 659
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 416

## Relational analysis of NS_A2_A2_B2_A1_A1

### Relational analysis result of NS_A2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1058247, upper bound: 0.1080459
time: 3.36 seconds

## Relational analysis of NS_A2_A2_B2_A1_A2

### Relational analysis result of NS_A2_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1060276, upper bound: 0.1076994
time: 4.66 seconds

## BFS NS instance: NS_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -14.1088133, -13.3898926, -14.1108379, -13.3893471, -0.2303072, 0.2516756
1: -15.8179646, -15.2675142, -15.8237753, -15.2659178, -0.1968781, 0.2299482
2: -8.7441711, -8.2243824, -8.7451077, -8.2222805, -0.1997784, 0.1934099
3: -7.1977587, -6.6218719, -7.1935987, -6.6178951, -0.2029763, 0.1631235
4: -6.7418985, -6.1858354, -6.7401471, -6.1857443, -0.2440304, 0.2443392
5: -1.6632323, -1.0710564, -1.6650543, -1.0756621, -0.1669346, 0.1801114
6: -15.9441481, -15.3118191, -15.9446602, -15.3120842, -0.2037085, 0.1992254
7: -5.4812779, -4.9580870, -5.4796658, -4.9573245, -0.2324493, 0.2555764
8: -2.3088317, -1.8081779, -2.3089066, -1.8057714, -0.1849650, 0.1528639
9: 2.5458577, 2.9009109, 2.5447478, 2.9001539, -0.1245997, 0.1296689

Time for backsubstitution: 8.64 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 416
type: B, layer: 3, pos: 416
type: A, layer: 3, pos: 228
type: B, layer: 3, pos: 228
type: A, layer: 3, pos: 1808
type: B, layer: 3, pos: 1109
type: B, layer: 3, pos: 1808
type: B, layer: 3, pos: 1195
type: A, layer: 3, pos: 659
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 416

## Relational analysis of NS_A2_A2_B2_A2_A1

### Relational analysis result of NS_A2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1077191, upper bound: 0.1080459
time: 3.35 seconds

## Relational analysis of NS_A2_A2_B2_A2_A2

### Relational analysis result of NS_A2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1083497, upper bound: 0.1076996
time: 3.34 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 15.48 seconds
NS_A1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 15.48
Output dim: 9, lower bound: -0.1051631, upper bound: 0.1082878
NS_A1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 15.48
Output dim: 9, lower bound: -0.1053710, upper bound: 0.1079255
NS_A1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 15.48
Output dim: 9, lower bound: -0.1070844, upper bound: 0.1082878
NS_A1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 15.48
Output dim: 9, lower bound: -0.1077225, upper bound: 0.1079256
NS_A1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 15.48
Output dim: 9, lower bound: -0.1053661, upper bound: 0.1082877
NS_A1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 15.48
Output dim: 9, lower bound: -0.1055736, upper bound: 0.1079255
NS_A1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 15.48
Output dim: 9, lower bound: -0.1072874, upper bound: 0.1082878
NS_A1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 15.48
Output dim: 9, lower bound: -0.1079255, upper bound: 0.1079256
NS_A1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 15.48
Output dim: 9, lower bound: -0.1049091, upper bound: 0.1087126
NS_A1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 15.48
Output dim: 9, lower bound: -0.1051182, upper bound: 0.1083496
NS_A1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 15.48
Output dim: 9, lower bound: -0.1069126, upper bound: 0.1087127
NS_A1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 15.48
Output dim: 9, lower bound: -0.1074966, upper bound: 0.1083497
NS_A1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 15.48
Output dim: 9, lower bound: -0.1051121, upper bound: 0.1087127
NS_A1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 15.48
Output dim: 9, lower bound: -0.1053209, upper bound: 0.1083496
NS_A1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 15.48
Output dim: 9, lower bound: -0.1071156, upper bound: 0.1087127
NS_A1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 15.48
Output dim: 9, lower bound: -0.1076996, upper bound: 0.1083496
NS_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 15.48
Output dim: 9, lower bound: -0.1085097, upper bound: 0.1051122
NS_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 15.48
Output dim: 9, lower bound: -0.1081467, upper bound: 0.1053208
NS_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 15.48
Output dim: 9, lower bound: -0.1085097, upper bound: 0.1071156
NS_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 15.48
Output dim: 9, lower bound: -0.1081466, upper bound: 0.1076994
NS_A2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 15.48
Output dim: 9, lower bound: -0.1056216, upper bound: 0.1080458
NS_A2_A1_B2_A1_A2, status: Status.VERIFIED, split count: 5, time: 15.48
Output dim: 9, lower bound: -0.1058249, upper bound: 0.1076996
NS_A2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 15.48
Output dim: 9, lower bound: -0.1075161, upper bound: 0.1080459
NS_A2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 15.48
Output dim: 9, lower bound: -0.1081466, upper bound: 0.1076997
NS_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 15.48
Output dim: 9, lower bound: -0.1087127, upper bound: 0.1051120
NS_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 15.48
Output dim: 9, lower bound: -0.1083496, upper bound: 0.1053208
NS_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 15.48
Output dim: 9, lower bound: -0.1087127, upper bound: 0.1071155
NS_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 15.48
Output dim: 9, lower bound: -0.1083497, upper bound: 0.1076995
NS_A2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 15.48
Output dim: 9, lower bound: -0.1058247, upper bound: 0.1080459
NS_A2_A2_B2_A1_A2, status: Status.VERIFIED, split count: 5, time: 15.48
Output dim: 9, lower bound: -0.1060276, upper bound: 0.1076994
NS_A2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 15.48
Output dim: 9, lower bound: -0.1077191, upper bound: 0.1080459
NS_A2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 15.48
Output dim: 9, lower bound: -0.1083497, upper bound: 0.1076996

## BFS NS instance: NS_A1_B1_A1_A1_A1

### Backsubstitution after applying NS history:
0: -14.0825796, -13.4058666, -14.0971928, -13.4020634, -0.2234550, 0.2281458
1: -15.8021517, -15.2628803, -15.8169365, -15.2646875, -0.1988657, 0.2067306
2: -8.7457695, -8.2261448, -8.7455368, -8.2231951, -0.1967190, 0.1918935
3: -7.1938410, -6.6388535, -7.1936679, -6.6219988, -0.1757548, 0.1641016
4: -6.7355237, -6.1873045, -6.7389445, -6.1867127, -0.2396448, 0.2415884
5: -1.6562476, -1.0792534, -1.6612520, -1.0774856, -0.1588144, 0.1661634
6: -15.9385281, -15.3249435, -15.9404049, -15.3215618, -0.1928219, 0.1944879
7: -5.4519320, -4.9687834, -5.4679637, -4.9619083, -0.2282798, 0.2301719
8: -2.3012857, -1.8279018, -2.3088851, -1.8136778, -0.1605539, 0.1593757
9: 2.5477223, 2.8977141, 2.5459177, 2.8984489, -0.1242179, 0.1241212

Time for backsubstitution: 8.70 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 228
type: B, layer: 3, pos: 228
type: B, layer: 3, pos: 416
type: A, layer: 3, pos: 1808
type: B, layer: 3, pos: 1808
type: B, layer: 3, pos: 1195
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 659
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 228

## Relational analysis of NS_A1_B1_A1_A1_A1_A1

### Relational analysis result of NS_A1_B1_A1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1043834, upper bound: 0.1077947
time: 4.32 seconds

## Relational analysis of NS_A1_B1_A1_A1_A1_A2

### Relational analysis result of NS_A1_B1_A1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1046932, upper bound: 0.1077949
time: 3.53 seconds

## BFS NS instance: NS_A1_B1_A1_A1_A2

### Backsubstitution after applying NS history:
0: -14.0778179, -13.4016533, -14.0951824, -13.4020691, -0.2232966, 0.2327586
1: -15.7992535, -15.2591686, -15.8156652, -15.2646875, -0.1989994, 0.2087716
2: -8.7471218, -8.2271004, -8.7455378, -8.2236500, -0.1980370, 0.1919823
3: -7.2096138, -6.6434021, -7.1936679, -6.6247749, -0.2005820, 0.1668583
4: -6.7302895, -6.1795487, -6.7361403, -6.1867127, -0.2400215, 0.2530832
5: -1.6553307, -1.0754228, -1.6603217, -1.0774879, -0.1595813, 0.1711562
6: -15.9339266, -15.3215904, -15.9385338, -15.3215637, -0.1923908, 0.2003285
7: -5.4471736, -4.9613953, -5.4656324, -4.9619083, -0.2289286, 0.2376366
8: -2.3010416, -1.8277802, -2.3087778, -1.8136826, -0.1605189, 0.1593764
9: 2.5464444, 2.8949773, 2.5459299, 2.8974004, -0.1279437, 0.1237395

Time for backsubstitution: 8.60 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 228
type: B, layer: 3, pos: 228
type: A, layer: 3, pos: 1808
type: B, layer: 3, pos: 1808
type: B, layer: 3, pos: 1195
type: B, layer: 3, pos: 416
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 659
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 228

## Relational analysis of NS_A1_B1_A1_A1_A2_A1

### Relational analysis result of NS_A1_B1_A1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1044802, upper bound: 0.1074474
time: 3.38 seconds

## Relational analysis of NS_A1_B1_A1_A1_A2_A2

### Relational analysis result of NS_A1_B1_A1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1049011, upper bound: 0.1074475
time: 3.81 seconds

## BFS NS instance: NS_A1_B1_A1_A2_A1

### Backsubstitution after applying NS history:
0: -14.0979290, -13.4034863, -14.1034098, -13.4020624, -0.2191522, 0.2459233
1: -15.8165417, -15.2657347, -15.8235226, -15.2646818, -0.1909262, 0.2303909
2: -8.7455359, -8.2239513, -8.7455378, -8.2222767, -0.1967324, 0.1933925
3: -7.1936679, -6.6286116, -7.1936679, -6.6172504, -0.1939111, 0.1560734
4: -6.7361913, -6.1881866, -6.7395301, -6.1867127, -0.2382542, 0.2429845
5: -1.6561997, -1.0773959, -1.6612520, -1.0766702, -0.1628610, 0.1649590
6: -15.9375019, -15.3218260, -15.9404049, -15.3201008, -0.1983048, 0.1923152
7: -5.4694867, -4.9651780, -5.4751248, -4.9619083, -0.2210699, 0.2513666
8: -2.3088388, -1.8078127, -2.3088851, -1.8056612, -0.1834681, 0.1529214
9: 2.5465746, 2.8979616, 2.5453238, 2.8984492, -0.1236957, 0.1256458

Time for backsubstitution: 8.64 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 228
type: B, layer: 3, pos: 228
type: B, layer: 3, pos: 416
type: A, layer: 3, pos: 1808
type: B, layer: 3, pos: 1109
type: B, layer: 3, pos: 1808
type: B, layer: 3, pos: 1195
type: A, layer: 3, pos: 659
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 228

## Relational analysis of NS_A1_B1_A1_A2_A1_A1

### Relational analysis result of NS_A1_B1_A1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1065702, upper bound: 0.1077949
time: 3.53 seconds

## Relational analysis of NS_A1_B1_A1_A2_A1_A2

### Relational analysis result of NS_A1_B1_A1_A2_A1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1065335, upper bound: 0.1077948
time: 3.31 seconds

## BFS NS instance: NS_A1_B1_A1_A2_A2

### Backsubstitution after applying NS history:
0: -14.0931702, -13.3992729, -14.1013985, -13.4020672, -0.2189939, 0.2510986
1: -15.8144035, -15.2614727, -15.8224001, -15.2646856, -0.1912431, 0.2333575
2: -8.7468872, -8.2249050, -8.7455397, -8.2227297, -0.1979355, 0.1934811
3: -7.2112093, -6.6314516, -7.1936679, -6.6198349, -0.2159724, 0.1598711
4: -6.7307868, -6.1803284, -6.7366848, -6.1867127, -0.2385222, 0.2552722
5: -1.6549261, -1.0739241, -1.6603217, -1.0766692, -0.1636280, 0.1707861
6: -15.9329844, -15.3183413, -15.9385338, -15.3201008, -0.1978741, 0.1983690
7: -5.4647532, -4.9576893, -5.4727764, -4.9619083, -0.2217227, 0.2587175
8: -2.3085461, -1.8077536, -2.3087778, -1.8056650, -0.1834319, 0.1529342
9: 2.5454450, 2.8951123, 2.5453353, 2.8974009, -0.1278878, 0.1252663

Time for backsubstitution: 8.74 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 228
type: B, layer: 3, pos: 228
type: A, layer: 3, pos: 1808
type: B, layer: 3, pos: 1109
type: B, layer: 3, pos: 1808
type: B, layer: 3, pos: 1195
type: B, layer: 3, pos: 416
type: A, layer: 3, pos: 659
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 228

## Relational analysis of NS_A1_B1_A1_A2_A2_A1

### Relational analysis result of NS_A1_B1_A1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1068122, upper bound: 0.1074475
time: 3.29 seconds

## Relational analysis of NS_A1_B1_A1_A2_A2_A2

### Relational analysis result of NS_A1_B1_A1_A2_A2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1072444, upper bound: 0.1074475
time: 3.16 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A1

### Backsubstitution after applying NS history:
0: -14.0843973, -13.4047308, -14.0971928, -13.4017963, -0.2276334, 0.2289101
1: -15.8018894, -15.2638121, -15.8169365, -15.2650681, -0.2038672, 0.2064315
2: -8.7448349, -8.2275743, -8.7455378, -8.2237263, -0.1996511, 0.1913013
3: -7.1979990, -6.6358476, -7.1936679, -6.6211762, -0.1848568, 0.1658874
4: -6.7380543, -6.1853323, -6.7389450, -6.1861219, -0.2422558, 0.2426848
5: -1.6597276, -1.0739222, -1.6625223, -1.0774865, -0.1610203, 0.1804186
6: -15.9398708, -15.3229523, -15.9407883, -15.3215599, -0.1936270, 0.1984140
7: -5.4567003, -4.9651761, -5.4679642, -4.9608097, -0.2355462, 0.2322578
8: -2.3012094, -1.8283277, -2.3088851, -1.8138447, -0.1619644, 0.1592764
9: 2.5476046, 2.8984718, 2.5459273, 2.8984492, -0.1244746, 0.1267151

Time for backsubstitution: 8.60 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 228
type: A, layer: 3, pos: 228
type: B, layer: 3, pos: 416
type: A, layer: 3, pos: 1808
type: B, layer: 3, pos: 1808
type: B, layer: 3, pos: 1195
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 659
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 228

## Relational analysis of NS_A1_B1_A2_A1_A1_B1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1048962, upper bound: 0.1073725
time: 3.69 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B2

### Relational analysis result of NS_A1_B1_A2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1048962, upper bound: 0.1077948
time: 3.92 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.76 + 549.02 = 605.78 seconds
