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
execution time: IAR + RelationalAnalysis = 21.56 + 33.53 = 55.09 seconds
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
type: A, layer: 3, pos: 1195
type: A, layer: 3, pos: 416
type: A, layer: 3, pos: 228
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1808
type: A, layer: 3, pos: 659
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1095381, upper bound: 0.1101964
time: 2.90 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1101965, upper bound: 0.1101964
time: 3.25 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 6.40 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 6.40
Output dim: 9, lower bound: -0.1095381, upper bound: 0.1101964
NS_A2, status: Status.UNKNOWN, split count: 1, time: 6.40
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

Time for backsubstitution: 7.29 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 1195
type: B, layer: 3, pos: 416
type: B, layer: 3, pos: 228
type: B, layer: 3, pos: 1109
type: B, layer: 3, pos: 1808
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 1836

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1095381, upper bound: 0.1095380
time: 4.28 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1095381, upper bound: 0.1101965
time: 3.04 seconds

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

Time for backsubstitution: 7.63 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 1195
type: B, layer: 3, pos: 416
type: B, layer: 3, pos: 228
type: B, layer: 3, pos: 1109
type: B, layer: 3, pos: 1808
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 1836

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1101965, upper bound: 0.1095379
time: 3.27 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1101965, upper bound: 0.1101964
time: 3.35 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 14.55 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 14.55
Output dim: 9, lower bound: -0.1095381, upper bound: 0.1095380
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 14.55
Output dim: 9, lower bound: -0.1095381, upper bound: 0.1101965
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 14.55
Output dim: 9, lower bound: -0.1101965, upper bound: 0.1095379
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 14.55
Output dim: 9, lower bound: -0.1101965, upper bound: 0.1101964

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

Time for backsubstitution: 7.63 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1195
type: A, layer: 3, pos: 416
type: A, layer: 3, pos: 228
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1808
type: A, layer: 3, pos: 659
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 1195

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1087984, upper bound: 0.1092353
time: 3.15 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1090014, upper bound: 0.1092353
time: 3.09 seconds

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

Time for backsubstitution: 7.41 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1195
type: A, layer: 3, pos: 416
type: A, layer: 3, pos: 228
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1808
type: A, layer: 3, pos: 659
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1195

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1087984, upper bound: 0.1096598
time: 3.06 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1090014, upper bound: 0.1096598
time: 3.32 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -14.1119051, -13.3889809, -14.1044760, -13.4014311, -0.2633085, 0.2666469
1: -15.8253345, -15.2650700, -15.8250790, -15.2642174, -0.2375338, 0.2357388
2: -8.7451077, -8.2213078, -8.7455387, -8.2218332, -0.1981854, 0.2005808
3: -7.1935987, -6.6163902, -7.1936679, -6.6151047, -0.1998816, 0.1997706
4: -6.7403646, -6.1856775, -6.7397442, -6.1860542, -0.2457001, 0.2456651
5: -1.6660295, -1.0754595, -1.6634974, -1.0764687, -0.1803270, 0.1787900
6: -15.9449377, -15.3116083, -15.9410648, -15.3196239, -0.2207506, 0.2256439
7: -5.4805431, -4.9569149, -5.4759989, -4.9604001, -0.2627286, 0.2616777
8: -2.3089042, -1.8049545, -2.3088870, -1.8050098, -0.1838824, 0.1839429
9: 2.5442443, 2.9001536, 2.5448291, 2.8984504, -0.1285340, 0.1313666

Time for backsubstitution: 7.99 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1195
type: A, layer: 3, pos: 416
type: A, layer: 3, pos: 228
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1808
type: A, layer: 3, pos: 659
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 1195

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1094568, upper bound: 0.1090014
time: 3.24 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1096598, upper bound: 0.1090013
time: 3.02 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -14.1119051, -13.3889809, -14.1119051, -13.3889809, -0.2529520, 0.2529520
1: -15.8253345, -15.2650700, -15.8253345, -15.2650700, -0.2324684, 0.2324684
2: -8.7451077, -8.2213078, -8.7451077, -8.2213078, -0.1977867, 0.1977867
3: -7.1935987, -6.6163902, -7.1935987, -6.6163902, -0.1958444, 0.1958444
4: -6.7403646, -6.1856775, -6.7403646, -6.1856775, -0.2456784, 0.2456784
5: -1.6660295, -1.0754595, -1.6660295, -1.0754595, -0.1743458, 0.1743458
6: -15.9449377, -15.3116083, -15.9449377, -15.3116083, -0.2057869, 0.2057870
7: -5.4805431, -4.9569149, -5.4805431, -4.9569149, -0.2595192, 0.2595193
8: -2.3089042, -1.8049545, -2.3089042, -1.8049545, -0.1838404, 0.1838404
9: 2.5442443, 2.9001536, 2.5442443, 2.9001536, -0.1274617, 0.1274617

Time for backsubstitution: 7.95 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1195
type: A, layer: 3, pos: 416
type: A, layer: 3, pos: 228
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1808
type: A, layer: 3, pos: 659
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 1195

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1094568, upper bound: 0.1090013
time: 2.99 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1096598, upper bound: 0.1090013
time: 2.96 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 14.13 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 14.13
Output dim: 9, lower bound: -0.1087984, upper bound: 0.1092353
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 14.13
Output dim: 9, lower bound: -0.1090014, upper bound: 0.1092353
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 14.13
Output dim: 9, lower bound: -0.1087984, upper bound: 0.1096598
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 14.13
Output dim: 9, lower bound: -0.1090014, upper bound: 0.1096598
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 14.13
Output dim: 9, lower bound: -0.1094568, upper bound: 0.1090014
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 14.13
Output dim: 9, lower bound: -0.1096598, upper bound: 0.1090013
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 14.13
Output dim: 9, lower bound: -0.1094568, upper bound: 0.1090013
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 14.13
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

Time for backsubstitution: 8.03 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 416
type: B, layer: 3, pos: 228
type: B, layer: 3, pos: 1109
type: B, layer: 3, pos: 1195
type: B, layer: 3, pos: 1808
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 416

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1083374, upper bound: 0.1075663
time: 3.09 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1079917, upper bound: 0.1081947
time: 2.94 seconds

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

Time for backsubstitution: 7.42 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 416
type: B, layer: 3, pos: 228
type: B, layer: 3, pos: 1109
type: B, layer: 3, pos: 1195
type: B, layer: 3, pos: 1808
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 416

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1085404, upper bound: 0.1075663
time: 3.25 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1081947, upper bound: 0.1081945
time: 2.79 seconds

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

Time for backsubstitution: 7.41 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 416
type: B, layer: 3, pos: 228
type: B, layer: 3, pos: 1109
type: B, layer: 3, pos: 1195
type: B, layer: 3, pos: 1808
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 416

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1080955, upper bound: 0.1079881
time: 3.21 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1077657, upper bound: 0.1086188
time: 3.00 seconds

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

Time for backsubstitution: 7.52 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 416
type: B, layer: 3, pos: 228
type: B, layer: 3, pos: 1109
type: B, layer: 3, pos: 1195
type: B, layer: 3, pos: 1808
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 416

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1082984, upper bound: 0.1079880
time: 3.10 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1079687, upper bound: 0.1086187
time: 3.09 seconds

## BFS NS instance: NS_A2_B1_A1

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

Time for backsubstitution: 7.48 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 416
type: B, layer: 3, pos: 228
type: B, layer: 3, pos: 1109
type: B, layer: 3, pos: 1195
type: B, layer: 3, pos: 1808
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 416

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1087623, upper bound: 0.1073913
time: 3.05 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1084158, upper bound: 0.1079686
time: 2.91 seconds

## BFS NS instance: NS_A2_B1_A2

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

Time for backsubstitution: 7.47 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 416
type: B, layer: 3, pos: 228
type: B, layer: 3, pos: 1109
type: B, layer: 3, pos: 1195
type: B, layer: 3, pos: 1808
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 416

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1089653, upper bound: 0.1073912
time: 3.04 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1086188, upper bound: 0.1079686
time: 3.17 seconds

## BFS NS instance: NS_A2_B2_A1

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

Time for backsubstitution: 7.97 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 416
type: B, layer: 3, pos: 228
type: B, layer: 3, pos: 1109
type: B, layer: 3, pos: 1195
type: B, layer: 3, pos: 1808
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 416

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1087623, upper bound: 0.1073913
time: 3.26 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1084158, upper bound: 0.1079687
time: 3.15 seconds

## BFS NS instance: NS_A2_B2_A2

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

Time for backsubstitution: 7.65 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 416
type: B, layer: 3, pos: 228
type: B, layer: 3, pos: 1109
type: B, layer: 3, pos: 1195
type: B, layer: 3, pos: 1808
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 416

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1089653, upper bound: 0.1073913
time: 3.78 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1086188, upper bound: 0.1079686
time: 3.01 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 14.67 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.67
Output dim: 9, lower bound: -0.1083374, upper bound: 0.1075663
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.67
Output dim: 9, lower bound: -0.1079917, upper bound: 0.1081947
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.67
Output dim: 9, lower bound: -0.1085404, upper bound: 0.1075663
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.67
Output dim: 9, lower bound: -0.1081947, upper bound: 0.1081945
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.67
Output dim: 9, lower bound: -0.1080955, upper bound: 0.1079881
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.67
Output dim: 9, lower bound: -0.1077657, upper bound: 0.1086188
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.67
Output dim: 9, lower bound: -0.1082984, upper bound: 0.1079880
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.67
Output dim: 9, lower bound: -0.1079687, upper bound: 0.1086187
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.67
Output dim: 9, lower bound: -0.1087623, upper bound: 0.1073913
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.67
Output dim: 9, lower bound: -0.1084158, upper bound: 0.1079686
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.67
Output dim: 9, lower bound: -0.1089653, upper bound: 0.1073912
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.67
Output dim: 9, lower bound: -0.1086188, upper bound: 0.1079686
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.67
Output dim: 9, lower bound: -0.1087623, upper bound: 0.1073913
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.67
Output dim: 9, lower bound: -0.1084158, upper bound: 0.1079687
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.67
Output dim: 9, lower bound: -0.1089653, upper bound: 0.1073913
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.67
Output dim: 9, lower bound: -0.1086188, upper bound: 0.1079686

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -14.1044769, -13.4034624, -14.1028366, -13.4020634, -0.2471275, 0.2437780
1: -15.8250790, -15.2656851, -15.8237247, -15.2646770, -0.2316737, 0.2291907
2: -8.7455387, -8.2225819, -8.7455368, -8.2225428, -0.1964160, 0.1957368
3: -7.1936679, -6.6191616, -7.1936679, -6.6216021, -0.1888399, 0.1902300
4: -6.7397451, -6.1881866, -6.7371898, -6.1867127, -0.2447083, 0.2399807
5: -1.6572220, -1.0764701, -1.6602297, -1.0764718, -0.1637843, 0.1695074
6: -15.9389334, -15.3196220, -15.9389725, -15.3196259, -0.2000111, 0.2003251
7: -5.4759998, -4.9651780, -5.4735556, -4.9619083, -0.2555850, 0.2487919
8: -2.3088851, -1.8054771, -2.3088384, -1.8051600, -0.1835934, 0.1832478
9: 2.5461173, 2.8984494, 2.5452425, 2.8979614, -0.1251243, 0.1264417

Time for backsubstitution: 7.51 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 416
type: A, layer: 3, pos: 228
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1808
type: A, layer: 3, pos: 659
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 416

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1073633, upper bound: 0.1075663
time: 2.93 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1073633, upper bound: 0.1075663
time: 2.76 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -14.1024656, -13.4034672, -14.0980787, -13.3978519, -0.2523032, 0.2436199
1: -15.8239613, -15.2656860, -15.8215189, -15.2604160, -0.2346416, 0.2293249
2: -8.7455368, -8.2230320, -8.7468910, -8.2234974, -0.1965059, 0.1969318
3: -7.1936679, -6.6217499, -7.2112093, -6.6245909, -0.1915970, 0.2124963
4: -6.7369003, -6.1881866, -6.7318530, -6.1788545, -0.2570139, 0.2403560
5: -1.6562939, -1.0764704, -1.6589546, -1.0728703, -0.1673142, 0.1702744
6: -15.9370642, -15.3196211, -15.9344540, -15.3161869, -0.2047381, 0.1998945
7: -5.4736505, -4.9651780, -5.4687839, -4.9544182, -0.2629280, 0.2494607
8: -2.3087778, -1.8054814, -2.3085446, -1.8051028, -0.1835445, 0.1832118
9: 2.5461287, 2.8974009, 2.5440402, 2.8951125, -0.1247452, 0.1296189

Time for backsubstitution: 7.50 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 228
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 416
type: A, layer: 3, pos: 1808
type: A, layer: 3, pos: 659
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 228

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1070829, upper bound: 0.1077166
time: 3.13 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1075136, upper bound: 0.1077165
time: 3.06 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -14.1062946, -13.4023247, -14.1028366, -13.4017992, -0.2513072, 0.2445424
1: -15.8248234, -15.2666159, -15.8237247, -15.2650614, -0.2366765, 0.2288920
2: -8.7446022, -8.2240133, -8.7455368, -8.2230711, -0.1993481, 0.1951448
3: -7.1978269, -6.6161561, -7.1936679, -6.6207800, -0.1979418, 0.1920154
4: -6.7422748, -6.1862125, -6.7371869, -6.1861219, -0.2473193, 0.2410762
5: -1.6607018, -1.0711398, -1.6615009, -1.0764713, -0.1659902, 0.1837631
6: -15.9402733, -15.3176336, -15.9393520, -15.3196259, -0.2008161, 0.2042524
7: -5.4807663, -4.9615726, -5.4735532, -4.9608097, -0.2628680, 0.2508774
8: -2.3088112, -1.8059034, -2.3088379, -1.8053293, -0.1850048, 0.1831481
9: 2.5459991, 2.8992074, 2.5452509, 2.8979597, -0.1253813, 0.1290357

Time for backsubstitution: 7.48 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 416
type: A, layer: 3, pos: 228
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1808
type: A, layer: 3, pos: 659
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 416

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1075663, upper bound: 0.1075663
time: 3.23 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1075663, upper bound: 0.1075663
time: 3.20 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -14.1042833, -13.4023323, -14.0980806, -13.3975849, -0.2564837, 0.2443841
1: -15.8237028, -15.2666197, -15.8215189, -15.2607994, -0.2396448, 0.2290261
2: -8.7446022, -8.2244663, -8.7468891, -8.2240276, -0.1994380, 0.1963392
3: -7.1978269, -6.6187429, -7.2112093, -6.6237707, -0.2006987, 0.2142819
4: -6.7394314, -6.1862125, -6.7318506, -6.1782627, -0.2596245, 0.2414517
5: -1.6597729, -1.0711393, -1.6602247, -1.0728719, -0.1695200, 0.1845302
6: -15.9384041, -15.3176365, -15.9348354, -15.3161888, -0.2055430, 0.2038218
7: -5.4784188, -4.9615726, -5.4687824, -4.9533195, -0.2702117, 0.2515459
8: -2.3087039, -1.8059072, -2.3085451, -1.8052707, -0.1849563, 0.1831120
9: 2.5460095, 2.8981586, 2.5440502, 2.8951113, -0.1250020, 0.1322129

Time for backsubstitution: 7.44 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 228
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 416
type: A, layer: 3, pos: 1808
type: A, layer: 3, pos: 659
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 228

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1072859, upper bound: 0.1077165
time: 3.12 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1077166, upper bound: 0.1077165
time: 3.06 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -14.1044769, -13.4034624, -14.1102619, -13.3896151, -0.2657979, 0.2590187
1: -15.8250790, -15.2656851, -15.8239765, -15.2655277, -0.2347844, 0.2341073
2: -8.7455387, -8.2225819, -8.7451057, -8.2220192, -0.1993244, 0.1962497
3: -7.1936679, -6.6191616, -7.1935987, -6.6225376, -0.1920236, 0.1942309
4: -6.7397451, -6.1881866, -6.7378073, -6.1863365, -0.2449545, 0.2402599
5: -1.6572220, -1.0764701, -1.6627603, -1.0754628, -0.1691803, 0.1764920
6: -15.9389334, -15.3196220, -15.9428482, -15.3116102, -0.2228442, 0.2182651
7: -5.4759998, -4.9651780, -5.4780965, -4.9584217, -0.2598681, 0.2541261
8: -2.3088851, -1.8054771, -2.3088598, -1.8051052, -0.1837308, 0.1833248
9: 2.5461173, 2.8984494, 2.5446577, 2.8996663, -0.1296927, 0.1281446

Time for backsubstitution: 8.14 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 416
type: A, layer: 3, pos: 228
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1808
type: A, layer: 3, pos: 659
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 416

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1071883, upper bound: 0.1079881
time: 3.11 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1071883, upper bound: 0.1079881
time: 3.05 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -14.1024656, -13.4034672, -14.1055107, -13.3854008, -0.2721732, 0.2588603
1: -15.8239613, -15.2656860, -15.8217697, -15.2612677, -0.2376971, 0.2342417
2: -8.7455368, -8.2230320, -8.7464600, -8.2229118, -0.1994143, 0.1972909
3: -7.1936679, -6.6217499, -7.2110615, -6.6262398, -0.1947811, 0.2172590
4: -6.7369003, -6.1881866, -6.7323050, -6.1786280, -0.2570205, 0.2406359
5: -1.6562939, -1.0764704, -1.6616089, -1.0715861, -0.1721451, 0.1772591
6: -15.9370642, -15.3196211, -15.9383278, -15.3081741, -0.2274413, 0.2178345
7: -5.4736505, -4.9651780, -5.4733286, -4.9509315, -0.2666287, 0.2547960
8: -2.3087778, -1.8054814, -2.3085666, -1.8050461, -0.1836817, 0.1832888
9: 2.5461287, 2.8974009, 2.5434556, 2.8968172, -0.1293136, 0.1310401

Time for backsubstitution: 7.49 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 228
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 416
type: A, layer: 3, pos: 1808
type: A, layer: 3, pos: 659
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 228

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1068570, upper bound: 0.1081407
time: 3.25 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1072876, upper bound: 0.1081408
time: 3.31 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -14.1062946, -13.4023247, -14.1102629, -13.3893490, -0.2699776, 0.2597830
1: -15.8248234, -15.2666159, -15.8239765, -15.2659111, -0.2397869, 0.2338085
2: -8.7446022, -8.2240133, -8.7451067, -8.2225485, -0.2022567, 0.1956573
3: -7.1978269, -6.6161561, -7.1935987, -6.6217165, -0.2011253, 0.1960162
4: -6.7422748, -6.1862125, -6.7378073, -6.1857443, -0.2475654, 0.2413566
5: -1.6607018, -1.0711398, -1.6640286, -1.0754633, -0.1713862, 0.1907476
6: -15.9402733, -15.3176336, -15.9432259, -15.3116093, -0.2236493, 0.2221925
7: -5.4807663, -4.9615726, -5.4780955, -4.9573245, -0.2671511, 0.2562118
8: -2.3088112, -1.8059034, -2.3088598, -1.8052726, -0.1851420, 0.1832254
9: 2.5459991, 2.8992074, 2.5446675, 2.8996654, -0.1299496, 0.1307389

Time for backsubstitution: 7.52 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 416
type: A, layer: 3, pos: 228
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1808
type: A, layer: 3, pos: 659
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 416

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1073914, upper bound: 0.1079882
time: 3.87 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1073914, upper bound: 0.1079881
time: 3.33 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -14.1042833, -13.4023323, -14.1055088, -13.3851347, -0.2763534, 0.2596247
1: -15.8237028, -15.2666197, -15.8217697, -15.2616501, -0.2427003, 0.2339430
2: -8.7446022, -8.2244663, -8.7464600, -8.2234421, -0.2023463, 0.1966988
3: -7.1978269, -6.6187429, -7.2110615, -6.6254148, -0.2038827, 0.2190447
4: -6.7394314, -6.1862125, -6.7323041, -6.1780348, -0.2596307, 0.2417316
5: -1.6597729, -1.0711393, -1.6628785, -1.0715871, -0.1743511, 0.1915147
6: -15.9384041, -15.3176365, -15.9387102, -15.3081760, -0.2282460, 0.2217619
7: -5.4784188, -4.9615726, -5.4733267, -4.9498343, -0.2739124, 0.2568808
8: -2.3087039, -1.8059072, -2.3085666, -1.8052139, -0.1850934, 0.1831892
9: 2.5460095, 2.8981586, 2.5434663, 2.8968155, -0.1295704, 0.1336341

Time for backsubstitution: 8.08 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 228
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 416
type: A, layer: 3, pos: 1808
type: A, layer: 3, pos: 659
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 228

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1070600, upper bound: 0.1081406
time: 3.13 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1074907, upper bound: 0.1081406
time: 3.13 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -14.1119022, -13.3910112, -14.1028366, -13.4020634, -0.2624600, 0.2624487
1: -15.8253345, -15.2665348, -15.8237247, -15.2646770, -0.2365789, 0.2323009
2: -8.7451086, -8.2220554, -8.7455368, -8.2225428, -0.1969289, 0.1986452
3: -7.1935987, -6.6204462, -7.1936679, -6.6216021, -0.1928407, 0.1941196
4: -6.7403626, -6.1878090, -6.7371898, -6.1867127, -0.2449887, 0.2402267
5: -1.6597555, -1.0754588, -1.6602297, -1.0764718, -0.1707170, 0.1749031
6: -15.9428062, -15.3116093, -15.9389725, -15.3196259, -0.2179509, 0.2231582
7: -5.4805412, -4.9616933, -5.4735556, -4.9619083, -0.2609181, 0.2530754
8: -2.3089070, -1.8054209, -2.3088384, -1.8051600, -0.1836712, 0.1833856
9: 2.5455329, 2.9001541, 2.5452425, 2.8979614, -0.1268272, 0.1309772

Time for backsubstitution: 7.54 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 416
type: A, layer: 3, pos: 228
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1808
type: A, layer: 3, pos: 659
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 416

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1077852, upper bound: 0.1073913
time: 3.23 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1077852, upper bound: 0.1073914
time: 3.19 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -14.1098938, -13.3910170, -14.0980787, -13.3978519, -0.2677251, 0.2622905
1: -15.8242159, -15.2665367, -15.8215189, -15.2604160, -0.2395337, 0.2324356
2: -8.7451067, -8.2225065, -8.7468910, -8.2234974, -0.1970185, 0.1998401
3: -7.1935987, -6.6233945, -7.2112093, -6.6245909, -0.1955979, 0.2162274
4: -6.7375202, -6.1878090, -6.7318530, -6.1788545, -0.2571260, 0.2406018
5: -1.6588247, -1.0754628, -1.6589546, -1.0728703, -0.1741741, 0.1756704
6: -15.9409370, -15.3116102, -15.9344540, -15.3161869, -0.2226774, 0.2227274
7: -5.4781938, -4.9616933, -5.4687839, -4.9544182, -0.2682617, 0.2537441
8: -2.3088002, -1.8054252, -2.3085446, -1.8051028, -0.1836233, 0.1833491
9: 2.5455427, 2.8991046, 2.5440402, 2.8951125, -0.1264487, 0.1340642

Time for backsubstitution: 8.07 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 228
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 416
type: A, layer: 3, pos: 1808
type: A, layer: 3, pos: 659
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 228

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1075071, upper bound: 0.1074905
time: 3.16 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1079377, upper bound: 0.1074906
time: 2.98 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -14.1137276, -13.3898735, -14.1028366, -13.4017992, -0.2666399, 0.2632134
1: -15.8250771, -15.2674665, -15.8237247, -15.2650614, -0.2415817, 0.2320021
2: -8.7441711, -8.2234879, -8.7455368, -8.2230711, -0.1998610, 0.1980531
3: -7.1977587, -6.6174426, -7.1936679, -6.6207800, -0.2019426, 0.1959052
4: -6.7428951, -6.1858354, -6.7371869, -6.1861219, -0.2475995, 0.2413220
5: -1.6632323, -1.0701327, -1.6615009, -1.0764713, -0.1729230, 0.1891588
6: -15.9441481, -15.3096170, -15.9393520, -15.3196259, -0.2187559, 0.2270857
7: -5.4853096, -4.9580870, -5.4735532, -4.9608097, -0.2682017, 0.2551608
8: -2.3088336, -1.8058491, -2.3088379, -1.8053293, -0.1850827, 0.1832857
9: 2.5454135, 2.9009118, 2.5452509, 2.8979597, -0.1270838, 0.1335711

Time for backsubstitution: 7.53 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 416
type: A, layer: 3, pos: 228
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1808
type: A, layer: 3, pos: 659
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 416

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1079881, upper bound: 0.1073913
time: 3.31 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1079881, upper bound: 0.1073913
time: 3.19 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -14.1117134, -13.3898802, -14.0980806, -13.3975849, -0.2719055, 0.2630551
1: -15.8239584, -15.2674685, -15.8215189, -15.2607994, -0.2445366, 0.2321367
2: -8.7441721, -8.2239399, -8.7468891, -8.2240276, -0.1999512, 0.1992477
3: -7.1977587, -6.6203899, -7.2112093, -6.6237707, -0.2046996, 0.2180132
4: -6.7400498, -6.1858354, -6.7318506, -6.1782627, -0.2597365, 0.2416975
5: -1.6623044, -1.0701337, -1.6602247, -1.0728719, -0.1763800, 0.1899263
6: -15.9422779, -15.3096199, -15.9348354, -15.3161888, -0.2234824, 0.2266550
7: -5.4829621, -4.9580870, -5.4687824, -4.9533195, -0.2755451, 0.2558291
8: -2.3087254, -1.8058534, -2.3085451, -1.8052707, -0.1850350, 0.1832496
9: 2.5454252, 2.8998625, 2.5440502, 2.8951113, -0.1267054, 0.1366583

Time for backsubstitution: 7.49 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 228
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 416
type: A, layer: 3, pos: 1808
type: A, layer: 3, pos: 659
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 228

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1077100, upper bound: 0.1074907
time: 3.06 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1081407, upper bound: 0.1074906
time: 3.15 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -14.1119022, -13.3910112, -14.1102619, -13.3896151, -0.2521031, 0.2487543
1: -15.8253345, -15.2665348, -15.8239765, -15.2655277, -0.2315137, 0.2290310
2: -8.7451086, -8.2220554, -8.7451057, -8.2220192, -0.1965300, 0.1958508
3: -7.1935987, -6.6204462, -7.1935987, -6.6225376, -0.1888038, 0.1901934
4: -6.7403626, -6.1878090, -6.7378073, -6.1863365, -0.2449672, 0.2402387
5: -1.6597555, -1.0754588, -1.6627603, -1.0754628, -0.1647358, 0.1704589
6: -15.9428062, -15.3116093, -15.9428482, -15.3116102, -0.2029874, 0.2033011
7: -5.4805412, -4.9616933, -5.4780965, -4.9584217, -0.2577090, 0.2509160
8: -2.3089070, -1.8054209, -2.3088598, -1.8051052, -0.1836287, 0.1832832
9: 2.5455329, 2.9001541, 2.5446577, 2.8996663, -0.1257548, 0.1270717

Time for backsubstitution: 7.49 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 416
type: A, layer: 3, pos: 228
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1808
type: A, layer: 3, pos: 659
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 416

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1077852, upper bound: 0.1073914
time: 3.23 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1077852, upper bound: 0.1073913
time: 3.35 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -14.1098938, -13.3910170, -14.1055107, -13.3854008, -0.2572827, 0.2485971
1: -15.8242159, -15.2665367, -15.8217697, -15.2612677, -0.2344816, 0.2291650
2: -8.7451067, -8.2225065, -8.7464600, -8.2229118, -0.1966196, 0.1970466
3: -7.1935987, -6.6233945, -7.2110615, -6.6262398, -0.1911502, 0.2124594
4: -6.7375202, -6.1878090, -6.7323050, -6.1786280, -0.2572554, 0.2405391
5: -1.6588247, -1.0754628, -1.6616089, -1.0715861, -0.1682647, 0.1714540
6: -15.9409370, -15.3116102, -15.9383278, -15.3081741, -0.2077134, 0.2028701
7: -5.4781938, -4.9616933, -5.4733286, -4.9509315, -0.2650521, 0.2515628
8: -2.3088002, -1.8054252, -2.3085666, -1.8050461, -0.1835799, 0.1832469
9: 2.5455427, 2.8991046, 2.5434556, 2.8968172, -0.1253754, 0.1302382

Time for backsubstitution: 7.51 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 228
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 416
type: A, layer: 3, pos: 1808
type: A, layer: 3, pos: 659
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 228

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1075070, upper bound: 0.1074906
time: 3.29 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1079377, upper bound: 0.1074905
time: 3.33 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -14.1137276, -13.3898735, -14.1102629, -13.3893490, -0.2562830, 0.2495185
1: -15.8250771, -15.2674665, -15.8239765, -15.2659111, -0.2365160, 0.2287318
2: -8.7441711, -8.2234879, -8.7451067, -8.2225485, -0.1994622, 0.1952587
3: -7.1977587, -6.6174426, -7.1935987, -6.6217165, -0.1979057, 0.1919789
4: -6.7428951, -6.1858354, -6.7378073, -6.1857443, -0.2475783, 0.2413347
5: -1.6632323, -1.0701327, -1.6640286, -1.0754633, -0.1669417, 0.1847146
6: -15.9441481, -15.3096170, -15.9432259, -15.3116093, -0.2037923, 0.2072287
7: -5.4853096, -4.9580870, -5.4780955, -4.9573245, -0.2649921, 0.2530022
8: -2.3088336, -1.8058491, -2.3088598, -1.8052726, -0.1850398, 0.1831832
9: 2.5454135, 2.9009118, 2.5446675, 2.8996654, -0.1260115, 0.1296659

Time for backsubstitution: 8.15 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 416
type: A, layer: 3, pos: 228
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 1808
type: A, layer: 3, pos: 659
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 416

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1079882, upper bound: 0.1073914
time: 3.53 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1079882, upper bound: 0.1073913
time: 4.24 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -14.1117134, -13.3898802, -14.1055088, -13.3851347, -0.2614634, 0.2493618
1: -15.8239584, -15.2674685, -15.8217697, -15.2616501, -0.2394841, 0.2288655
2: -8.7441721, -8.2239399, -8.7464600, -8.2234421, -0.1995521, 0.1964545
3: -7.1977587, -6.6203899, -7.2110615, -6.6254148, -0.2002517, 0.2142451
4: -6.7400498, -6.1858354, -6.7323041, -6.1780348, -0.2598660, 0.2416341
5: -1.6623044, -1.0701337, -1.6628785, -1.0715871, -0.1704707, 0.1857098
6: -15.9422779, -15.3096199, -15.9387102, -15.3081760, -0.2085184, 0.2067975
7: -5.4829621, -4.9580870, -5.4733267, -4.9498343, -0.2723351, 0.2536476
8: -2.3087254, -1.8058534, -2.3085666, -1.8052139, -0.1849915, 0.1831472
9: 2.5454252, 2.8998625, 2.5434663, 2.8968155, -0.1256321, 0.1328322

Time for backsubstitution: 7.56 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 228
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 416
type: A, layer: 3, pos: 1808
type: A, layer: 3, pos: 659
type: A, layer: 3, pos: 2229

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 228

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1077101, upper bound: 0.1074906
time: 3.30 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1081407, upper bound: 0.1074907
time: 3.39 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 14.49 seconds
NS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 14.49
Output dim: 9, lower bound: -0.1073633, upper bound: 0.1075663
NS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 14.49
Output dim: 9, lower bound: -0.1073633, upper bound: 0.1075663
NS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 14.49
Output dim: 9, lower bound: -0.1070829, upper bound: 0.1077166
NS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 14.49
Output dim: 9, lower bound: -0.1075136, upper bound: 0.1077165
NS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 14.49
Output dim: 9, lower bound: -0.1075663, upper bound: 0.1075663
NS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 14.49
Output dim: 9, lower bound: -0.1075663, upper bound: 0.1075663
NS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 14.49
Output dim: 9, lower bound: -0.1072859, upper bound: 0.1077165
NS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 14.49
Output dim: 9, lower bound: -0.1077166, upper bound: 0.1077165
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.49
Output dim: 9, lower bound: -0.1071883, upper bound: 0.1079881
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.49
Output dim: 9, lower bound: -0.1071883, upper bound: 0.1079881
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.49
Output dim: 9, lower bound: -0.1068570, upper bound: 0.1081407
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.49
Output dim: 9, lower bound: -0.1072876, upper bound: 0.1081408
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.49
Output dim: 9, lower bound: -0.1073914, upper bound: 0.1079882
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.49
Output dim: 9, lower bound: -0.1073914, upper bound: 0.1079881
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.49
Output dim: 9, lower bound: -0.1070600, upper bound: 0.1081406
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.49
Output dim: 9, lower bound: -0.1074907, upper bound: 0.1081406
NS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 14.49
Output dim: 9, lower bound: -0.1077852, upper bound: 0.1073913
NS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 14.49
Output dim: 9, lower bound: -0.1077852, upper bound: 0.1073914
NS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 14.49
Output dim: 9, lower bound: -0.1075071, upper bound: 0.1074905
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.49
Output dim: 9, lower bound: -0.1079377, upper bound: 0.1074906
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.49
Output dim: 9, lower bound: -0.1079881, upper bound: 0.1073913
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.49
Output dim: 9, lower bound: -0.1079881, upper bound: 0.1073913
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 14.49
Output dim: 9, lower bound: -0.1077100, upper bound: 0.1074907
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.49
Output dim: 9, lower bound: -0.1081407, upper bound: 0.1074906
NS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 14.49
Output dim: 9, lower bound: -0.1077852, upper bound: 0.1073914
NS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 14.49
Output dim: 9, lower bound: -0.1077852, upper bound: 0.1073913
NS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 14.49
Output dim: 9, lower bound: -0.1075070, upper bound: 0.1074906
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.49
Output dim: 9, lower bound: -0.1079377, upper bound: 0.1074905
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.49
Output dim: 9, lower bound: -0.1079882, upper bound: 0.1073914
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.49
Output dim: 9, lower bound: -0.1079882, upper bound: 0.1073913
NS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 14.49
Output dim: 9, lower bound: -0.1077101, upper bound: 0.1074906
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.49
Output dim: 9, lower bound: -0.1081407, upper bound: 0.1074907

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -14.1028366, -13.4034672, -14.1102619, -13.3896151, -0.2634597, 0.2590070
1: -15.8237247, -15.2656889, -15.8239765, -15.2655277, -0.2334208, 0.2340915
2: -8.7455387, -8.2230587, -8.7451057, -8.2220192, -0.1993244, 0.1957625
3: -7.1936679, -6.6243973, -7.1935987, -6.6225376, -0.1920236, 0.1890568
4: -6.7371869, -6.1881866, -6.7378073, -6.1863365, -0.2418102, 0.2402599
5: -1.6561997, -1.0764718, -1.6627603, -1.0754628, -0.1682642, 0.1764883
6: -15.9375019, -15.3196239, -15.9428482, -15.3116102, -0.2212216, 0.2182574
7: -5.4735537, -4.9651780, -5.4780965, -4.9584217, -0.2571144, 0.2541261
8: -2.3088388, -1.8054829, -2.3088598, -1.8051052, -0.1836798, 0.1833117
9: 2.5461335, 2.8979602, 2.5446577, 2.8996663, -0.1296625, 0.1273456

Time for backsubstitution: 8.11 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 228
type: B, layer: 3, pos: 1109
type: B, layer: 3, pos: 1195
type: B, layer: 3, pos: 1808
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 228

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1076174, upper bound: 0.1074739
time: 3.28 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1076174, upper bound: 0.1074371
time: 3.25 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -14.0980816, -13.3992558, -14.1102619, -13.3896151, -0.2626572, 0.2653670
1: -15.8215189, -15.2614269, -15.8239765, -15.2655277, -0.2314882, 0.2381575
2: -8.7468901, -8.2240162, -8.7451057, -8.2220192, -0.2009640, 0.1951228
3: -7.2112093, -6.6273880, -7.1935987, -6.6225376, -0.2148556, 0.1931252
4: -6.7318525, -6.1803284, -6.7378073, -6.1863365, -0.2418140, 0.2543416
5: -1.6549261, -1.0728707, -1.6627603, -1.0754628, -0.1677029, 0.1808318
6: -15.9329844, -15.3161869, -15.9428482, -15.3116102, -0.2197653, 0.2241967
7: -5.4687839, -4.9576893, -5.4780965, -4.9584217, -0.2546546, 0.2637229
8: -2.3085446, -1.8054261, -2.3088598, -1.8051052, -0.1834074, 0.1833644
9: 2.5449307, 2.8951128, 2.5446577, 2.8996663, -0.1333264, 0.1273640

Time for backsubstitution: 7.42 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 228
type: B, layer: 3, pos: 1109
type: B, layer: 3, pos: 1195
type: B, layer: 3, pos: 1808
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 228

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1076174, upper bound: 0.1074739
time: 3.23 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1076174, upper bound: 0.1074371
time: 3.26 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -14.0959101, -13.4034662, -14.1039715, -13.3854008, -0.2621720, 0.2563546
1: -15.8228569, -15.2656879, -15.8215199, -15.2612686, -0.2364826, 0.2339793
2: -8.7433653, -8.2230339, -8.7459631, -8.2229118, -0.1977172, 0.1968668
3: -7.1936679, -6.6220703, -7.2110615, -6.6263142, -0.1943632, 0.2152691
4: -6.7369008, -6.1952219, -6.7323046, -6.1803241, -0.2551882, 0.2316239
5: -1.6562939, -1.0765269, -1.6616089, -1.0715995, -0.1721355, 0.1772130
6: -15.9324799, -15.3196239, -15.9372082, -15.3081751, -0.2207713, 0.2162651
7: -5.4731073, -4.9651780, -5.4732056, -4.9509315, -0.2660484, 0.2546906
8: -2.3087778, -1.8072500, -2.3085661, -1.8054600, -0.1827408, 0.1797191
9: 2.5469804, 2.8974013, 2.5436563, 2.8968155, -0.1286249, 0.1309350

Time for backsubstitution: 7.53 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1109
type: B, layer: 3, pos: 1195
type: B, layer: 3, pos: 1808
type: B, layer: 3, pos: 228
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 1109

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1065863, upper bound: 0.1055563
time: 3.58 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1065864, upper bound: 0.1078716
time: 3.60 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -14.0890789, -13.3998699, -14.1006927, -13.3854046, -0.2617421, 0.2718956
1: -15.8211937, -15.2653179, -15.8207293, -15.2612686, -0.2363865, 0.2334536
2: -8.7431116, -8.2212229, -8.7455406, -8.2229128, -0.1978670, 0.2003587
3: -7.1913166, -6.6268759, -7.2110615, -6.6281066, -0.1963447, 0.2135488
4: -6.7470822, -6.1941843, -6.7323055, -6.1806812, -0.2679361, 0.2340569
5: -1.6563036, -1.0765824, -1.6616089, -1.0716317, -0.1720875, 0.1771874
6: -15.9338789, -15.3132439, -15.9370308, -15.3081741, -0.2223775, 0.2276776
7: -5.4719868, -4.9653673, -5.4727087, -4.9509315, -0.2657127, 0.2542739
8: -2.3082166, -1.8121576, -2.3085666, -1.8074503, -0.1877952, 0.1782854
9: 2.5473566, 2.8982177, 2.5438833, 2.8968155, -0.1287492, 0.1315036

Time for backsubstitution: 7.53 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1109
type: B, layer: 3, pos: 1195
type: B, layer: 3, pos: 1808
type: B, layer: 3, pos: 228
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 1109

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1070185, upper bound: 0.1055575
time: 3.68 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1070186, upper bound: 0.1078715
time: 3.52 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -14.1046562, -13.4023314, -14.1102629, -13.3893490, -0.2676392, 0.2597713
1: -15.8234673, -15.2666187, -15.8239765, -15.2659111, -0.2384236, 0.2337925
2: -8.7446022, -8.2244921, -8.7451067, -8.2225485, -0.2022564, 0.1951702
3: -7.1978269, -6.6213923, -7.1935987, -6.6217165, -0.2011253, 0.1908423
4: -6.7397184, -6.1862125, -6.7378073, -6.1857443, -0.2444211, 0.2413566
5: -1.6596794, -1.0711398, -1.6640286, -1.0754633, -0.1704700, 0.1907437
6: -15.9388399, -15.3176365, -15.9432259, -15.3116093, -0.2220268, 0.2221850
7: -5.4783220, -4.9615726, -5.4780955, -4.9573245, -0.2643988, 0.2562118
8: -2.3087630, -1.8059096, -2.3088598, -1.8052726, -0.1850908, 0.1832120
9: 2.5460141, 2.8987198, 2.5446675, 2.8996654, -0.1299193, 0.1299400

Time for backsubstitution: 8.15 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 228
type: B, layer: 3, pos: 1109
type: B, layer: 3, pos: 1195
type: B, layer: 3, pos: 1808
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 228

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1078204, upper bound: 0.1074738
time: 3.55 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1078204, upper bound: 0.1074371
time: 3.51 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -14.0999012, -13.3981199, -14.1102629, -13.3893490, -0.2668366, 0.2661312
1: -15.8212595, -15.2623577, -15.8239765, -15.2659111, -0.2364907, 0.2378585
2: -8.7459545, -8.2254467, -8.7451067, -8.2225485, -0.2038960, 0.1945305
3: -7.2153702, -6.6243811, -7.1935987, -6.6217165, -0.2239572, 0.1949109
4: -6.7343836, -6.1783524, -6.7378073, -6.1857443, -0.2444252, 0.2554381
5: -1.6584063, -1.0675416, -1.6640286, -1.0754633, -0.1699088, 0.1950887
6: -15.9343214, -15.3142014, -15.9432259, -15.3116093, -0.2205706, 0.2281253
7: -5.4735527, -4.9540815, -5.4780955, -4.9573245, -0.2619407, 0.2658086
8: -2.3084707, -1.8058519, -2.3088598, -1.8052726, -0.1848186, 0.1832650
9: 2.5448112, 2.8958700, 2.5446675, 2.8996654, -0.1335833, 0.1299584

Time for backsubstitution: 8.12 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 228
type: B, layer: 3, pos: 1109
type: B, layer: 3, pos: 1195
type: B, layer: 3, pos: 1808
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 228

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1078204, upper bound: 0.1074738
time: 3.32 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1078204, upper bound: 0.1074370
time: 3.32 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -14.0977287, -13.4023333, -14.1039715, -13.3851347, -0.2663531, 0.2571187
1: -15.8225965, -15.2666187, -15.8215199, -15.2616501, -0.2414861, 0.2336799
2: -8.7424307, -8.2244663, -8.7459641, -8.2234440, -0.2006493, 0.1962743
3: -7.1978269, -6.6190643, -7.2110615, -6.6254911, -0.2034649, 0.2170546
4: -6.7394309, -6.1932487, -6.7323031, -6.1797352, -0.2577980, 0.2327197
5: -1.6597729, -1.0711966, -1.6628785, -1.0715966, -0.1743414, 0.1914685
6: -15.9338207, -15.3176346, -15.9375916, -15.3081751, -0.2215765, 0.2201927
7: -5.4778738, -4.9615726, -5.4732046, -4.9498343, -0.2733307, 0.2567761
8: -2.3087020, -1.8076773, -2.3085675, -1.8056283, -0.1841524, 0.1796191
9: 2.5468609, 2.8981569, 2.5436649, 2.8968155, -0.1288810, 0.1335291

Time for backsubstitution: 8.19 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1109
type: B, layer: 3, pos: 1195
type: B, layer: 3, pos: 1808
type: B, layer: 3, pos: 228
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 2229

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 1109

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1067893, upper bound: 0.1055563
time: 3.70 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1067893, upper bound: 0.1078714
time: 3.44 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -14.0908976, -13.3987322, -14.1006918, -13.3851347, -0.2659228, 0.2726598
1: -15.8209400, -15.2662516, -15.8207293, -15.2616501, -0.2413902, 0.2331547
2: -8.7421741, -8.2226543, -8.7455406, -8.2234421, -0.2007992, 0.1997664
3: -7.1954775, -6.6238699, -7.2110615, -6.6272845, -0.2054471, 0.2153343
4: -6.7496138, -6.1922112, -6.7323027, -6.1800909, -0.2705474, 0.2351532
5: -1.6597815, -1.0712519, -1.6628785, -1.0716293, -0.1742935, 0.1914428
6: -15.9352188, -15.3112526, -15.9374113, -15.3081760, -0.2231827, 0.2316053
7: -5.4767551, -4.9617615, -5.4727087, -4.9498343, -0.2729964, 0.2563598
8: -2.3081427, -1.8125834, -2.3085670, -1.8076162, -0.1892066, 0.1781857
9: 2.5472369, 2.8989756, 2.5438941, 2.8968153, -0.1290059, 0.1340982

Time for backsubstitution: 8.09 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 55.09 + 548.41 = 603.50 seconds
