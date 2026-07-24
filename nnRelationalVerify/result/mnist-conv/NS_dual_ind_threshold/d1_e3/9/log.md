## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.117360782


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (8.2021646, 8.8049355, 8.2021646, 8.8049355, -0.1806938, 0.1806939)
1: (-14.5419455, -13.6810780, -14.5419455, -13.6810780, -0.3769088, 0.3769090)
2: (-4.4596033, -3.7353108, -4.4596033, -3.7353108, -0.3111380, 0.3111379)
3: (-11.2466173, -10.5855494, -11.2466173, -10.5855494, -0.2081676, 0.2081676)
4: (-10.9946346, -10.2116566, -10.9946346, -10.2116566, -0.2295593, 0.2295593)
5: (-5.0402365, -4.4762449, -5.0402365, -4.4762449, -0.1522999, 0.1522999)
6: (-3.7309656, -3.1356418, -3.7309656, -3.1356418, -0.1604722, 0.1604723)
7: (-10.1710739, -9.2903175, -10.1710739, -9.2903175, -0.3422725, 0.3422724)
8: (-3.1486435, -2.5063982, -3.1486435, -2.5063982, -0.2667599, 0.2667599)
9: (-2.4534123, -1.7026299, -2.4534123, -1.7026299, -0.2661802, 0.2661802)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.64 + 34.85 = 57.49 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.1197558, upper bound: 0.1197559

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 4617
type: A, layer: 1, pos: 161

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5859

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188654, upper bound: 0.1197550
time: 3.41 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1197546, upper bound: 0.1197549
time: 4.65 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 8.21 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 8.21
Output dim: 0, lower bound: -0.1188654, upper bound: 0.1197550
NS_A2, status: Status.UNKNOWN, split count: 1, time: 8.21
Output dim: 0, lower bound: -0.1197546, upper bound: 0.1197549

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 8.2040825, 8.8049297, 8.2029085, 8.8049326, -0.1787168, 0.1799191
1: -14.5390377, -13.6811390, -14.5408134, -13.6810999, -0.3738449, 0.3755715
2: -4.4584866, -3.7353880, -4.4591699, -3.7353406, -0.3099728, 0.3106673
3: -11.2465887, -10.5859985, -11.2466068, -10.5857258, -0.2079762, 0.2076983
4: -10.9946289, -10.2178402, -10.9946299, -10.2140617, -0.2271008, 0.2232417
5: -5.0401278, -4.4762635, -5.0401936, -4.4762526, -0.1521274, 0.1522089
6: -3.7287033, -3.1356723, -3.7300870, -3.1356537, -0.1581967, 0.1595654
7: -10.1709795, -9.3000259, -10.1710377, -9.2940931, -0.3384063, 0.3324440
8: -3.1462250, -2.5063982, -3.1477013, -2.5063982, -0.2642513, 0.2657855
9: -2.4534123, -1.7075442, -2.4534123, -1.7045407, -0.2642255, 0.2611518

Time for backsubstitution: 21.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4617
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 161

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4617

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1197522
time: 4.32 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188627, upper bound: 0.1197522
time: 4.47 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 8.2019653, 8.8090448, 8.2021656, 8.8049355, -0.1806902, 0.1838027
1: -14.5424004, -13.6752081, -14.5419455, -13.6810780, -0.3773415, 0.3817124
2: -4.4597306, -3.7324133, -4.4596043, -3.7353089, -0.3107970, 0.3143048
3: -11.2477875, -10.5855141, -11.2466183, -10.5855484, -0.2094302, 0.2080227
4: -11.0078020, -10.2111034, -10.9946346, -10.2116623, -0.2317423, 0.2279408
5: -5.0404801, -4.4759016, -5.0402369, -4.4762449, -0.1527032, 0.1526580
6: -3.7309737, -3.1307862, -3.7309623, -3.1356416, -0.1593892, 0.1622376
7: -10.1918659, -9.2898607, -10.1710749, -9.2903233, -0.3484126, 0.3390590
8: -3.1492672, -2.5015011, -3.1486430, -2.5063982, -0.2666278, 0.2685788
9: -2.4637671, -1.7019699, -2.4534123, -1.7026336, -0.2687519, 0.2655094

Time for backsubstitution: 20.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4617
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 161

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4617

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188276, upper bound: 0.1197525
time: 3.72 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1197520, upper bound: 0.1197525
time: 3.37 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 27.59 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 27.59
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1197522
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 27.59
Output dim: 0, lower bound: -0.1188627, upper bound: 0.1197522
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 27.59
Output dim: 0, lower bound: -0.1188276, upper bound: 0.1197525
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 27.59
Output dim: 0, lower bound: -0.1197520, upper bound: 0.1197525

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: 8.2040844, 8.8029928, 8.2044907, 8.8009071, -0.1746844, 0.1763912
1: -14.5372162, -13.6811762, -14.5368404, -13.6828327, -0.3703201, 0.3715782
2: -4.4584327, -3.7355695, -4.4586549, -3.7357583, -0.3095286, 0.3099134
3: -11.2434616, -10.5860062, -11.2401276, -10.5883312, -0.2022060, 0.2012035
4: -10.9946222, -10.2191820, -10.9934845, -10.2168455, -0.2243029, 0.2206801
5: -5.0381737, -4.4762673, -5.0361443, -4.4778929, -0.1484876, 0.1481436
6: -3.7286427, -3.1356831, -3.7299552, -3.1357841, -0.1579803, 0.1594319
7: -10.1709385, -9.3001013, -10.1707306, -9.2942543, -0.3382416, 0.3320795
8: -3.1443434, -2.5063992, -3.1436787, -2.5079956, -0.2606596, 0.2618012
9: -2.4534116, -1.7075820, -2.4533482, -1.7047466, -0.2639673, 0.2610618

Time for backsubstitution: 20.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4617
type: A, layer: 1, pos: 161

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4617

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1188280
time: 3.58 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1197522
time: 3.80 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: 8.2040825, 8.8049288, 8.2029085, 8.8049297, -0.1741178, 0.1799185
1: -14.5390377, -13.6811371, -14.5408096, -13.6811018, -0.3738434, 0.3713017
2: -4.4584866, -3.7353890, -4.4591694, -3.7353401, -0.3095930, 0.3106670
3: -11.2465849, -10.5859995, -11.2466011, -10.5857248, -0.2074944, 0.2001824
4: -10.9946289, -10.2178402, -10.9946308, -10.2140608, -0.2239552, 0.2232168
5: -5.0401268, -4.4762640, -5.0401888, -4.4762516, -0.1518061, 0.1475182
6: -3.7287033, -3.1356726, -3.7300861, -3.1356521, -0.1581967, 0.1594359
7: -10.1709776, -9.3000259, -10.1710367, -9.2940912, -0.3383226, 0.3324441
8: -3.1462240, -2.5063982, -3.1476984, -2.5063982, -0.2642511, 0.2614679
9: -2.4534123, -1.7075423, -2.4534123, -1.7045414, -0.2642094, 0.2611256

Time for backsubstitution: 21.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4617
type: A, layer: 1, pos: 161

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 4617

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188628, upper bound: 0.1188278
time: 4.25 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188628, upper bound: 0.1197522
time: 3.81 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 8.2019663, 8.8071079, 8.2037497, 8.8009090, -0.1766577, 0.1791235
1: -14.5405779, -13.6752434, -14.5379696, -13.6828117, -0.3738174, 0.3777126
2: -4.4596782, -3.7325957, -4.4590878, -3.7357271, -0.3103527, 0.3135505
3: -11.2446604, -10.5855227, -11.2401400, -10.5881567, -0.2029295, 0.2015278
4: -11.0077944, -10.2124472, -10.9934883, -10.2144470, -0.2289407, 0.2253790
5: -5.0385261, -4.4759059, -5.0361862, -4.4778852, -0.1490633, 0.1485927
6: -3.7309129, -3.1307991, -3.7308311, -3.1357727, -0.1591730, 0.1621041
7: -10.1918249, -9.2899361, -10.1707668, -9.2904882, -0.3482448, 0.3386954
8: -3.1473875, -2.5015001, -3.1446209, -2.5079956, -0.2630355, 0.2645931
9: -2.4637666, -1.7020066, -2.4533482, -1.7028401, -0.2684934, 0.2654194

Time for backsubstitution: 21.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4617
type: A, layer: 1, pos: 161

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 4617

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188276, upper bound: 0.1188281
time: 3.74 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188276, upper bound: 0.1197525
time: 3.29 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 8.2019653, 8.8090420, 8.2021656, 8.8049297, -0.1760918, 0.1822074
1: -14.5424004, -13.6752052, -14.5419369, -13.6810780, -0.3773408, 0.3773670
2: -4.4597311, -3.7324142, -4.4596038, -3.7353094, -0.3104174, 0.3143044
3: -11.2477856, -10.5855160, -11.2466135, -10.5855494, -0.2079784, 0.2005069
4: -11.0078030, -10.2111063, -10.9946346, -10.2116632, -0.2285526, 0.2279158
5: -5.0404792, -4.4759016, -5.0402331, -4.4762459, -0.1522305, 0.1479676
6: -3.7309740, -3.1307874, -3.7309623, -3.1356411, -0.1593893, 0.1620906
7: -10.1918650, -9.2898588, -10.1710739, -9.2903242, -0.3482536, 0.3390595
8: -3.1492653, -2.5015011, -3.1486406, -2.5063982, -0.2666274, 0.2641462
9: -2.4637671, -1.7019699, -2.4534123, -1.7026348, -0.2686592, 0.2654834

Time for backsubstitution: 21.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4617
type: A, layer: 1, pos: 161

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4617

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1197521, upper bound: 0.1188278
time: 3.52 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1197521, upper bound: 0.1197523
time: 3.67 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 28.79 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 28.79
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1188280
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.79
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1197522
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.79
Output dim: 0, lower bound: -0.1188628, upper bound: 0.1188278
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.79
Output dim: 0, lower bound: -0.1188628, upper bound: 0.1197522
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 28.79
Output dim: 0, lower bound: -0.1188276, upper bound: 0.1188281
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.79
Output dim: 0, lower bound: -0.1188276, upper bound: 0.1197525
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.79
Output dim: 0, lower bound: -0.1197521, upper bound: 0.1188278
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.79
Output dim: 0, lower bound: -0.1197521, upper bound: 0.1197523

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: 8.2056618, 8.8009033, 8.2044907, 8.8009071, -0.1730980, 0.1743006
1: -14.5350647, -13.6828709, -14.5368404, -13.6828327, -0.3681588, 0.3698862
2: -4.4579716, -3.7358062, -4.4586549, -3.7357583, -0.3090518, 0.3097463
3: -11.2401085, -10.5886078, -11.2401276, -10.5883312, -0.1988459, 0.1985683
4: -10.9934816, -10.2206230, -10.9934845, -10.2168455, -0.2231464, 0.2192874
5: -5.0360765, -4.4779043, -5.0361443, -4.4778929, -0.1464029, 0.1464843
6: -3.7285726, -3.1358030, -3.7299552, -3.1357841, -0.1579251, 0.1592938
7: -10.1706734, -9.3001890, -10.1707306, -9.2942543, -0.3379455, 0.3319829
8: -3.1421995, -2.5079956, -3.1436787, -2.5079956, -0.2586595, 0.2601941
9: -2.4533482, -1.7077494, -2.4533482, -1.7047466, -0.2639005, 0.2608267

Time for backsubstitution: 21.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 161

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1179399
time: 3.31 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1188283
time: 3.65 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: 8.2040825, 8.8049240, 8.2044907, 8.8009071, -0.1746844, 0.1781556
1: -14.5390320, -13.6811390, -14.5368404, -13.6828327, -0.3721437, 0.3715847
2: -4.4584861, -3.7353890, -4.4586549, -3.7357583, -0.3095553, 0.3101635
3: -11.2465820, -10.5859985, -11.2401276, -10.5883312, -0.2024482, 0.2007584
4: -10.9946289, -10.2178411, -10.9934845, -10.2168455, -0.2242913, 0.2220715
5: -5.0401249, -4.4762635, -5.0361443, -4.4778929, -0.1486502, 0.1478098
6: -3.7287037, -3.1356723, -3.7299552, -3.1357841, -0.1580529, 0.1594376
7: -10.1709785, -9.3000269, -10.1707306, -9.2942543, -0.3382647, 0.3321247
8: -3.1462212, -2.5063982, -3.1436787, -2.5079956, -0.2615730, 0.2617860
9: -2.4534123, -1.7075440, -2.4533482, -1.7047466, -0.2639647, 0.2610180

Time for backsubstitution: 21.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 161

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1188641
time: 3.92 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1197522
time: 4.30 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: 8.2056618, 8.8009033, 8.2029085, 8.8049297, -0.1771281, 0.1758870
1: -14.5350647, -13.6828709, -14.5408096, -13.6811018, -0.3698571, 0.3738708
2: -4.4579716, -3.7358062, -4.4591694, -3.7353401, -0.3094693, 0.3102493
3: -11.2401085, -10.5886078, -11.2466011, -10.5857248, -0.2009972, 0.2022094
4: -10.9934816, -10.2206230, -10.9946308, -10.2140608, -0.2253648, 0.2204323
5: -5.0360765, -4.4779043, -5.0401888, -4.4762516, -0.1477442, 0.1487157
6: -3.7285726, -3.1358030, -3.7300861, -3.1356521, -0.1580689, 0.1594215
7: -10.1706734, -9.3001890, -10.1710367, -9.2940912, -0.3380868, 0.3323021
8: -3.1421995, -2.5079956, -3.1476984, -2.5063982, -0.2602666, 0.2628782
9: -2.4533482, -1.7077494, -2.4534123, -1.7045414, -0.2640920, 0.2608910

Time for backsubstitution: 21.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 161

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1179394
time: 4.46 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1188280
time: 3.79 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: 8.2040825, 8.8049240, 8.2029085, 8.8049297, -0.1741178, 0.1753205
1: -14.5390320, -13.6811390, -14.5408096, -13.6811018, -0.3695741, 0.3713012
2: -4.4584861, -3.7353890, -4.4591694, -3.7353401, -0.3095930, 0.3102875
3: -11.2465820, -10.5859985, -11.2466011, -10.5857248, -0.2004602, 0.2001826
4: -10.9946289, -10.2178411, -10.9946308, -10.2140608, -0.2239550, 0.2200959
5: -5.0401249, -4.4762635, -5.0401888, -4.4762516, -0.1474367, 0.1475182
6: -3.7287037, -3.1356723, -3.7300861, -3.1356521, -0.1580673, 0.1594359
7: -10.1709785, -9.3000269, -10.1710367, -9.2940912, -0.3383229, 0.3323606
8: -3.1462212, -2.5063982, -3.1476984, -2.5063982, -0.2599337, 0.2614679
9: -2.4534123, -1.7075440, -2.4534123, -1.7045414, -0.2642094, 0.2611355

Time for backsubstitution: 21.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 161

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1188641
time: 3.23 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1197521
time: 4.61 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: 8.2035465, 8.8050175, 8.2037497, 8.8009090, -0.1750715, 0.1781682
1: -14.5384293, -13.6769352, -14.5379696, -13.6828117, -0.3716564, 0.3760084
2: -4.4592161, -3.7328293, -4.4590878, -3.7357271, -0.3098754, 0.3133833
3: -11.2413063, -10.5881243, -11.2401400, -10.5881567, -0.2003002, 0.1988926
4: -11.0066566, -10.2138882, -10.9934883, -10.2144470, -0.2277802, 0.2239864
5: -5.0364289, -4.4775410, -5.0361862, -4.4778852, -0.1469787, 0.1469334
6: -3.7308433, -3.1309190, -3.7308311, -3.1357727, -0.1591179, 0.1619653
7: -10.1915569, -9.2900257, -10.1707668, -9.2904882, -0.3479459, 0.3385987
8: -3.1452456, -2.5030975, -3.1446209, -2.5079956, -0.2610363, 0.2629836
9: -2.4637039, -1.7021768, -2.4533482, -1.7028401, -0.2684259, 0.2651833

Time for backsubstitution: 21.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 161

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188274, upper bound: 0.1179387
time: 3.92 seconds

## Relational analysis of NS_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1181224, upper bound: 0.1188168
time: 2.86 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188039, upper bound: 0.1188043
time: 4.82 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: 8.2019653, 8.8090420, 8.2037497, 8.8009090, -0.1766579, 0.1791328
1: -14.5423956, -13.6752081, -14.5379696, -13.6828117, -0.3756406, 0.3761054
2: -4.4597321, -3.7324126, -4.4590878, -3.7357271, -0.3103794, 0.3138008
3: -11.2477808, -10.5855150, -11.2401400, -10.5881567, -0.2029324, 0.2010851
4: -11.0078030, -10.2111063, -10.9934883, -10.2144470, -0.2278131, 0.2263631
5: -5.0404768, -4.4759016, -5.0361862, -4.4778852, -0.1490746, 0.1479473
6: -3.7309747, -3.1307859, -3.7308311, -3.1357727, -0.1592456, 0.1620377
7: -10.1918650, -9.2898607, -10.1707668, -9.2904882, -0.3481438, 0.3387401
8: -3.1492643, -2.5015011, -3.1446209, -2.5079956, -0.2637469, 0.2629838
9: -2.4637671, -1.7019703, -2.4533482, -1.7028401, -0.2684300, 0.2653755

Time for backsubstitution: 21.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 161

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188274, upper bound: 0.1188630
time: 4.03 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188278, upper bound: 0.1188629
time: 6.17 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: 8.2035465, 8.8050175, 8.2021656, 8.8049297, -0.1787912, 0.1781684
1: -14.5384293, -13.6769352, -14.5419369, -13.6810780, -0.3733549, 0.3771576
2: -4.4592161, -3.7328293, -4.4596038, -3.7353094, -0.3102932, 0.3138869
3: -11.2413063, -10.5881243, -11.2466135, -10.5855494, -0.2014815, 0.2025359
4: -11.0066566, -10.2138882, -10.9946346, -10.2116632, -0.2284335, 0.2251312
5: -5.0364289, -4.4775410, -5.0402331, -4.4762459, -0.1481687, 0.1488532
6: -3.7308433, -3.1309190, -3.7309623, -3.1356411, -0.1592615, 0.1620031
7: -10.1915569, -9.2900257, -10.1710739, -9.2903242, -0.3480074, 0.3389179
8: -3.1452456, -2.5030975, -3.1486406, -2.5063982, -0.2626434, 0.2640762
9: -2.4637039, -1.7021768, -2.4534123, -1.7026348, -0.2686061, 0.2652476

Time for backsubstitution: 22.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 161

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188274, upper bound: 0.1179383
time: 4.29 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188278, upper bound: 0.1179388
time: 3.85 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: 8.2019653, 8.8090420, 8.2021656, 8.8049297, -0.1760918, 0.1791679
1: -14.5423956, -13.6752081, -14.5419369, -13.6810780, -0.3730712, 0.3775195
2: -4.4597321, -3.7324126, -4.4596038, -3.7353094, -0.3104177, 0.3139260
3: -11.2477808, -10.5855150, -11.2466135, -10.5855494, -0.2019145, 0.2005069
4: -11.0078030, -10.2111063, -10.9946346, -10.2116632, -0.2285947, 0.2247950
5: -5.0404768, -4.4759016, -5.0402331, -4.4762459, -0.1480125, 0.1479677
6: -3.7309747, -3.1307859, -3.7309623, -3.1356411, -0.1592597, 0.1621386
7: -10.1918650, -9.2898607, -10.1710739, -9.2903242, -0.3484179, 0.3389760
8: -3.1492643, -2.5015011, -3.1486406, -2.5063982, -0.2623112, 0.2642648
9: -2.4637671, -1.7019703, -2.4534123, -1.7026348, -0.2687358, 0.2654929

Time for backsubstitution: 22.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 161

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188274, upper bound: 0.1188629
time: 4.19 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188278, upper bound: 0.1188632
time: 4.82 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 31.62 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.62
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1179399
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.62
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1188283
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.62
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1188641
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.62
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1197522
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.62
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1179394
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.62
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1188280
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.62
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1188641
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.62
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1197521
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.62
Output dim: 0, lower bound: -0.1181224, upper bound: 0.1188168
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.62
Output dim: 0, lower bound: -0.1188039, upper bound: 0.1188043
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.62
Output dim: 0, lower bound: -0.1188274, upper bound: 0.1188630
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.62
Output dim: 0, lower bound: -0.1188278, upper bound: 0.1188629
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.62
Output dim: 0, lower bound: -0.1188274, upper bound: 0.1179383
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.62
Output dim: 0, lower bound: -0.1188278, upper bound: 0.1179388
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.62
Output dim: 0, lower bound: -0.1188274, upper bound: 0.1188629
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.62
Output dim: 0, lower bound: -0.1188278, upper bound: 0.1188632

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 8.2056618, 8.8009033, 8.2056618, 8.8009033, -0.1730936, 0.1730936
1: -14.5350647, -13.6828709, -14.5350647, -13.6828709, -0.3680546, 0.3680546
2: -4.4579716, -3.7358062, -4.4579716, -3.7358062, -0.3090394, 0.3090391
3: -11.2401085, -10.5886078, -11.2401085, -10.5886078, -0.1985621, 0.1985621
4: -10.9934816, -10.2206230, -10.9934816, -10.2206230, -0.2192852, 0.2192852
5: -5.0360765, -4.4779043, -5.0360765, -4.4779043, -0.1463856, 0.1463856
6: -3.7285726, -3.1358030, -3.7285726, -3.1358030, -0.1579089, 0.1579089
7: -10.1706734, -9.3001890, -10.1706734, -9.3001890, -0.3319504, 0.3319504
8: -3.1421995, -2.5079956, -3.1421995, -2.5079956, -0.2586595, 0.2586596
9: -2.4533482, -1.7077494, -2.4533482, -1.7077494, -0.2608266, 0.2608267

Time for backsubstitution: 21.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1179276, upper bound: 0.1172349
time: 4.55 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1179145, upper bound: 0.1179159
time: 3.90 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 8.2056618, 8.8009033, 8.2035465, 8.8050175, -0.1761934, 0.1755310
1: -14.5350647, -13.6828709, -14.5384293, -13.6769352, -0.3730211, 0.3720479
2: -4.4579716, -3.7358062, -4.4592161, -3.7328293, -0.3122270, 0.3102937
3: -11.2401085, -10.5886078, -11.2413063, -10.5881243, -0.1990842, 0.1998354
4: -10.9934816, -10.2206230, -11.0066566, -10.2138882, -0.2255820, 0.2214646
5: -5.0360765, -4.4779043, -5.0364289, -4.4775410, -0.1467721, 0.1467987
6: -3.7285726, -3.1358030, -3.7308433, -3.1309190, -0.1596991, 0.1601815
7: -10.1706734, -9.3001890, -10.1915569, -9.2900257, -0.3425915, 0.3381337
8: -3.1421995, -2.5079956, -3.1452456, -2.5030975, -0.2604803, 0.2618353
9: -2.4533482, -1.7077494, -2.4637039, -1.7021768, -0.2668515, 0.2633952

Time for backsubstitution: 21.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1179276, upper bound: 0.1181231
time: 3.64 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1179145, upper bound: 0.1188043
time: 4.10 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 8.2040825, 8.8049240, 8.2056618, 8.8009033, -0.1746800, 0.1771237
1: -14.5390320, -13.6811390, -14.5350647, -13.6828709, -0.3720396, 0.3697531
2: -4.4584861, -3.7353890, -4.4579716, -3.7358062, -0.3095424, 0.3094563
3: -11.2465820, -10.5859985, -11.2401085, -10.5886078, -0.2022035, 0.2007524
4: -10.9946289, -10.2178411, -10.9934816, -10.2206230, -0.2204301, 0.2220694
5: -5.0401249, -4.4762635, -5.0360765, -4.4779043, -0.1486321, 0.1477262
6: -3.7287037, -3.1356723, -3.7285726, -3.1358030, -0.1580367, 0.1580527
7: -10.1709785, -9.3000269, -10.1706734, -9.3001890, -0.3322697, 0.3320923
8: -3.1462212, -2.5063982, -3.1421995, -2.5079956, -0.2615730, 0.2602665
9: -2.4534123, -1.7075440, -2.4533482, -1.7077494, -0.2608910, 0.2610180

Time for backsubstitution: 21.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1179274, upper bound: 0.1181590
time: 3.40 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1179143, upper bound: 0.1188397
time: 5.06 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 8.2040825, 8.8049240, 8.2035465, 8.8050175, -0.1761935, 0.1787020
1: -14.5390320, -13.6811390, -14.5384293, -13.6769352, -0.3741705, 0.3737464
2: -4.4584861, -3.7353890, -4.4592161, -3.7328293, -0.3127306, 0.3107111
3: -11.2465820, -10.5859985, -11.2413063, -10.5881243, -0.2025236, 0.2010148
4: -10.9946289, -10.2178411, -11.0066566, -10.2138882, -0.2256149, 0.2221180
5: -5.0401249, -4.4762635, -5.0364289, -4.4775410, -0.1486918, 0.1481003
6: -3.7287037, -3.1356723, -3.7308433, -3.1309190, -0.1597369, 0.1603252
7: -10.1709785, -9.3000269, -10.1915569, -9.2900257, -0.3429108, 0.3381953
8: -3.1462212, -2.5063982, -3.1452456, -2.5030975, -0.2615730, 0.2625974
9: -2.4534123, -1.7075440, -2.4637039, -1.7021768, -0.2669159, 0.2635752

Time for backsubstitution: 21.67 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 57.49 + 547.44 = 604.93 seconds
