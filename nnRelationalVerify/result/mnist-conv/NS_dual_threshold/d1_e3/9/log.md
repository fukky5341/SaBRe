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
execution time: IAR + RelationalAnalysis = 22.39 + 34.69 = 57.07 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.1197558, upper bound: 0.1197559

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 4617
type: B, layer: 1, pos: 4617
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5859

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188654, upper bound: 0.1197550
time: 3.44 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1197546, upper bound: 0.1197549
time: 4.74 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 8.40 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 8.40
Output dim: 0, lower bound: -0.1188654, upper bound: 0.1197550
NS_A2, status: Status.UNKNOWN, split count: 1, time: 8.40
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

Time for backsubstitution: 22.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4617
type: A, layer: 1, pos: 4617
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 4617

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1197522
time: 4.41 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188627, upper bound: 0.1197522
time: 4.40 seconds

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

Time for backsubstitution: 21.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4617
type: B, layer: 1, pos: 4617
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4617

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1197521, upper bound: 0.1188279
time: 3.72 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1197521, upper bound: 0.1197524
time: 3.27 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 28.26 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 28.26
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1197522
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 28.26
Output dim: 0, lower bound: -0.1188627, upper bound: 0.1197522
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 28.26
Output dim: 0, lower bound: -0.1197521, upper bound: 0.1188279
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 28.26
Output dim: 0, lower bound: -0.1197521, upper bound: 0.1197524

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

Time for backsubstitution: 22.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 4617
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of NS_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1188641
time: 3.80 seconds

## Relational analysis of NS_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1197522
time: 4.16 seconds

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

Time for backsubstitution: 21.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 4617
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188627, upper bound: 0.1188638
time: 5.01 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188627, upper bound: 0.1197522
time: 4.51 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: 8.2035465, 8.8050175, 8.2021656, 8.8029985, -0.1771625, 0.1797636
1: -14.5384293, -13.6769352, -14.5401192, -13.6811132, -0.3733482, 0.3771180
2: -4.4592161, -3.7328293, -4.4595490, -3.7354918, -0.3100431, 0.3138597
3: -11.2413063, -10.5881243, -11.2434940, -10.5855570, -0.2029353, 0.2022524
4: -11.0066566, -10.2138882, -10.9946222, -10.2130060, -0.2284329, 0.2251428
5: -5.0364289, -4.4775410, -5.0382833, -4.4762492, -0.1486380, 0.1488513
6: -3.7308433, -3.1309190, -3.7309008, -3.1356533, -0.1592558, 0.1620015
7: -10.1915569, -9.2900257, -10.1710339, -9.2904005, -0.3480428, 0.3388951
8: -3.1452456, -2.5030975, -3.1467638, -2.5063992, -0.2626435, 0.2640547
9: -2.4637039, -1.7021768, -2.4534116, -1.7026715, -0.2686627, 0.2652501

Time for backsubstitution: 21.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 4617
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1197519, upper bound: 0.1179384
time: 4.30 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1197523, upper bound: 0.1179385
time: 5.52 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: 8.2019653, 8.8090420, 8.2021656, 8.8049335, -0.1806896, 0.1791838
1: -14.5423956, -13.6752081, -14.5419426, -13.6810780, -0.3730712, 0.3800994
2: -4.4597321, -3.7324126, -4.4596038, -3.7353106, -0.3107971, 0.3139260
3: -11.2477808, -10.5855150, -11.2466154, -10.5855465, -0.2019145, 0.2075821
4: -11.0078030, -10.2111063, -10.9946337, -10.2116623, -0.2306013, 0.2247950
5: -5.0404768, -4.4759016, -5.0402355, -4.4762440, -0.1480126, 0.1520091
6: -3.7309747, -3.1307859, -3.7309618, -3.1356416, -0.1592599, 0.1621657
7: -10.1918650, -9.2898607, -10.1710768, -9.2903242, -0.3482878, 0.3389764
8: -3.1492643, -2.5015011, -3.1486425, -2.5063982, -0.2623110, 0.2669693
9: -2.4637671, -1.7019703, -2.4534123, -1.7026349, -0.2686552, 0.2654929

Time for backsubstitution: 21.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4617
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4617

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188276, upper bound: 0.1197525
time: 4.11 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188276, upper bound: 0.1197525
time: 3.33 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 29.52 seconds
NS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 29.52
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1188641
NS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 29.52
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1197522
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 29.52
Output dim: 0, lower bound: -0.1188627, upper bound: 0.1188638
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 29.52
Output dim: 0, lower bound: -0.1188627, upper bound: 0.1197522
NS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 29.52
Output dim: 0, lower bound: -0.1197519, upper bound: 0.1179384
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 29.52
Output dim: 0, lower bound: -0.1197523, upper bound: 0.1179385
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 29.52
Output dim: 0, lower bound: -0.1188276, upper bound: 0.1197525
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 29.52
Output dim: 0, lower bound: -0.1188276, upper bound: 0.1197525

## BFS NS instance: NS_A1_B1_B1

### Backsubstitution after applying NS history:
0: 8.2040844, 8.8029928, 8.2056618, 8.8009033, -0.1746799, 0.1751843
1: -14.5372162, -13.6811762, -14.5350647, -13.6828709, -0.3702159, 0.3697467
2: -4.4584327, -3.7355695, -4.4579716, -3.7358062, -0.3095157, 0.3092062
3: -11.2434616, -10.5860062, -11.2401085, -10.5886078, -0.2019224, 0.2011973
4: -10.9946222, -10.2191820, -10.9934816, -10.2206230, -0.2204416, 0.2206780
5: -5.0381737, -4.4762673, -5.0360765, -4.4779043, -0.1484703, 0.1480449
6: -3.7286427, -3.1356831, -3.7285726, -3.1358030, -0.1579641, 0.1580471
7: -10.1709385, -9.3001013, -10.1706734, -9.3001890, -0.3322463, 0.3320471
8: -3.1443434, -2.5063992, -3.1421995, -2.5079956, -0.2606596, 0.2602668
9: -2.4534116, -1.7075820, -2.4533482, -1.7077494, -0.2608933, 0.2610618

Time for backsubstitution: 21.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4617
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4617

## Relational analysis of NS_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1179399
time: 3.53 seconds

## Relational analysis of NS_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1188641
time: 3.97 seconds

## BFS NS instance: NS_A1_B1_B2

### Backsubstitution after applying NS history:
0: 8.2040844, 8.8029928, 8.2035465, 8.8050175, -0.1777889, 0.1776217
1: -14.5372162, -13.6811762, -14.5384293, -13.6769352, -0.3741307, 0.3737400
2: -4.4584327, -3.7355695, -4.4592161, -3.7328293, -0.3127036, 0.3104608
3: -11.2434616, -10.5860062, -11.2413063, -10.5881243, -0.2024443, 0.2024705
4: -10.9946222, -10.2191820, -11.0066566, -10.2138882, -0.2267425, 0.2221174
5: -5.0381737, -4.4762673, -5.0364289, -4.4775410, -0.1486900, 0.1484580
6: -3.7286427, -3.1356831, -3.7308433, -3.1309190, -0.1597353, 0.1603196
7: -10.1709385, -9.3001013, -10.1915569, -9.2900257, -0.3428874, 0.3382305
8: -3.1443434, -2.5063992, -3.1452456, -2.5030975, -0.2615516, 0.2634425
9: -2.4534116, -1.7075820, -2.4637039, -1.7021768, -0.2669183, 0.2636319

Time for backsubstitution: 21.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4617
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4617

## Relational analysis of NS_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1188283
time: 3.77 seconds

## Relational analysis of NS_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1197522
time: 4.44 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: 8.2040825, 8.8049288, 8.2040825, 8.8049240, -0.1741134, 0.1787115
1: -14.5390377, -13.6811371, -14.5390320, -13.6811390, -0.3737400, 0.3694704
2: -4.4584866, -3.7353890, -4.4584861, -3.7353890, -0.3095803, 0.3099601
3: -11.2465849, -10.5859995, -11.2465820, -10.5859985, -0.2072496, 0.2001762
4: -10.9946289, -10.2178402, -10.9946289, -10.2178411, -0.2200937, 0.2232145
5: -5.0401268, -4.4762640, -5.0401249, -4.4762635, -0.1517881, 0.1474195
6: -3.7287033, -3.1356726, -3.7287037, -3.1356723, -0.1581806, 0.1580511
7: -10.1709776, -9.3000259, -10.1709785, -9.3000269, -0.3323278, 0.3324116
8: -3.1462240, -2.5063982, -3.1462212, -2.5063982, -0.2642511, 0.2599337
9: -2.4534123, -1.7075423, -2.4534123, -1.7075440, -0.2611356, 0.2611256

Time for backsubstitution: 21.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4617
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4617

## Relational analysis of NS_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1179394
time: 4.50 seconds

## Relational analysis of NS_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1188641
time: 3.32 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: 8.2040825, 8.8049288, 8.2019653, 8.8090420, -0.1772090, 0.1811489
1: -14.5390377, -13.6811371, -14.5423956, -13.6752081, -0.3771124, 0.3734627
2: -4.4584866, -3.7353890, -4.4597321, -3.7324126, -0.3127692, 0.3112149
3: -11.2465849, -10.5859995, -11.2477808, -10.5855150, -0.2075698, 0.2014494
4: -10.9946289, -10.2178402, -11.0078030, -10.2111063, -0.2263545, 0.2242857
5: -5.0401268, -4.4762640, -5.0404768, -4.4759016, -0.1518478, 0.1478325
6: -3.7287033, -3.1356726, -3.7309747, -3.1307859, -0.1598995, 0.1603235
7: -10.1709776, -9.3000259, -10.1918650, -9.2898607, -0.3429689, 0.3384759
8: -3.1462240, -2.5063982, -3.1492643, -2.5015011, -0.2644663, 0.2631097
9: -2.4534123, -1.7075423, -2.4637671, -1.7019703, -0.2671609, 0.2636243

Time for backsubstitution: 21.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4617
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4617

## Relational analysis of NS_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1188280
time: 3.86 seconds

## Relational analysis of NS_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1197521
time: 4.74 seconds

## BFS NS instance: NS_A2_A1_B1

### Backsubstitution after applying NS history:
0: 8.2035465, 8.8050175, 8.2040844, 8.8029928, -0.1776217, 0.1777889
1: -14.5384293, -13.6769352, -14.5372162, -13.6811762, -0.3737402, 0.3741310
2: -4.4592161, -3.7328293, -4.4584327, -3.7355695, -0.3104609, 0.3127036
3: -11.2413063, -10.5881243, -11.2434616, -10.5860062, -0.2024705, 0.2024443
4: -11.0066566, -10.2138882, -10.9946222, -10.2191820, -0.2221174, 0.2267425
5: -5.0364289, -4.4775410, -5.0381737, -4.4762673, -0.1484580, 0.1486900
6: -3.7308433, -3.1309190, -3.7286427, -3.1356831, -0.1603196, 0.1597353
7: -10.1915569, -9.2900257, -10.1709385, -9.3001013, -0.3382305, 0.3428874
8: -3.1452456, -2.5030975, -3.1443434, -2.5063992, -0.2634425, 0.2615515
9: -2.4637039, -1.7021768, -2.4534116, -1.7075820, -0.2636319, 0.2669184

Time for backsubstitution: 21.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4617
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 4617

## Relational analysis of NS_A2_A1_B1_B1

### Relational analysis result of NS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188277, upper bound: 0.1179386
time: 3.90 seconds

## Relational analysis of NS_A2_A1_B1_B2

### Relational analysis result of NS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188277, upper bound: 0.1179383
time: 4.19 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: 8.2035465, 8.8050175, 8.2019663, 8.8071079, -0.1771795, 0.1766748
1: -14.5384293, -13.6769352, -14.5405779, -13.6752434, -0.3733606, 0.3738296
2: -4.4592161, -3.7328293, -4.4596782, -3.7325957, -0.3113219, 0.3116324
3: -11.2413063, -10.5881243, -11.2446604, -10.5855227, -0.2019222, 0.2026469
4: -11.0066566, -10.2138882, -11.0077944, -10.2124472, -0.2253813, 0.2251451
5: -5.0364289, -4.4775410, -5.0385261, -4.4759059, -0.1486683, 0.1490937
6: -3.7308433, -3.1309190, -3.7309129, -3.1307991, -0.1595050, 0.1594217
7: -10.1915569, -9.2900257, -10.1918249, -9.2899361, -0.3394120, 0.3396122
8: -3.1452456, -2.5030975, -3.1473875, -2.5015001, -0.2626435, 0.2630353
9: -2.4637039, -1.7021768, -2.4637666, -1.7020066, -0.2654195, 0.2652501

Time for backsubstitution: 21.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4617
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4617

## Relational analysis of NS_A2_A1_B2_B1

### Relational analysis result of NS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188281, upper bound: 0.1179386
time: 3.99 seconds

## Relational analysis of NS_A2_A1_B2_B2

### Relational analysis result of NS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188281, upper bound: 0.1179386
time: 4.01 seconds

## BFS NS instance: NS_A2_A2_B1

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

Time for backsubstitution: 22.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of NS_A2_A2_B1_B1

### Relational analysis result of NS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188274, upper bound: 0.1188623
time: 4.79 seconds

## Relational analysis of NS_A2_A2_B1_B2

### Relational analysis result of NS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188278, upper bound: 0.1188624
time: 5.68 seconds

## BFS NS instance: NS_A2_A2_B2

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

Time for backsubstitution: 22.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of NS_A2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188274, upper bound: 0.1188630
time: 3.39 seconds

## Relational analysis of NS_A2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188278, upper bound: 0.1188630
time: 4.34 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 30.00 seconds
NS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 30.00
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1179399
NS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 30.00
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1188641
NS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 30.00
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1188283
NS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 30.00
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1197522
NS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 30.00
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1179394
NS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 30.00
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1188641
NS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 30.00
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1188280
NS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 30.00
Output dim: 0, lower bound: -0.1179384, upper bound: 0.1197521
NS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 30.00
Output dim: 0, lower bound: -0.1188277, upper bound: 0.1179386
NS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 30.00
Output dim: 0, lower bound: -0.1188277, upper bound: 0.1179383
NS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 30.00
Output dim: 0, lower bound: -0.1188281, upper bound: 0.1179386
NS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 30.00
Output dim: 0, lower bound: -0.1188281, upper bound: 0.1179386
NS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 30.00
Output dim: 0, lower bound: -0.1188274, upper bound: 0.1188623
NS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 30.00
Output dim: 0, lower bound: -0.1188278, upper bound: 0.1188624
NS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 30.00
Output dim: 0, lower bound: -0.1188274, upper bound: 0.1188630
NS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 30.00
Output dim: 0, lower bound: -0.1188278, upper bound: 0.1188630

## BFS NS instance: NS_A1_B1_B1_A1

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

Time for backsubstitution: 21.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A1_B1_B1_A1_A1

### Relational analysis result of NS_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1179309, upper bound: 0.1172346
time: 5.47 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2

### Relational analysis result of NS_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1179154, upper bound: 0.1179162
time: 3.86 seconds

## BFS NS instance: NS_A1_B1_B1_A2

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

Time for backsubstitution: 21.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A1_B1_B1_A2_A1

### Relational analysis result of NS_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1179309, upper bound: 0.1181587
time: 5.43 seconds

## Relational analysis of NS_A1_B1_B1_A2_A2

### Relational analysis result of NS_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1179154, upper bound: 0.1188401
time: 6.32 seconds

## BFS NS instance: NS_A1_B1_B2_A1

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

Time for backsubstitution: 21.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A1_B1_B2_A1_A1

### Relational analysis result of NS_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1179274, upper bound: 0.1181224
time: 5.57 seconds

## Relational analysis of NS_A1_B1_B2_A1_A2

### Relational analysis result of NS_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1179143, upper bound: 0.1188039
time: 5.25 seconds

## BFS NS instance: NS_A1_B1_B2_A2

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

Time for backsubstitution: 21.97 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 57.07 + 554.90 = 611.97 seconds
