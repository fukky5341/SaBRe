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
execution time: IAR + RelationalAnalysis = 23.43 + 33.07 = 56.50 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.1111598, upper bound: 0.1111597

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 659
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 2229
type: DSZ, layer: 3, pos: 1808

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 659

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1107061, upper bound: 0.1105784
time: 3.06 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1105785, upper bound: 0.1107061
time: 2.88 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 5.95 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 5.95
Output dim: 9, lower bound: -0.1107061, upper bound: 0.1105784
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 5.95
Output dim: 9, lower bound: -0.1105785, upper bound: 0.1107061

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2635510, 0.2618908
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.2341809, 0.2321763
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1980840, 0.1979194
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.2000859, 0.1994388
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2447162, 0.2453704
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1781108, 0.1790777
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.2205019, 0.2198699
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2612212, 0.2635050
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1837651, 0.1826979
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1287994, 0.1287537

Time for backsubstitution: 8.73 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 2229

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 228

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1096828, upper bound: 0.1101004
time: 3.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1102280, upper bound: 0.1095551
time: 3.15 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2641456, 0.2635511
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.2365746, 0.2341809
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1989318, 0.1980840
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.2008582, 0.2000859
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2453704, 0.2459674
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1788921, 0.1781108
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.2207536, 0.2205019
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2624643, 0.2612214
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1826979, 0.1839114
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1287537, 0.1294789

Time for backsubstitution: 8.74 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 2229
type: DSZ, layer: 3, pos: 1195

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1836

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1089939, upper bound: 0.1097428
time: 3.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1096028, upper bound: 0.1090843
time: 3.16 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 15.06 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 15.06
Output dim: 9, lower bound: -0.1096828, upper bound: 0.1101004
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 15.06
Output dim: 9, lower bound: -0.1102280, upper bound: 0.1095551
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 15.06
Output dim: 9, lower bound: -0.1089939, upper bound: 0.1097428
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 15.06
Output dim: 9, lower bound: -0.1096028, upper bound: 0.1090843

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2539060, 0.2518156
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.2330289, 0.2309284
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1963868, 0.1965431
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1963754, 0.1974491
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2381380, 0.2363586
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1780475, 0.1790319
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.2140297, 0.2150039
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2606556, 0.2626016
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1787701, 0.1791281
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1281110, 0.1282614

Time for backsubstitution: 8.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2229
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1836

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2229

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1086889, upper bound: 0.1097125
time: 3.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1093153, upper bound: 0.1091426
time: 3.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2534759, 0.2522458
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.2329330, 0.2310245
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1967077, 0.1962222
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1980963, 0.1957281
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2357042, 0.2387924
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1780649, 0.1790144
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.2156360, 0.2133976
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2603180, 0.2629392
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1801952, 0.1777029
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1283071, 0.1280653

Time for backsubstitution: 8.22 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 2229

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1109

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1077246, upper bound: 0.1092846
time: 3.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1099589, upper bound: 0.1070086
time: 3.12 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2479784, 0.2511972
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.2324829, 0.2302382
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1976737, 0.1963855
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1958424, 0.1951088
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2448364, 0.2454209
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1743403, 0.1726136
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.2028116, 0.2050242
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2594444, 0.2561545
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1825930, 0.1838251
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1261064, 0.1274544

Time for backsubstitution: 8.40 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 2229

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 228

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1079706, upper bound: 0.1092647
time: 5.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1085158, upper bound: 0.1087195
time: 3.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2529619, 0.2462137
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.2326322, 0.2300889
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1977757, 0.1962833
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1958811, 0.1950702
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2445999, 0.2456572
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1733949, 0.1735590
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.2057787, 0.2020570
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2573975, 0.2582011
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1826116, 0.1838067
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1267292, 0.1268317

Time for backsubstitution: 8.42 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 2229
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 1109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 416

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1079311, upper bound: 0.1080517
time: 3.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1085618, upper bound: 0.1074743
time: 2.92 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 14.44 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 14.44
Output dim: 9, lower bound: -0.1086889, upper bound: 0.1097125
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 14.44
Output dim: 9, lower bound: -0.1093153, upper bound: 0.1091426
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 14.44
Output dim: 9, lower bound: -0.1077246, upper bound: 0.1092846
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 14.44
Output dim: 9, lower bound: -0.1099589, upper bound: 0.1070086
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 14.44
Output dim: 9, lower bound: -0.1079706, upper bound: 0.1092647
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 14.44
Output dim: 9, lower bound: -0.1085158, upper bound: 0.1087195
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 14.44
Output dim: 9, lower bound: -0.1079311, upper bound: 0.1080517
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 14.44
Output dim: 9, lower bound: -0.1085618, upper bound: 0.1074743

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2475747, 0.2474214
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.2227049, 0.2224824
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1811409, 0.1787508
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1955038, 0.1961099
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2251192, 0.2214416
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1779978, 0.1789837
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.2115409, 0.2127554
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2539581, 0.2558403
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1773111, 0.1780646
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1269869, 0.1272811

Time for backsubstitution: 8.44 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1836

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1109

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1065959, upper bound: 0.1094434
time: 3.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1083730, upper bound: 0.1067843
time: 3.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2495210, 0.2454842
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.2245829, 0.2206042
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1787536, 0.1812972
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1950362, 0.1965776
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2232209, 0.2233400
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1779993, 0.1789821
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.2117689, 0.2125152
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2539357, 0.2559040
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1777991, 0.1776690
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1271272, 0.1271374

Time for backsubstitution: 8.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 1109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1195

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1085756, upper bound: 0.1086059
time: 3.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1087786, upper bound: 0.1084029
time: 3.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2223777, 0.2256071
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.1886470, 0.1896223
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1952299, 0.1940536
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1702040, 0.1653589
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2319473, 0.2350881
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1733956, 0.1738668
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.2059376, 0.2030398
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2221297, 0.2298810
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1487126, 0.1392797
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1257732, 0.1259192

Time for backsubstitution: 8.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2229
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1836

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2229

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1071693, upper bound: 0.1089628
time: 3.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1069120, upper bound: 0.1082449
time: 3.05 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2264614, 0.2211474
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.1911600, 0.1867386
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1945391, 0.1946021
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1677270, 0.1677781
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2319893, 0.2350352
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1729172, 0.1743451
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.2052782, 0.2034668
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2273105, 0.2247512
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1417720, 0.1462039
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1261397, 0.1255313

Time for backsubstitution: 8.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2229
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1195

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2229

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1089545, upper bound: 0.1062806
time: 3.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1095710, upper bound: 0.1064682
time: 3.10 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2383330, 0.2411216
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.2313311, 0.2289903
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1959767, 0.1950094
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1921317, 0.1931190
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2382584, 0.2364089
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1742769, 0.1725677
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.1963390, 0.2001579
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2588784, 0.2552512
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1775980, 0.1802553
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1254181, 0.1269622

Time for backsubstitution: 8.42 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 2229
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 416

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1195

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1072309, upper bound: 0.1087280
time: 2.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1074339, upper bound: 0.1085250
time: 3.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2379029, 0.2415518
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.2312350, 0.2290862
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1962976, 0.1946884
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1938527, 0.1913981
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2358246, 0.2388427
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1742944, 0.1725503
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.1979455, 0.1985516
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2585407, 0.2555890
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1790231, 0.1788301
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1256142, 0.1267660

Time for backsubstitution: 8.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 2229
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1808

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1195

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1077761, upper bound: 0.1081828
time: 3.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1079792, upper bound: 0.1079797
time: 3.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2506235, 0.2437266
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.2312686, 0.2288731
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1973961, 0.1958039
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1934665, 0.1898975
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2415131, 0.2429504
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1724782, 0.1736400
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.2041564, 0.2000082
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2547046, 0.2562466
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1825603, 0.1837277
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1255719, 0.1260327

Time for backsubstitution: 8.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 2229
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 1195

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1109

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1057675, upper bound: 0.1077826
time: 4.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1076621, upper bound: 0.1054149
time: 3.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2529619, 0.2438753
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.2326322, 0.2287253
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1972963, 0.1962833
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1907084, 0.1950702
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2445999, 0.2425704
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1733949, 0.1726422
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.2057787, 0.2004347
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2573975, 0.2555082
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1826116, 0.1837554
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1259303, 0.1268317

Time for backsubstitution: 8.80 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2229
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 1195

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2229

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1074155, upper bound: 0.1071020
time: 3.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1081671, upper bound: 0.1062286
time: 2.96 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 14.84 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 9, lower bound: -0.1065959, upper bound: 0.1094434
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 9, lower bound: -0.1083730, upper bound: 0.1067843
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 9, lower bound: -0.1085756, upper bound: 0.1086059
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 9, lower bound: -0.1087786, upper bound: 0.1084029
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 9, lower bound: -0.1071693, upper bound: 0.1089628
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 9, lower bound: -0.1069120, upper bound: 0.1082449
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 9, lower bound: -0.1089545, upper bound: 0.1062806
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 9, lower bound: -0.1095710, upper bound: 0.1064682
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 9, lower bound: -0.1072309, upper bound: 0.1087280
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 9, lower bound: -0.1074339, upper bound: 0.1085250
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 9, lower bound: -0.1077761, upper bound: 0.1081828
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 9, lower bound: -0.1079792, upper bound: 0.1079797
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 14.84
Output dim: 9, lower bound: -0.1057675, upper bound: 0.1077826
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 14.84
Output dim: 9, lower bound: -0.1076621, upper bound: 0.1054149
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 14.84
Output dim: 9, lower bound: -0.1074155, upper bound: 0.1071020
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 9, lower bound: -0.1081671, upper bound: 0.1062286

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2097921, 0.2137229
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.1662265, 0.1687545
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1783624, 0.1754237
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1649657, 0.1631206
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2189261, 0.2152905
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1733048, 0.1738125
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.1991895, 0.1999771
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2096306, 0.2166429
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1416881, 0.1355175
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1242144, 0.1249129

Time for backsubstitution: 8.42 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 416

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1836

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1049376, upper bound: 0.1084609
time: 3.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1056430, upper bound: 0.1078749
time: 3.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2142514, 0.2096387
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.1691102, 0.1660043
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1778138, 0.1761144
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1625145, 0.1655976
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2191120, 0.2152486
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1728265, 0.1742907
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.1987625, 0.2006363
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2147604, 0.2115128
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1347640, 0.1423776
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1246023, 0.1245086

Time for backsubstitution: 8.49 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1195

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1836

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1067582, upper bound: 0.1057894
time: 3.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1074748, upper bound: 0.1052571
time: 3.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2484267, 0.2436242
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.2222109, 0.2185295
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1762264, 0.1793613
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1911710, 0.1909267
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2220243, 0.2210453
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1683930, 0.1715819
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.2089767, 0.2105283
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2501745, 0.2500548
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1771952, 0.1771634
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1262197, 0.1264897

Time for backsubstitution: 9.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1836

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1808

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1082663, upper bound: 0.1078485
time: 3.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1078181, upper bound: 0.1082967
time: 3.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2476610, 0.2443899
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.2225080, 0.2182323
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1768178, 0.1787699
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1893852, 0.1927124
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2209261, 0.2221434
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1705990, 0.1693759
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.2097819, 0.2097231
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2480869, 0.2521427
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1772934, 0.1770651
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1264795, 0.1262299

Time for backsubstitution: 8.53 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1808

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1084694, upper bound: 0.1076455
time: 3.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1080212, upper bound: 0.1080937
time: 3.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2093619, 0.2145381
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.1661305, 0.1692231
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1788255, 0.1752620
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1667445, 0.1613996
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2164925, 0.2177348
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1733223, 0.1737950
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.2010282, 0.1983584
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2092930, 0.2170217
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1431298, 0.1341849
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1244105, 0.1247645

Time for backsubstitution: 8.42 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 1808

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1195

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1064296, upper bound: 0.1084261
time: 2.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1066327, upper bound: 0.1082231
time: 3.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2112989, 0.2125914
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.1680086, 0.1671057
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1762791, 0.1776491
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1662448, 0.1618673
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2145942, 0.2197664
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1733238, 0.1737935
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.2012685, 0.1981304
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2092294, 0.2170441
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1434447, 0.1336969
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1245543, 0.1245566

Time for backsubstitution: 8.58 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 1808

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 416

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1059190, upper bound: 0.1070057
time: 4.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1059634, upper bound: 0.1067194
time: 3.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2134458, 0.2100782
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.1686434, 0.1661003
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1781348, 0.1758105
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1642355, 0.1638189
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2165345, 0.2176821
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1728440, 0.1742733
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.2003689, 0.1987851
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2144738, 0.2118917
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1361891, 0.1410284
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1247771, 0.1243089

Time for backsubstitution: 8.39 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 1808

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 416

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1068839, upper bound: 0.1054109
time: 3.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1076715, upper bound: 0.1054274
time: 3.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2153829, 0.2081317
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.1707590, 0.1642220
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1755884, 0.1781977
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1637678, 0.1643185
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2146362, 0.2195804
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1728455, 0.1742718
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.2006091, 0.1985574
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2143594, 0.2119142
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1365847, 0.1406210
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1249585, 0.1241687

Time for backsubstitution: 8.47 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 1195

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1836

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1079653, upper bound: 0.1055028
time: 3.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1086009, upper bound: 0.1048473
time: 3.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2372388, 0.2392616
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.2289586, 0.2269151
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1934495, 0.1930735
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1882665, 0.1874681
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2370620, 0.2341146
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1646707, 0.1651675
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.1935471, 0.1981712
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2551171, 0.2494023
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1769940, 0.1797496
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1245106, 0.1263146

Time for backsubstitution: 8.47 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 2229

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 416

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1060869, upper bound: 0.1076870
time: 3.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1063120, upper bound: 0.1068925
time: 3.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2364732, 0.2400273
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.2292559, 0.2266181
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1940408, 0.1924821
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1864808, 0.1892539
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2359639, 0.2352126
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1668767, 0.1629615
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.1943523, 0.1973660
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2530295, 0.2514899
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1770922, 0.1796514
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1247704, 0.1260548

Time for backsubstitution: 9.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 2229
type: DSZ, layer: 3, pos: 416

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1109

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1048314, upper bound: 0.1082559
time: 3.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1071633, upper bound: 0.1060387
time: 3.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2368087, 0.2396919
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.2288628, 0.2270112
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1937704, 0.1927525
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1899874, 0.1857471
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2346283, 0.2365482
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1646881, 0.1651500
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.1951534, 0.1965649
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2547795, 0.2497401
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1784191, 0.1783245
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1247067, 0.1261184

Time for backsubstitution: 8.44 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2229
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2229

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1068068, upper bound: 0.1078085
time: 3.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1074044, upper bound: 0.1072473
time: 3.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2360431, 0.2404575
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.2291598, 0.2267139
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1943617, 0.1921611
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1882017, 0.1875328
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2335303, 0.2376463
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1668942, 0.1629440
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.1959586, 0.1957597
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2526919, 0.2518277
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1785173, 0.1782261
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1249666, 0.1258587

Time for backsubstitution: 8.23 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 2229
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1808

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 416

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1061990, upper bound: 0.1070533
time: 3.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1069457, upper bound: 0.1067804
time: 2.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2491792, 0.2382218
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.2247081, 0.2187011
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1798795, 0.1813302
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1891536, 0.1944624
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2310978, 0.2303033
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1733702, 0.1726204
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.2034314, 0.1978858
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2504463, 0.2482371
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1815368, 0.1822863
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1251786, 0.1259524

Time for backsubstitution: 8.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 1109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1808

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1078578, upper bound: 0.1054711
time: 2.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1074096, upper bound: 0.1059193
time: 3.11 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 14.56 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.56
Output dim: 9, lower bound: -0.1049376, upper bound: 0.1084609
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.56
Output dim: 9, lower bound: -0.1056430, upper bound: 0.1078749
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 9, lower bound: -0.1067582, upper bound: 0.1057894
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 9, lower bound: -0.1074748, upper bound: 0.1052571
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.56
Output dim: 9, lower bound: -0.1082663, upper bound: 0.1078485
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.56
Output dim: 9, lower bound: -0.1078181, upper bound: 0.1082967
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.56
Output dim: 9, lower bound: -0.1084694, upper bound: 0.1076455
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.56
Output dim: 9, lower bound: -0.1080212, upper bound: 0.1080937
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.56
Output dim: 9, lower bound: -0.1064296, upper bound: 0.1084261
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.56
Output dim: 9, lower bound: -0.1066327, upper bound: 0.1082231
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 9, lower bound: -0.1059190, upper bound: 0.1070057
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 9, lower bound: -0.1059634, upper bound: 0.1067194
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 9, lower bound: -0.1068839, upper bound: 0.1054109
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 9, lower bound: -0.1076715, upper bound: 0.1054274
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.56
Output dim: 9, lower bound: -0.1079653, upper bound: 0.1055028
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.56
Output dim: 9, lower bound: -0.1086009, upper bound: 0.1048473
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 9, lower bound: -0.1060869, upper bound: 0.1076870
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 9, lower bound: -0.1063120, upper bound: 0.1068925
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.56
Output dim: 9, lower bound: -0.1048314, upper bound: 0.1082559
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 9, lower bound: -0.1071633, upper bound: 0.1060387
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 9, lower bound: -0.1068068, upper bound: 0.1078085
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 9, lower bound: -0.1074044, upper bound: 0.1072473
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 9, lower bound: -0.1061990, upper bound: 0.1070533
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 9, lower bound: -0.1069457, upper bound: 0.1067804
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.56
Output dim: 9, lower bound: -0.1078578, upper bound: 0.1054711
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 9, lower bound: -0.1074096, upper bound: 0.1059193

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.1886511, 0.1973073
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.1623381, 0.1650154
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1767651, 0.1736849
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1583906, 0.1569536
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2182714, 0.2143914
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1679182, 0.1675365
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.1804810, 0.1841708
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2059634, 0.2109289
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1415947, 0.1354426
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1214831, 0.1227960

Time for backsubstitution: 8.47 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1195

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1808

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1046283, upper bound: 0.1077035
time: 3.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1041802, upper bound: 0.1081517
time: 3.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.1936347, 0.1925820
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.1624873, 0.1651532
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1768672, 0.1738264
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1587987, 0.1569151
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2180268, 0.2146279
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1670288, 0.1684818
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.1834481, 0.1812685
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2039168, 0.2130175
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1416118, 0.1354240
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1221060, 0.1221816

Time for backsubstitution: 8.23 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 1195

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 416

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1046197, upper bound: 0.1068379
time: 3.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1046809, upper bound: 0.1060521
time: 3.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2450238, 0.2400718
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.2167099, 0.2131286
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1707993, 0.1736609
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1892780, 0.1890177
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2043194, 0.2040726
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1654119, 0.1672882
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.2010916, 0.2022822
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2462070, 0.2480159
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1740217, 0.1737683
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1216923, 0.1221476

Time for backsubstitution: 8.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1836

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1109

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1053594, upper bound: 0.1075323
time: 3.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1080415, upper bound: 0.1057475
time: 3.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2448742, 0.2402214
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.2168100, 0.2130284
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1705261, 0.1739342
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1892620, 0.1890337
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2050515, 0.2033403
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1640992, 0.1686009
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.2007307, 0.2026431
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2481351, 0.2460876
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1738001, 0.1739899
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1218776, 0.1219623

Time for backsubstitution: 7.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 416

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1109

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1049112, upper bound: 0.1079805
time: 3.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1075933, upper bound: 0.1061957
time: 3.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2442582, 0.2408375
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.2170072, 0.2128314
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1713906, 0.1730695
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1874923, 0.1908034
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2032212, 0.2051706
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1676179, 0.1650821
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.2018968, 0.2014771
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2441194, 0.2501035
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1741199, 0.1736701
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1219520, 0.1218878

Time for backsubstitution: 8.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1836

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1109

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1055624, upper bound: 0.1073293
time: 3.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1082445, upper bound: 0.1055445
time: 3.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2441086, 0.2409869
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.2171073, 0.2127312
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1711174, 0.1733428
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1874763, 0.1908194
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2039534, 0.2044384
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1663053, 0.1663948
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.2015359, 0.2018379
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2460475, 0.2481754
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1738983, 0.1738917
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1221373, 0.1217026

Time for backsubstitution: 8.90 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1836

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 416

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1068147, upper bound: 0.1068125
time: 3.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1070868, upper bound: 0.1060898
time: 3.10 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2082677, 0.2126781
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.1637585, 0.1671484
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1762982, 0.1733261
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1628792, 0.1557487
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2152957, 0.2154399
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1637160, 0.1663948
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.1982358, 0.1963710
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2055310, 0.2111721
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1425259, 0.1336793
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1235030, 0.1241167

Time for backsubstitution: 8.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 416

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1808

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1061204, upper bound: 0.1076687
time: 3.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1056722, upper bound: 0.1081169
time: 3.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2075020, 0.2134438
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.1640557, 0.1668512
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1768896, 0.1727348
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1610935, 0.1575344
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2141975, 0.2165380
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1659220, 0.1641888
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.1990409, 0.1955658
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2034434, 0.2132597
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1426242, 0.1335810
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1237628, 0.1238570

Time for backsubstitution: 8.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1808

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1836

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1049745, upper bound: 0.1072406
time: 3.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1056797, upper bound: 0.1066546
time: 3.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.1942420, 0.1917840
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.1668705, 0.1604829
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1739911, 0.1764589
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1575622, 0.1581515
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2139813, 0.2186813
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1675149, 0.1679958
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.1819006, 0.1827511
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2106920, 0.2062004
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1364912, 0.1405461
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1222273, 0.1220602

Time for backsubstitution: 8.40 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1808

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1195

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1072256, upper bound: 0.1049661
time: 3.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1074286, upper bound: 0.1047631
time: 3.28 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 14.96 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 14.96
Output dim: 9, lower bound: -0.1046283, upper bound: 0.1077035
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 14.96
Output dim: 9, lower bound: -0.1041802, upper bound: 0.1081517
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 14.96
Output dim: 9, lower bound: -0.1046197, upper bound: 0.1068379
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 14.96
Output dim: 9, lower bound: -0.1046809, upper bound: 0.1060521
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 14.96
Output dim: 9, lower bound: -0.1053594, upper bound: 0.1075323
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 14.96
Output dim: 9, lower bound: -0.1080415, upper bound: 0.1057475
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 14.96
Output dim: 9, lower bound: -0.1049112, upper bound: 0.1079805
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 14.96
Output dim: 9, lower bound: -0.1075933, upper bound: 0.1061957
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 14.96
Output dim: 9, lower bound: -0.1055624, upper bound: 0.1073293
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 14.96
Output dim: 9, lower bound: -0.1082445, upper bound: 0.1055445
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 14.96
Output dim: 9, lower bound: -0.1068147, upper bound: 0.1068125
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 14.96
Output dim: 9, lower bound: -0.1070868, upper bound: 0.1060898
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 14.96
Output dim: 9, lower bound: -0.1061204, upper bound: 0.1076687
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 14.96
Output dim: 9, lower bound: -0.1056722, upper bound: 0.1081169
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 14.96
Output dim: 9, lower bound: -0.1049745, upper bound: 0.1072406
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 14.96
Output dim: 9, lower bound: -0.1056797, upper bound: 0.1066546
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 14.96
Output dim: 9, lower bound: -0.1072256, upper bound: 0.1049661
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 14.96
Output dim: 9, lower bound: -0.1074286, upper bound: 0.1047631
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.96
Output dim: 9, lower bound: -0.1086009, upper bound: 0.1048473
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.96
Output dim: 9, lower bound: -0.1048314, upper bound: 0.1082559
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.96
Output dim: 9, lower bound: -0.1078578, upper bound: 0.1054711

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 56.50 + 549.52 = 606.02 seconds
