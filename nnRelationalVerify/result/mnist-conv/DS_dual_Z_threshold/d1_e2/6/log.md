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
execution time: IAR + RelationalAnalysis = 22.47 + 32.85 = 55.33 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.1111598, upper bound: 0.1111597

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2229
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 659
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 228

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 2229

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1102022, upper bound: 0.1107718
time: 2.75 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1107719, upper bound: 0.1102021
time: 2.81 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 5.80 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 5.80
Output dim: 9, lower bound: -0.1102022, upper bound: 0.1107718
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 5.80
Output dim: 9, lower bound: -0.1107719, upper bound: 0.1102021

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2583696, 0.2603158
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.2266569, 0.2285351
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1840063, 0.1816190
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1999872, 0.1995194
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2344583, 0.2325598
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1788445, 0.1788460
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.2186443, 0.2188722
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2559587, 0.2559364
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1824465, 0.1828420
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1284004, 0.1285406

Time for backsubstitution: 7.40 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 659
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 228

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 1808

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1098930, upper bound: 0.1100145
time: 3.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1094448, upper bound: 0.1104627
time: 2.94 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2603159, 0.2583696
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.2285352, 0.2266570
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1816190, 0.1840062
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1995194, 0.1999871
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2325598, 0.2344582
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1788461, 0.1788445
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.2188722, 0.2186443
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2559363, 0.2559586
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1828420, 0.1824465
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1285406, 0.1284004

Time for backsubstitution: 7.47 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 659
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 228

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 1808

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1104627, upper bound: 0.1094447
time: 2.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1100145, upper bound: 0.1098929
time: 2.98 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 13.47 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 13.47
Output dim: 9, lower bound: -0.1098930, upper bound: 0.1100145
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 13.47
Output dim: 9, lower bound: -0.1094448, upper bound: 0.1104627
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 13.47
Output dim: 9, lower bound: -0.1104627, upper bound: 0.1094447
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 13.47
Output dim: 9, lower bound: -0.1100145, upper bound: 0.1098929

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2549670, 0.2567637
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.2211561, 0.2231345
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1785790, 0.1759185
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1980942, 0.1976106
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2167536, 0.2155875
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1758634, 0.1745523
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.2107597, 0.2106267
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2519912, 0.2538967
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1792729, 0.1794468
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1238730, 0.1241986

Time for backsubstitution: 7.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 659
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 228

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 1109

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1077667, upper bound: 0.1097453
time: 3.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1095770, upper bound: 0.1070782
time: 2.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2548174, 0.2569133
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.2212563, 0.2230344
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1783057, 0.1761918
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1980783, 0.1976266
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2174858, 0.2148553
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1745507, 0.1758649
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.2103987, 0.2109876
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2539192, 0.2519689
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1790513, 0.1796684
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1240583, 0.1240133

Time for backsubstitution: 7.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 659
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 228

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 1109

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1073185, upper bound: 0.1101936
time: 3.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1091288, upper bound: 0.1075263
time: 3.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2569133, 0.2548175
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.2230344, 0.2212563
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1761918, 0.1783057
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1976266, 0.1980783
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2148553, 0.2174859
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1758650, 0.1745507
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.2109875, 0.2103988
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2519687, 0.2539191
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1796684, 0.1790513
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1240132, 0.1240584

Time for backsubstitution: 8.50 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 659
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 228

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 1109

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1075264, upper bound: 0.1091288
time: 3.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1101936, upper bound: 0.1073185
time: 2.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2567637, 0.2549670
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.2231345, 0.2211561
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1759185, 0.1785790
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1976106, 0.1980943
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2155875, 0.2167537
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1745522, 0.1758634
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.2106267, 0.2107596
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2538968, 0.2519913
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1794468, 0.1792729
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1241986, 0.1238731

Time for backsubstitution: 7.79 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 659
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 228

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 1109

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1070782, upper bound: 0.1095770
time: 2.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1097454, upper bound: 0.1077666
time: 2.93 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 13.76 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 13.76
Output dim: 9, lower bound: -0.1077667, upper bound: 0.1097453
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 13.76
Output dim: 9, lower bound: -0.1095770, upper bound: 0.1070782
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 13.76
Output dim: 9, lower bound: -0.1073185, upper bound: 0.1101936
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 13.76
Output dim: 9, lower bound: -0.1091288, upper bound: 0.1075263
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 13.76
Output dim: 9, lower bound: -0.1075264, upper bound: 0.1091288
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 13.76
Output dim: 9, lower bound: -0.1101936, upper bound: 0.1073185
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 13.76
Output dim: 9, lower bound: -0.1070782, upper bound: 0.1095770
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 13.76
Output dim: 9, lower bound: -0.1097454, upper bound: 0.1077666

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2207567, 0.2270131
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.1657535, 0.1708547
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1753118, 0.1721027
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1681020, 0.1651093
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2107319, 0.2096076
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1711620, 0.1693726
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.1981649, 0.1976049
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2076624, 0.2146982
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1447704, 0.1380037
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1210857, 0.1218668

Time for backsubstitution: 8.76 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 659
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 228

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 1836

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1061084, upper bound: 0.1087753
time: 3.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1068138, upper bound: 0.1081396
time: 3.06 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2252160, 0.2225534
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.1686372, 0.1677318
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1747632, 0.1726512
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1655930, 0.1675863
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2107736, 0.2095658
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1706837, 0.1698509
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.1977379, 0.1980317
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2127925, 0.2095681
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1378298, 0.1448638
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1214736, 0.1214113

Time for backsubstitution: 8.33 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 659
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 228

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 1836

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1079623, upper bound: 0.1060959
time: 3.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1086788, upper bound: 0.1055136
time: 2.96 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2206071, 0.2271627
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.1658536, 0.1707545
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1750385, 0.1723760
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1680860, 0.1651253
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2114641, 0.2088754
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1698493, 0.1706853
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.1978040, 0.1979657
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2095905, 0.2127701
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1445489, 0.1382253
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1212709, 0.1216815

Time for backsubstitution: 8.45 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 659
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 228

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 1836

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1056601, upper bound: 0.1092234
time: 3.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1063656, upper bound: 0.1085878
time: 3.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2250664, 0.2227030
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.1687373, 0.1676316
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1744899, 0.1729245
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1655770, 0.1676023
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2115059, 0.2088335
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1693710, 0.1711636
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.1973770, 0.1983925
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2147203, 0.2076402
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1376082, 0.1450854
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1216589, 0.1212260

Time for backsubstitution: 7.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 659
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 228

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 1836

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1075141, upper bound: 0.1065441
time: 3.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1082307, upper bound: 0.1059618
time: 3.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2227030, 0.2250664
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.1676316, 0.1687373
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1729245, 0.1744899
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1676023, 0.1655770
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2088335, 0.2115059
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1711636, 0.1693710
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.1983925, 0.1973770
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2076400, 0.2147205
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1450853, 0.1376082
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1212259, 0.1216589

Time for backsubstitution: 7.80 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 659
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 228

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 1836

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1059618, upper bound: 0.1082306
time: 3.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1065441, upper bound: 0.1075141
time: 3.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2271627, 0.2206071
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.1707545, 0.1658536
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1723760, 0.1750385
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1651253, 0.1680860
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2088754, 0.2114640
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1706852, 0.1698493
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.1979658, 0.1978040
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2127701, 0.2095906
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1382253, 0.1445489
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1216815, 0.1212710

Time for backsubstitution: 8.40 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 659
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 228

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 1836

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1085878, upper bound: 0.1063656
time: 3.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1092235, upper bound: 0.1056601
time: 3.00 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2225534, 0.2252160
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.1677318, 0.1686371
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1726512, 0.1747631
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1675863, 0.1655930
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2095657, 0.2107736
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1698508, 0.1706837
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.1980317, 0.1977379
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2095681, 0.2127924
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1448638, 0.1378298
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1214113, 0.1214736

Time for backsubstitution: 8.45 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 659
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 228

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 1836

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1055136, upper bound: 0.1086788
time: 3.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1060959, upper bound: 0.1079623
time: 3.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2270131, 0.2207567
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.1708547, 0.1657535
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1721027, 0.1753117
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1651093, 0.1681020
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2096076, 0.2107319
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1693726, 0.1711620
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.1976050, 0.1981649
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2146982, 0.2076625
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1380037, 0.1447704
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1218668, 0.1210857

Time for backsubstitution: 8.48 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 659
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 228

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 1836

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1081396, upper bound: 0.1068137
time: 2.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1087753, upper bound: 0.1061083
time: 3.02 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 14.64 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.64
Output dim: 9, lower bound: -0.1061084, upper bound: 0.1087753
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.64
Output dim: 9, lower bound: -0.1068138, upper bound: 0.1081396
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.64
Output dim: 9, lower bound: -0.1079623, upper bound: 0.1060959
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.64
Output dim: 9, lower bound: -0.1086788, upper bound: 0.1055136
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.64
Output dim: 9, lower bound: -0.1056601, upper bound: 0.1092234
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.64
Output dim: 9, lower bound: -0.1063656, upper bound: 0.1085878
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 14.64
Output dim: 9, lower bound: -0.1075141, upper bound: 0.1065441
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.64
Output dim: 9, lower bound: -0.1082307, upper bound: 0.1059618
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.64
Output dim: 9, lower bound: -0.1059618, upper bound: 0.1082306
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 14.64
Output dim: 9, lower bound: -0.1065441, upper bound: 0.1075141
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.64
Output dim: 9, lower bound: -0.1085878, upper bound: 0.1063656
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.64
Output dim: 9, lower bound: -0.1092235, upper bound: 0.1056601
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.64
Output dim: 9, lower bound: -0.1055136, upper bound: 0.1086788
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.64
Output dim: 9, lower bound: -0.1060959, upper bound: 0.1079623
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.64
Output dim: 9, lower bound: -0.1081396, upper bound: 0.1068137
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.64
Output dim: 9, lower bound: -0.1087753, upper bound: 0.1061083

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2007864, 0.2119587
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.1618650, 0.1671154
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1742565, 0.1711496
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1615268, 0.1589423
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2102938, 0.2089330
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1657755, 0.1630967
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.1799597, 0.1823667
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2039954, 0.2089844
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1446770, 0.1379288
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1183544, 0.1197499

Time for backsubstitution: 8.35 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 659
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 228

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 416

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1046598, upper bound: 0.1077342
time: 3.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1050250, upper bound: 0.1071353
time: 3.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2057700, 0.2070428
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.1620142, 0.1669661
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1743585, 0.1710474
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1619350, 0.1589037
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2100573, 0.2091695
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1648861, 0.1640420
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.1829268, 0.1793996
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2019488, 0.2110311
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1446955, 0.1379103
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1189772, 0.1191355

Time for backsubstitution: 8.41 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 659
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 228

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 416

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1054552, upper bound: 0.1071034
time: 3.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1057243, upper bound: 0.1065226
time: 3.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2052457, 0.2074993
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.1647487, 0.1639926
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1737079, 0.1716981
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1593875, 0.1614193
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2103355, 0.2088912
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1653565, 0.1635750
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.1795326, 0.1827935
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2091252, 0.2038544
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1377363, 0.1447889
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1187423, 0.1192944

Time for backsubstitution: 7.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 659
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 228

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 416

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1060403, upper bound: 0.1051651
time: 3.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1066741, upper bound: 0.1051323
time: 3.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2102295, 0.2025831
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.1648979, 0.1638430
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1738099, 0.1715959
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1594260, 0.1609698
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2100991, 0.2091275
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1644078, 0.1644644
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.1824998, 0.1798264
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2070786, 0.2059011
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1377549, 0.1447703
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1193651, 0.1186799

Time for backsubstitution: 7.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 659
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 228

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 416

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1066042, upper bound: 0.1045140
time: 3.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1073909, upper bound: 0.1044362
time: 3.05 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2006367, 0.2121083
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.1619651, 0.1670153
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1739832, 0.1714228
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1615108, 0.1589583
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2110260, 0.2082008
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1644628, 0.1644093
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.1795988, 0.1827275
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2059235, 0.2070564
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1444554, 0.1381504
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1185397, 0.1195647

Time for backsubstitution: 10.28 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 659
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 228

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 416

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1042116, upper bound: 0.1081824
time: 4.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1045768, upper bound: 0.1075835
time: 3.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2056203, 0.2071923
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.1621144, 0.1668659
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1740853, 0.1713207
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1619190, 0.1589197
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2107896, 0.2084373
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1635734, 0.1653547
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.1825658, 0.1797605
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2038769, 0.2091030
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1444740, 0.1381319
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1191625, 0.1189502

Time for backsubstitution: 7.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 659
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 228

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 416

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1050070, upper bound: 0.1075516
time: 3.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1052762, upper bound: 0.1069708
time: 3.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2100798, 0.2027326
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.1649981, 0.1637428
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1735367, 0.1718692
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1594101, 0.1609858
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2108313, 0.2083954
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1630951, 0.1657770
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.1821389, 0.1801873
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2090067, 0.2039731
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1375333, 0.1449919
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1195505, 0.1184946

Time for backsubstitution: 7.69 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 659
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 228

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 416

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1061560, upper bound: 0.1049622
time: 3.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1069427, upper bound: 0.1048844
time: 3.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2027326, 0.2100798
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.1637428, 0.1649981
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1718692, 0.1735367
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1609859, 0.1594100
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2083955, 0.2108313
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1657771, 0.1630951
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.1801872, 0.1821389
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2039730, 0.2090067
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1449919, 0.1375333
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1184947, 0.1195505

Time for backsubstitution: 7.97 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 659
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 228

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 416

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1048844, upper bound: 0.1069427
time: 3.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1049622, upper bound: 0.1061560
time: 3.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2071923, 0.2056203
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.1668659, 0.1621144
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1713206, 0.1740853
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1589197, 0.1619190
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2084373, 0.2107896
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1653547, 0.1635734
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.1797605, 0.1825659
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2091030, 0.2038769
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1381319, 0.1444740
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1189502, 0.1191625

Time for backsubstitution: 7.97 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 659
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 228

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 416

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1069708, upper bound: 0.1052761
time: 3.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1075516, upper bound: 0.1050069
time: 3.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2121083, 0.2006367
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.1670153, 0.1619652
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1714229, 0.1739832
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1589583, 0.1615108
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2082009, 0.2110261
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1644094, 0.1644628
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.1827276, 0.1795988
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2070564, 0.2059236
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1381504, 0.1444554
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1195646, 0.1185397

Time for backsubstitution: 8.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 659
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 228

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 416

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1075835, upper bound: 0.1045768
time: 3.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1081825, upper bound: 0.1042116
time: 3.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2025831, 0.2102294
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.1638430, 0.1648979
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1715960, 0.1738100
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1609699, 0.1594260
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2091277, 0.2100991
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1644643, 0.1644078
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.1798264, 0.1824997
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2059010, 0.2070786
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1447703, 0.1377549
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1186799, 0.1193652

Time for backsubstitution: 8.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 659
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 228

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 416

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1044362, upper bound: 0.1073909
time: 3.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1045140, upper bound: 0.1066041
time: 3.08 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2074993, 0.2052456
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.1639925, 0.1647487
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1716980, 0.1737078
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1614194, 0.1593874
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2088912, 0.2103356
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1635750, 0.1653565
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.1827935, 0.1795326
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2038545, 0.2091253
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1447889, 0.1377363
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1192943, 0.1187423

Time for backsubstitution: 7.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 659
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 228

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 416

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1051324, upper bound: 0.1066741
time: 3.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1051651, upper bound: 0.1060403
time: 3.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2070428, 0.2057699
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.1669661, 0.1620142
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1710474, 0.1743586
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1589037, 0.1619350
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2091696, 0.2100574
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1640420, 0.1648861
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.1793996, 0.1829267
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2110311, 0.2019488
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1379102, 0.1446955
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1191355, 0.1189772

Time for backsubstitution: 7.90 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 659
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 228

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 416

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1065226, upper bound: 0.1057243
time: 3.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1071034, upper bound: 0.1054552
time: 3.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2119587, 0.2007864
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.1671154, 0.1618650
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1711496, 0.1742564
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1589424, 0.1615268
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2089331, 0.2102938
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1630966, 0.1657755
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.1823667, 0.1799596
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2089845, 0.2039955
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1379288, 0.1446770
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1197500, 0.1183544

Time for backsubstitution: 8.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 659
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 228

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 416

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1071353, upper bound: 0.1050249
time: 3.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1077343, upper bound: 0.1046598
time: 3.03 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 15.07 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 15.07
Output dim: 9, lower bound: -0.1046598, upper bound: 0.1077342
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 15.07
Output dim: 9, lower bound: -0.1050250, upper bound: 0.1071353
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 15.07
Output dim: 9, lower bound: -0.1054552, upper bound: 0.1071034
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 15.07
Output dim: 9, lower bound: -0.1057243, upper bound: 0.1065226
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 15.07
Output dim: 9, lower bound: -0.1060403, upper bound: 0.1051651
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 15.07
Output dim: 9, lower bound: -0.1066741, upper bound: 0.1051323
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 15.07
Output dim: 9, lower bound: -0.1066042, upper bound: 0.1045140
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 15.07
Output dim: 9, lower bound: -0.1073909, upper bound: 0.1044362
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.07
Output dim: 9, lower bound: -0.1042116, upper bound: 0.1081824
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 15.07
Output dim: 9, lower bound: -0.1045768, upper bound: 0.1075835
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 15.07
Output dim: 9, lower bound: -0.1050070, upper bound: 0.1075516
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 15.07
Output dim: 9, lower bound: -0.1052762, upper bound: 0.1069708
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 15.07
Output dim: 9, lower bound: -0.1061560, upper bound: 0.1049622
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 15.07
Output dim: 9, lower bound: -0.1069427, upper bound: 0.1048844
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 15.07
Output dim: 9, lower bound: -0.1048844, upper bound: 0.1069427
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 15.07
Output dim: 9, lower bound: -0.1049622, upper bound: 0.1061560
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 15.07
Output dim: 9, lower bound: -0.1069708, upper bound: 0.1052761
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 15.07
Output dim: 9, lower bound: -0.1075516, upper bound: 0.1050069
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 15.07
Output dim: 9, lower bound: -0.1075835, upper bound: 0.1045768
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.07
Output dim: 9, lower bound: -0.1081825, upper bound: 0.1042116
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 15.07
Output dim: 9, lower bound: -0.1044362, upper bound: 0.1073909
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 15.07
Output dim: 9, lower bound: -0.1045140, upper bound: 0.1066041
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 15.07
Output dim: 9, lower bound: -0.1051324, upper bound: 0.1066741
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 15.07
Output dim: 9, lower bound: -0.1051651, upper bound: 0.1060403
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 15.07
Output dim: 9, lower bound: -0.1065226, upper bound: 0.1057243
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 15.07
Output dim: 9, lower bound: -0.1071034, upper bound: 0.1054552
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 15.07
Output dim: 9, lower bound: -0.1071353, upper bound: 0.1050249
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 15.07
Output dim: 9, lower bound: -0.1077343, upper bound: 0.1046598

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.1976218, 0.2089460
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.1597210, 0.1653044
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1734799, 0.1708199
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1577099, 0.1515227
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2076198, 0.2051420
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1626352, 0.1633514
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.1767874, 0.1795362
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2023183, 0.2040646
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1443303, 0.1379976
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1167118, 0.1180952

Time for backsubstitution: 7.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 659
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 228

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 659

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1037534, upper bound: 0.1075887
time: 3.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1036964, upper bound: 0.1077287
time: 3.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -14.1193066, -13.4014282, -14.1193066, -13.4014282, -0.2121083, 0.1976218
1: -15.8250790, -15.2617521, -15.8250790, -15.2617521, -0.1670153, 0.1597210
2: -8.7469130, -8.2218313, -8.7469130, -8.2218313, -0.1708198, 0.1739832
3: -7.1936679, -6.6119804, -7.1936679, -6.6119804, -0.1515227, 0.1615108
4: -6.7397580, -6.1853905, -6.7397580, -6.1853905, -0.2082009, 0.2076198
5: -1.6634974, -1.0721319, -1.6634974, -1.0721319, -0.1644094, 0.1626352
6: -15.9529762, -15.3196201, -15.9529762, -15.3196201, -0.1827276, 0.1767873
7: -5.4760170, -4.9558496, -5.4760170, -4.9558496, -0.2070564, 0.2023183
8: -2.3089771, -1.8050060, -2.3089771, -1.8050060, -0.1381504, 0.1443303
9: 2.5422225, 2.8984499, 2.5422225, 2.8984499, -0.1180951, 0.1185397

Time for backsubstitution: 8.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 659
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 228

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 659

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1077287, upper bound: 0.1036963
time: 3.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1075887, upper bound: 0.1037534
time: 3.29 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 15.34 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 15.34
Output dim: 9, lower bound: -0.1037534, upper bound: 0.1075887
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 15.34
Output dim: 9, lower bound: -0.1036964, upper bound: 0.1077287
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 15.34
Output dim: 9, lower bound: -0.1077287, upper bound: 0.1036963
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 15.34
Output dim: 9, lower bound: -0.1075887, upper bound: 0.1037534

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 55.33 + 443.87 = 499.19 seconds
