## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 5)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.5357107799999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (9.2011337, 10.3698511, 9.2011337, 10.3698511, -0.9421344, 0.9421344)
1: (-16.9597416, -14.8090668, -16.9597416, -14.8090668, -1.2560759, 1.2560759)
2: (-6.5898213, -5.1094699, -6.5898213, -5.1094699, -0.9759345, 0.9759345)
3: (-8.5745993, -6.8540311, -8.5745993, -6.8540311, -1.1960912, 1.1960917)
4: (-10.3516960, -8.8085842, -10.3516960, -8.8085842, -1.2359519, 1.2359519)
5: (-2.0151374, -0.7350942, -2.0151374, -0.7350942, -0.9747214, 0.9747217)
6: (-1.3565187, -0.2398213, -1.3565187, -0.2398213, -0.9679022, 0.9679022)
7: (-8.2114935, -6.5324726, -8.2114935, -6.5324726, -1.4011202, 1.4011202)
8: (-1.7078848, -0.6203923, -1.7078848, -0.6203923, -0.7618291, 0.7618291)
9: (-4.0698876, -2.7048616, -4.0698876, -2.7048616, -0.9506359, 0.9506359)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.01 + 34.68 = 57.69 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.5411211, upper bound: 0.5411220

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 6177
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 145

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 550

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5401553, upper bound: 0.5354497
time: 3.58 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5411190, upper bound: 0.5411189
time: 3.86 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.67 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 7.67
Output dim: 0, lower bound: -0.5401553, upper bound: 0.5354497
NS_A2, status: Status.UNKNOWN, split count: 1, time: 7.67
Output dim: 0, lower bound: -0.5411190, upper bound: 0.5411189

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 9.2092648, 10.3452625, 9.2023716, 10.3594685, -0.9254689, 0.9161775
1: -16.9170837, -14.8196840, -16.9416065, -14.8096676, -1.2072115, 1.2241220
2: -6.5665746, -5.1163144, -6.5799851, -5.1101789, -0.9514923, 0.9591255
3: -8.5675259, -6.8567481, -8.5717602, -6.8548088, -1.1892571, 1.1905193
4: -10.2943478, -8.8216019, -10.3274918, -8.8090744, -1.1782513, 1.1773880
5: -2.0095775, -0.7450522, -2.0136113, -0.7392819, -0.9653506, 0.9634705
6: -1.3524792, -0.2512550, -1.3555877, -0.2445858, -0.9542360, 0.9552722
7: -8.1564426, -6.5472460, -8.1882439, -6.5333900, -1.3450012, 1.3507214
8: -1.7022676, -0.6238775, -1.7056327, -0.6218061, -0.7541053, 0.7561827
9: -4.0390697, -2.7111893, -4.0570097, -2.7054622, -0.9174185, 0.9274561

Time for backsubstitution: 21.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 6177
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 145

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 550

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5354482, upper bound: 0.5354493
time: 5.76 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5354482, upper bound: 0.5354490
time: 3.67 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 9.2011347, 10.3698483, 9.2011328, 10.3698502, -0.9421344, 0.9219093
1: -16.9597397, -14.8090668, -16.9597435, -14.8090658, -1.2214656, 1.2547364
2: -6.5898180, -5.1094675, -6.5898228, -5.1094685, -0.9540238, 0.9759338
3: -8.5746002, -6.8540301, -8.5745983, -6.8540297, -1.1934552, 1.1960902
4: -10.3516932, -8.8085852, -10.3516970, -8.8085852, -1.1849155, 1.2324548
5: -2.0151365, -0.7350948, -2.0151362, -0.7350944, -0.9747219, 0.9669795
6: -1.3565195, -0.2398232, -1.3565195, -0.2398211, -0.9672809, 0.9607716
7: -8.2114897, -6.5324731, -8.2114935, -6.5324736, -1.3472056, 1.3986330
8: -1.7078848, -0.6203938, -1.7078838, -0.6203923, -0.7617133, 0.7655516
9: -4.0698862, -2.7048626, -4.0698867, -2.7048619, -0.9295673, 0.9506359

Time for backsubstitution: 21.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 6177
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 145

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 550

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5354482, upper bound: 0.5401564
time: 5.94 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5354482, upper bound: 0.5411199
time: 3.95 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 31.94 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 31.94
Output dim: 0, lower bound: -0.5354482, upper bound: 0.5354493
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 31.94
Output dim: 0, lower bound: -0.5354482, upper bound: 0.5354490
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 31.94
Output dim: 0, lower bound: -0.5354482, upper bound: 0.5401564
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 31.94
Output dim: 0, lower bound: -0.5354482, upper bound: 0.5411199

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 9.2011347, 10.3698483, 9.2092648, 10.3452625, -0.9174194, 0.9274862
1: -16.9597397, -14.8090668, -16.9170837, -14.8196840, -1.2364898, 1.2069068
2: -6.5898180, -5.1094675, -6.5665746, -5.1163144, -0.9674377, 0.9522524
3: -8.5746002, -6.8540301, -8.5675259, -6.8567481, -1.1933274, 1.1900654
4: -10.3516932, -8.8085852, -10.2943478, -8.8216019, -1.1787663, 1.1752577
5: -2.0151365, -0.7350948, -2.0095775, -0.7450522, -0.9647641, 0.9695399
6: -1.3565195, -0.2398232, -1.3524792, -0.2512550, -0.9557195, 0.9598455
7: -8.2114897, -6.5324731, -8.1564426, -6.5472460, -1.3510861, 1.3435936
8: -1.7078848, -0.6203938, -1.7022676, -0.6238775, -0.7579238, 0.7563949
9: -4.0698862, -2.7048626, -4.0390697, -2.7111893, -0.9309318, 0.9180136

Time for backsubstitution: 22.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 6177
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 145

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 4573

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5354440, upper bound: 0.5390246
time: 4.02 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5354440, upper bound: 0.5401511
time: 4.25 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 9.2011347, 10.3698483, 9.2011347, 10.3698483, -0.9219093, 0.9219089
1: -16.9597397, -14.8090668, -16.9597397, -14.8090668, -1.2214651, 1.2214651
2: -6.5898180, -5.1094675, -6.5898180, -5.1094675, -0.9540234, 0.9540234
3: -8.5746002, -6.8540301, -8.5746002, -6.8540301, -1.1934547, 1.1934547
4: -10.3516932, -8.8085852, -10.3516932, -8.8085852, -1.1849151, 1.1849155
5: -2.0151365, -0.7350948, -2.0151365, -0.7350948, -0.9669795, 0.9669795
6: -1.3565195, -0.2398232, -1.3565195, -0.2398232, -0.9607711, 0.9607711
7: -8.2114897, -6.5324731, -8.2114897, -6.5324731, -1.3472061, 1.3472061
8: -1.7078848, -0.6203938, -1.7078848, -0.6203938, -0.7655501, 0.7655501
9: -4.0698862, -2.7048626, -4.0698862, -2.7048626, -0.9295673, 0.9295673

Time for backsubstitution: 22.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 6177
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 145

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4573

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5354440, upper bound: 0.5399525
time: 5.06 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5354440, upper bound: 0.5411134
time: 4.46 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 31.77 seconds
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 31.77
Output dim: 0, lower bound: -0.5354440, upper bound: 0.5390246
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 31.77
Output dim: 0, lower bound: -0.5354440, upper bound: 0.5401511
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 31.77
Output dim: 0, lower bound: -0.5354440, upper bound: 0.5399525
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 31.77
Output dim: 0, lower bound: -0.5354440, upper bound: 0.5411134

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: 9.2100582, 10.3673267, 9.2112284, 10.3440666, -0.9041204, 0.9194121
1: -16.9486809, -14.8253603, -16.9157333, -14.8273373, -1.2140136, 1.1891098
2: -6.5814667, -5.1257324, -6.5662184, -5.1237173, -0.9483268, 0.9353471
3: -8.5644083, -6.8650541, -8.5659761, -6.8616652, -1.1779757, 1.1774292
4: -10.3467121, -8.8126659, -10.2931099, -8.8223009, -1.1718664, 1.1682355
5: -1.9807135, -0.7540857, -1.9934829, -0.7457699, -0.9294729, 0.9270129
6: -1.3453918, -0.2492967, -1.3475029, -0.2519558, -0.9430866, 0.9449148
7: -8.2025976, -6.5400195, -8.1550016, -6.5504994, -1.3370452, 1.3340688
8: -1.7055998, -0.6247287, -1.7014852, -0.6246624, -0.7539797, 0.7511194
9: -4.0602770, -2.7204604, -4.0386081, -2.7187078, -0.9104185, 0.9021027

Time for backsubstitution: 21.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6177
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 145

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6177

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5341413, upper bound: 0.5387217
time: 5.29 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5354424, upper bound: 0.5390228
time: 4.58 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: 9.2011385, 10.3698444, 9.2092676, 10.3452587, -0.9113832, 0.9272728
1: -16.9597397, -14.8090782, -16.9170818, -14.8196888, -1.2295594, 1.1910200
2: -6.5898194, -5.1094847, -6.5665731, -5.1163216, -0.9608130, 0.9423454
3: -8.5745955, -6.8540440, -8.5675240, -6.8567543, -1.1933203, 1.1827044
4: -10.3516865, -8.8085880, -10.2943439, -8.8216038, -1.1760511, 1.1730194
5: -2.0151048, -0.7350957, -2.0095654, -0.7450511, -0.9431672, 0.9533525
6: -1.3565044, -0.2398252, -1.3524737, -0.2512553, -0.9463620, 0.9598365
7: -8.2114868, -6.5324821, -8.1564426, -6.5472512, -1.3467579, 1.3361712
8: -1.7078810, -0.6203957, -1.7022672, -0.6238775, -0.7575436, 0.7555566
9: -4.0698853, -2.7048800, -4.0390692, -2.7111940, -0.9232991, 0.9015617

Time for backsubstitution: 22.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6177
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 145

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6177

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5341413, upper bound: 0.5351526
time: 7.13 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5354424, upper bound: 0.5401483
time: 8.24 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: 9.2100582, 10.3673267, 9.2030935, 10.3686609, -0.9086099, 0.9138465
1: -16.9486809, -14.8253603, -16.9583817, -14.8167191, -1.2019100, 1.2036920
2: -6.5814667, -5.1257324, -6.5894608, -5.1168671, -0.9379020, 0.9370909
3: -8.5644083, -6.8650541, -8.5730505, -6.8589468, -1.1781049, 1.1808157
4: -10.3467121, -8.8126659, -10.3504524, -8.8092785, -1.1780281, 1.1779332
5: -1.9807135, -0.7540857, -1.9990486, -0.7358145, -0.9316878, 0.9315195
6: -1.3453918, -0.2492967, -1.3515451, -0.2405272, -0.9481106, 0.9458227
7: -8.2025976, -6.5400195, -8.2100458, -6.5357275, -1.3344665, 1.3377047
8: -1.7055998, -0.6247287, -1.7071066, -0.6211834, -0.7616127, 0.7602906
9: -4.0602770, -2.7204604, -4.0694237, -2.7123804, -0.9120793, 0.9136534

Time for backsubstitution: 22.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6177
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 145

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6177

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5352879, upper bound: 0.5396273
time: 3.79 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5366561, upper bound: 0.5399516
time: 3.96 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: 9.2011385, 10.3698444, 9.2011356, 10.3698463, -0.9158731, 0.9264789
1: -16.9597397, -14.8090782, -16.9597397, -14.8090725, -1.2214546, 1.2055788
2: -6.5898194, -5.1094847, -6.5898180, -5.1094761, -0.9540153, 0.9441149
3: -8.5745955, -6.8540440, -8.5745974, -6.8540363, -1.1934476, 1.1860933
4: -10.3516865, -8.8085880, -10.3516903, -8.8085861, -1.1837344, 1.1858015
5: -2.0151048, -0.7350957, -2.0151234, -0.7350955, -0.9453812, 0.9584146
6: -1.3565044, -0.2398252, -1.3565130, -0.2398251, -0.9514141, 0.9607635
7: -8.2114868, -6.5324821, -8.2114887, -6.5324779, -1.3471994, 1.3406887
8: -1.7078810, -0.6203957, -1.7078824, -0.6203947, -0.7651739, 0.7646918
9: -4.0698853, -2.7048800, -4.0698843, -2.7048693, -0.9295623, 0.9131143

Time for backsubstitution: 22.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6177
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 145

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6177

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5352880, upper bound: 0.5407907
time: 4.13 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5366561, upper bound: 0.5411136
time: 4.05 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 30.52 seconds
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.52
Output dim: 0, lower bound: -0.5341413, upper bound: 0.5387217
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.52
Output dim: 0, lower bound: -0.5354424, upper bound: 0.5390228
NS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 30.52
Output dim: 0, lower bound: -0.5341413, upper bound: 0.5351526
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.52
Output dim: 0, lower bound: -0.5354424, upper bound: 0.5401483
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.52
Output dim: 0, lower bound: -0.5352879, upper bound: 0.5396273
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.52
Output dim: 0, lower bound: -0.5366561, upper bound: 0.5399516
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.52
Output dim: 0, lower bound: -0.5352880, upper bound: 0.5407907
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.52
Output dim: 0, lower bound: -0.5366561, upper bound: 0.5411136

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 9.2110748, 10.3662786, 9.2134705, 10.3417664, -0.8998690, 0.9144621
1: -16.9444141, -14.8257484, -16.9063530, -14.8281841, -1.2077150, 1.1789904
2: -6.5801654, -5.1265106, -6.5633559, -5.1254387, -0.9432344, 0.9305708
3: -8.5635843, -6.8666406, -8.5641460, -6.8651519, -1.1733541, 1.1734290
4: -10.3465843, -8.8148432, -10.2928257, -8.8270159, -1.1642408, 1.1626678
5: -1.9797851, -0.7545421, -1.9914447, -0.7468014, -0.9262919, 0.9231653
6: -1.3428456, -0.2494930, -1.3419154, -0.2523878, -0.9359827, 0.9353828
7: -8.2011881, -6.5418377, -8.1519146, -6.5544939, -1.3284693, 1.3251204
8: -1.7029428, -0.6248732, -1.6956425, -0.6249809, -0.7481441, 0.7425854
9: -4.0599937, -2.7209506, -4.0379748, -2.7197769, -0.9089952, 0.9007487

Time for backsubstitution: 22.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 6177
type: A, layer: 1, pos: 145

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5336358, upper bound: 0.5387201
time: 5.25 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5341397, upper bound: 0.5387201
time: 6.54 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 9.2100601, 10.3673229, 9.2069359, 10.3442230, -0.9036455, 0.9195712
1: -16.9486809, -14.8253632, -16.9170189, -14.8141832, -1.2151725, 1.1867678
2: -6.5814619, -5.1257362, -6.5668764, -5.1206946, -0.9479065, 0.9352367
3: -8.5644054, -6.8650594, -8.5717688, -6.8609605, -1.1776695, 1.1829267
4: -10.3467131, -8.8126659, -10.2990913, -8.8214436, -1.1696639, 1.1671920
5: -1.9807127, -0.7540853, -1.9947129, -0.7435070, -0.9322500, 0.9274638
6: -1.3453879, -0.2492974, -1.3482366, -0.2442327, -0.9476027, 0.9454956
7: -8.2025948, -6.5400233, -8.1610146, -6.5495548, -1.3352690, 1.3332033
8: -1.7055960, -0.6247292, -1.7020168, -0.6169477, -0.7595117, 0.7503381
9: -4.0602779, -2.7204623, -4.0397053, -2.7183509, -0.9105134, 0.9030554

Time for backsubstitution: 21.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 6177
type: A, layer: 1, pos: 145

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5349246, upper bound: 0.5390207
time: 5.46 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5354407, upper bound: 0.5390212
time: 3.98 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 9.2011404, 10.3698444, 9.2049694, 10.3454142, -0.9109097, 0.9274151
1: -16.9597359, -14.8090801, -16.9183826, -14.8065338, -1.2307196, 1.1886852
2: -6.5898147, -5.1094837, -6.5672345, -5.1133099, -0.9603796, 0.9422350
3: -8.5745964, -6.8540459, -8.5733166, -6.8560481, -1.1930108, 1.1882000
4: -10.3516874, -8.8085909, -10.3003283, -8.8207445, -1.1738482, 1.1719525
5: -2.0151029, -0.7350975, -2.0107913, -0.7427897, -0.9453094, 0.9537990
6: -1.3565005, -0.2398257, -1.3532059, -0.2435347, -0.9508801, 0.9569137
7: -8.2114859, -6.5324879, -8.1624527, -6.5463061, -1.3449769, 1.3353009
8: -1.7078762, -0.6203952, -1.7027988, -0.6161604, -0.7630665, 0.7547748
9: -4.0698853, -2.7048802, -4.0401664, -2.7108395, -0.9233942, 0.9025130

Time for backsubstitution: 21.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 6177
type: A, layer: 1, pos: 145

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5349245, upper bound: 0.5401477
time: 4.16 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5354406, upper bound: 0.5401465
time: 5.16 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 9.2110748, 10.3662786, 9.2053490, 10.3663578, -0.9043584, 0.9092240
1: -16.9444141, -14.8257484, -16.9490070, -14.8175764, -1.1967106, 1.1935759
2: -6.5801654, -5.1265106, -6.5866079, -5.1185770, -0.9332051, 0.9323072
3: -8.5635843, -6.8666406, -8.5712223, -6.8624277, -1.1734848, 1.1768341
4: -10.3465843, -8.8148432, -10.3501682, -8.8139935, -1.1706271, 1.1730247
5: -1.9797851, -0.7545421, -1.9970125, -0.7368398, -0.9285097, 0.9277737
6: -1.3428456, -0.2494930, -1.3459545, -0.2409604, -0.9410090, 0.9362822
7: -8.2011881, -6.5418377, -8.2069674, -6.5397167, -1.3261747, 1.3294115
8: -1.7029428, -0.6248732, -1.7012706, -0.6215019, -0.7557774, 0.7517564
9: -4.0599937, -2.7209506, -4.0687904, -2.7134535, -0.9106109, 0.9123049

Time for backsubstitution: 21.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 6177
type: A, layer: 1, pos: 145

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5347951, upper bound: 0.5396255
time: 3.79 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5352864, upper bound: 0.5396258
time: 3.83 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 9.2100601, 10.3673229, 9.1988010, 10.3688145, -0.9081364, 0.9164243
1: -16.9486809, -14.8253632, -16.9596748, -14.8035660, -1.2151971, 1.2013323
2: -6.5814619, -5.1257362, -6.5901194, -5.1138406, -0.9394212, 0.9369862
3: -8.5644054, -6.8650594, -8.5788469, -6.8582497, -1.1777887, 1.1863217
4: -10.3467131, -8.8126659, -10.3564415, -8.8084202, -1.1783948, 1.1821604
5: -1.9807127, -0.7540853, -2.0002658, -0.7335502, -0.9344649, 0.9319654
6: -1.3453879, -0.2492974, -1.3522712, -0.2327929, -0.9526343, 0.9464035
7: -8.2025948, -6.5400233, -8.2160616, -6.5347919, -1.3353548, 1.3410969
8: -1.7055960, -0.6247292, -1.7076387, -0.6134591, -0.7671475, 0.7595119
9: -4.0602779, -2.7204623, -4.0705266, -2.7120247, -0.9122052, 0.9146128

Time for backsubstitution: 21.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 6177
type: A, layer: 1, pos: 145

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5361516, upper bound: 0.5399501
time: 3.90 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5366545, upper bound: 0.5399501
time: 4.14 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 9.2021542, 10.3687963, 9.2033863, 10.3675451, -0.9116249, 0.9218574
1: -16.9554710, -14.8094664, -16.9503632, -14.8099232, -1.2162580, 1.1954613
2: -6.5885158, -5.1102629, -6.5869627, -5.1111908, -0.9493079, 0.9393194
3: -8.5737667, -6.8556280, -8.5727654, -6.8575191, -1.1888227, 1.1821156
4: -10.3515625, -8.8107672, -10.3514118, -8.8133011, -1.1763391, 1.1808853
5: -2.0141780, -0.7355533, -2.0130868, -0.7361229, -0.9422054, 0.9545753
6: -1.3539610, -0.2400216, -1.3509259, -0.2402553, -0.9443150, 0.9512239
7: -8.2100916, -6.5343018, -8.2084198, -6.5364671, -1.3389125, 1.3323979
8: -1.7052221, -0.6205406, -1.7020454, -0.6207128, -0.7593377, 0.7561595
9: -4.0696001, -2.7053723, -4.0692534, -2.7059474, -0.9280968, 0.9117684

Time for backsubstitution: 22.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 6177
type: A, layer: 1, pos: 145

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5347950, upper bound: 0.5407885
time: 8.84 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5352863, upper bound: 0.5407892
time: 4.05 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 9.2011404, 10.3698444, 9.1968384, 10.3700027, -0.9154000, 0.9290352
1: -16.9597359, -14.8090801, -16.9610424, -14.7959194, -1.2347407, 1.2032309
2: -6.5898147, -5.1094837, -6.5904799, -5.1064658, -0.9555101, 0.9440105
3: -8.5745964, -6.8540459, -8.5803938, -6.8533421, -1.1931305, 1.1915984
4: -10.3516874, -8.8085909, -10.3576832, -8.8077278, -1.1841021, 1.1899996
5: -2.0151029, -0.7350975, -2.0163341, -0.7328327, -0.9481583, 0.9588513
6: -1.3565005, -0.2398257, -1.3572391, -0.2320934, -0.9559393, 0.9613457
7: -8.2114859, -6.5324879, -8.2174988, -6.5315466, -1.3480906, 1.3440661
8: -1.7078762, -0.6203952, -1.7084136, -0.6126695, -0.7707009, 0.7639129
9: -4.0698853, -2.7048802, -4.0709887, -2.7045145, -0.9296873, 0.9140735

Time for backsubstitution: 22.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 6177
type: A, layer: 1, pos: 145

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5361516, upper bound: 0.5411109
time: 4.49 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5366545, upper bound: 0.5411121
time: 4.18 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 30.90 seconds
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 30.90
Output dim: 0, lower bound: -0.5336358, upper bound: 0.5387201
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 30.90
Output dim: 0, lower bound: -0.5341397, upper bound: 0.5387201
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 30.90
Output dim: 0, lower bound: -0.5349246, upper bound: 0.5390207
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 30.90
Output dim: 0, lower bound: -0.5354407, upper bound: 0.5390212
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 30.90
Output dim: 0, lower bound: -0.5349245, upper bound: 0.5401477
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 30.90
Output dim: 0, lower bound: -0.5354406, upper bound: 0.5401465
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 30.90
Output dim: 0, lower bound: -0.5347951, upper bound: 0.5396255
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 30.90
Output dim: 0, lower bound: -0.5352864, upper bound: 0.5396258
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 30.90
Output dim: 0, lower bound: -0.5361516, upper bound: 0.5399501
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 30.90
Output dim: 0, lower bound: -0.5366545, upper bound: 0.5399501
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 30.90
Output dim: 0, lower bound: -0.5347950, upper bound: 0.5407885
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 30.90
Output dim: 0, lower bound: -0.5352863, upper bound: 0.5407892
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 30.90
Output dim: 0, lower bound: -0.5361516, upper bound: 0.5411109
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 30.90
Output dim: 0, lower bound: -0.5366545, upper bound: 0.5411121

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: 9.2134056, 10.3661966, 9.2143650, 10.3417349, -0.8972898, 0.9132571
1: -16.9417362, -14.8270359, -16.9052773, -14.8286800, -1.1988943, 1.1702785
2: -6.5781078, -5.1305652, -6.5625887, -5.1269908, -0.9401112, 0.9258947
3: -8.5532570, -6.8670092, -8.5601950, -6.8652892, -1.1621213, 1.1671200
4: -10.3423300, -8.8162823, -10.2911987, -8.8275585, -1.1585956, 1.1588969
5: -1.9764067, -0.7555943, -1.9901488, -0.7471931, -0.9224291, 0.9208286
6: -1.3418202, -0.2558140, -1.3415284, -0.2548079, -0.9322066, 0.9278197
7: -8.1985874, -6.5458870, -8.1508989, -6.5560436, -1.3166904, 1.3120379
8: -1.7020459, -0.6258020, -1.6953077, -0.6253357, -0.7469347, 0.7411535
9: -4.0594392, -2.7240236, -4.0377645, -2.7209582, -0.9070182, 0.8971739

Time for backsubstitution: 22.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 145

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 4573

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5325021, upper bound: 0.5387193
time: 6.87 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5325020, upper bound: 0.5387201
time: 4.51 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 9.2066431, 10.3696795, 9.2134781, 10.3417654, -0.9035759, 0.9146030
1: -16.9468575, -14.8234415, -16.9063492, -14.8281898, -1.2084665, 1.1757708
2: -6.5923519, -5.1242924, -6.5633492, -5.1254487, -0.9467986, 0.9312503
3: -8.5694218, -6.8493247, -8.5641060, -6.8651519, -1.1803074, 1.1834884
4: -10.3475103, -8.8045235, -10.2928114, -8.8270206, -1.1643386, 1.1675768
5: -1.9814916, -0.7442572, -1.9914393, -0.7468029, -0.9274559, 0.9266365
6: -1.3543683, -0.2474534, -1.3419114, -0.2524069, -0.9482088, 0.9359479
7: -8.2145138, -6.5402412, -8.1519051, -6.5545034, -1.3371801, 1.3209023
8: -1.7083273, -0.6238360, -1.6956396, -0.6249828, -0.7532763, 0.7435625
9: -4.0660648, -2.7198792, -4.0379720, -2.7197835, -0.9119570, 0.9016795

Time for backsubstitution: 22.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 145

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4573

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5329983, upper bound: 0.5387201
time: 3.78 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5329983, upper bound: 0.5387201
time: 4.33 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 57.69 + 549.11 = 606.80 seconds
