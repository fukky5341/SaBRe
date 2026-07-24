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
execution time: IAR + RelationalAnalysis = 23.21 + 33.64 = 56.85 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.5411211, upper bound: 0.5411220

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 6177
type: A, layer: 1, pos: 6177
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 145
type: A, layer: 1, pos: 145

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 550

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5401553, upper bound: 0.5354497
time: 3.51 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5411190, upper bound: 0.5411189
time: 3.94 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.68 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 7.68
Output dim: 0, lower bound: -0.5401553, upper bound: 0.5354497
NS_A2, status: Status.UNKNOWN, split count: 1, time: 7.68
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

Time for backsubstitution: 22.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 6177
type: A, layer: 1, pos: 6177
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 145
type: B, layer: 1, pos: 145

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 4573

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5401511, upper bound: 0.5343205
time: 5.60 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5401511, upper bound: 0.5354443
time: 5.53 seconds

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

Time for backsubstitution: 22.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 6177
type: B, layer: 1, pos: 6177
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 145
type: B, layer: 1, pos: 145

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 550

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5354482, upper bound: 0.5401564
time: 5.96 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5354482, upper bound: 0.5411199
time: 4.06 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 32.65 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 32.65
Output dim: 0, lower bound: -0.5401511, upper bound: 0.5343205
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 32.65
Output dim: 0, lower bound: -0.5401511, upper bound: 0.5354443
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 32.65
Output dim: 0, lower bound: -0.5354482, upper bound: 0.5401564
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 32.65
Output dim: 0, lower bound: -0.5354482, upper bound: 0.5411199

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: 9.2181988, 10.3427334, 9.2043343, 10.3582783, -0.9121823, 0.9081140
1: -16.9060402, -14.8359776, -16.9402466, -14.8173208, -1.1876216, 1.2063203
2: -6.5582228, -5.1325846, -6.5796251, -5.1175776, -0.9354076, 0.9421892
3: -8.5573225, -6.8677869, -8.5702057, -6.8597250, -1.1738987, 1.1778669
4: -10.2893915, -8.8256941, -10.3262558, -8.8097725, -1.1713796, 1.1703534
5: -1.9751360, -0.7640445, -1.9975215, -0.7400011, -0.9300528, 0.9280100
6: -1.3413604, -0.2607164, -1.3506134, -0.2452878, -0.9416161, 0.9403486
7: -8.1475487, -6.5547996, -8.1868019, -6.5366459, -1.3322616, 1.3411937
8: -1.6999679, -0.6282024, -1.7048554, -0.6225944, -0.7501678, 0.7509084
9: -4.0294743, -2.7267871, -4.0565486, -2.7129793, -0.8999414, 0.9115236

Time for backsubstitution: 22.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 6177
type: A, layer: 1, pos: 6177
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 145
type: B, layer: 1, pos: 145

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 550

## Relational analysis of NS_A1_A1_B1

### Relational analysis result of NS_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5354441, upper bound: 0.5343215
time: 3.96 seconds

## Relational analysis of NS_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5354441, upper bound: 0.5343215
time: 4.61 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: 9.2092695, 10.3452568, 9.2023726, 10.3594666, -0.9187031, 0.9207425
1: -16.9170799, -14.8196983, -16.9416046, -14.8096733, -1.2072010, 1.2082329
2: -6.5665736, -5.1163330, -6.5799861, -5.1101851, -0.9514842, 0.9492183
3: -8.5675240, -6.8567638, -8.5717564, -6.8548155, -1.1892500, 1.1831584
4: -10.2943439, -8.8216038, -10.3274899, -8.8090763, -1.1770535, 1.1751270
5: -2.0095463, -0.7450529, -2.0135992, -0.7392831, -0.9437542, 0.9549339
6: -1.3524647, -0.2512560, -1.3555822, -0.2445858, -0.9448795, 0.9552641
7: -8.1564407, -6.5472555, -8.1882420, -6.5333948, -1.3449960, 1.3432856
8: -1.7022657, -0.6238799, -1.7056322, -0.6218071, -0.7537270, 0.7553365
9: -4.0390687, -2.7112057, -4.0570083, -2.7054684, -0.9174132, 0.9110098

Time for backsubstitution: 22.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 6177
type: A, layer: 1, pos: 6177
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 145
type: B, layer: 1, pos: 145

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 550

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5354440, upper bound: 0.5354454
time: 4.88 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5354440, upper bound: 0.5354454
time: 3.87 seconds

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

Time for backsubstitution: 22.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 6177
type: B, layer: 1, pos: 6177
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 145
type: A, layer: 1, pos: 145

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 4573

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5343200, upper bound: 0.5401512
time: 3.93 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5354439, upper bound: 0.5401512
time: 5.12 seconds

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

Time for backsubstitution: 22.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 6177
type: A, layer: 1, pos: 6177
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 145
type: A, layer: 1, pos: 145

Time for candidate selection: 0.23 seconds

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
time: 4.41 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 31.78 seconds
NS_A1_A1_B1, status: Status.VERIFIED, split count: 3, time: 31.78
Output dim: 0, lower bound: -0.5354441, upper bound: 0.5343215
NS_A1_A1_B2, status: Status.VERIFIED, split count: 3, time: 31.78
Output dim: 0, lower bound: -0.5354441, upper bound: 0.5343215
NS_A1_A2_B1, status: Status.VERIFIED, split count: 3, time: 31.78
Output dim: 0, lower bound: -0.5354440, upper bound: 0.5354454
NS_A1_A2_B2, status: Status.VERIFIED, split count: 3, time: 31.78
Output dim: 0, lower bound: -0.5354440, upper bound: 0.5354454
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 31.78
Output dim: 0, lower bound: -0.5343200, upper bound: 0.5401512
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 31.78
Output dim: 0, lower bound: -0.5354439, upper bound: 0.5401512
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 31.78
Output dim: 0, lower bound: -0.5354440, upper bound: 0.5399525
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 31.78
Output dim: 0, lower bound: -0.5354440, upper bound: 0.5411134

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: 9.2030935, 10.3686609, 9.2181988, 10.3427334, -0.9093599, 0.9141967
1: -16.9583817, -14.8167191, -16.9060402, -14.8359776, -1.2186494, 1.1873169
2: -6.5894608, -5.1168671, -6.5582228, -5.1325846, -0.9504280, 0.9361658
3: -8.5730505, -6.8589468, -8.5573225, -6.8677869, -1.1806741, 1.1747069
4: -10.3504524, -8.8092785, -10.2893915, -8.8256941, -1.1717262, 1.1683733
5: -1.9990486, -0.7358145, -1.9751360, -0.7640445, -0.9280672, 0.9342260
6: -1.3515451, -0.2405272, -1.3413604, -0.2607164, -0.9408021, 0.9472222
7: -8.2100458, -6.5357275, -8.1475487, -6.5547996, -1.3415613, 1.3295441
8: -1.7071066, -0.6211834, -1.6999679, -0.6282024, -0.7526522, 0.7524529
9: -4.0694237, -2.7123804, -4.0294743, -2.7267871, -0.9149573, 0.9005322

Time for backsubstitution: 21.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6177
type: B, layer: 1, pos: 6177
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 145
type: A, layer: 1, pos: 145

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 6177

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5340252, upper bound: 0.5388388
time: 4.36 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5343185, upper bound: 0.5401492
time: 5.11 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: 9.2011356, 10.3698463, 9.2092695, 10.3452568, -0.9219899, 0.9192429
1: -16.9597397, -14.8090725, -16.9170799, -14.8196983, -1.2201118, 1.2068961
2: -6.5898180, -5.1094761, -6.5665736, -5.1163330, -0.9561653, 0.9522445
3: -8.5745974, -6.8540363, -8.5675240, -6.8567638, -1.1859660, 1.1900578
4: -10.3516903, -8.8085861, -10.2943439, -8.8216038, -1.1764884, 1.1725450
5: -2.0151234, -0.7350955, -2.0095463, -0.7450529, -0.9544153, 0.9441693
6: -1.3565130, -0.2398251, -1.3524647, -0.2512560, -0.9557114, 0.9504881
7: -8.2114887, -6.5324779, -8.1564407, -6.5472555, -1.3436518, 1.3392653
8: -1.7078824, -0.6203947, -1.7022657, -0.6238799, -0.7570682, 0.7560201
9: -4.0698843, -2.7048693, -4.0390687, -2.7112057, -0.9137583, 0.9155571

Time for backsubstitution: 21.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6177
type: B, layer: 1, pos: 6177
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 145
type: A, layer: 1, pos: 145

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 6177

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5351527, upper bound: 0.5388388
time: 3.59 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5354425, upper bound: 0.5401492
time: 4.80 seconds

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

Time for backsubstitution: 22.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6177
type: A, layer: 1, pos: 6177
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 145
type: A, layer: 1, pos: 145

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 6177

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5352879, upper bound: 0.5396273
time: 3.80 seconds

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

Time for backsubstitution: 22.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6177
type: A, layer: 1, pos: 6177
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 145
type: B, layer: 1, pos: 145

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 6177

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5352880, upper bound: 0.5407907
time: 4.04 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5366561, upper bound: 0.5411136
time: 4.03 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 30.48 seconds
NS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 30.48
Output dim: 0, lower bound: -0.5340252, upper bound: 0.5388388
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 30.48
Output dim: 0, lower bound: -0.5343185, upper bound: 0.5401492
NS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 30.48
Output dim: 0, lower bound: -0.5351527, upper bound: 0.5388388
NS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 30.48
Output dim: 0, lower bound: -0.5354425, upper bound: 0.5401492
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.48
Output dim: 0, lower bound: -0.5352879, upper bound: 0.5396273
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.48
Output dim: 0, lower bound: -0.5366561, upper bound: 0.5399516
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.48
Output dim: 0, lower bound: -0.5352880, upper bound: 0.5407907
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.48
Output dim: 0, lower bound: -0.5366561, upper bound: 0.5411136

## BFS NS instance: NS_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: 9.2053490, 10.3663578, 9.2192097, 10.3416834, -0.9047370, 0.9098268
1: -16.9490070, -14.8175764, -16.9017696, -14.8363657, -1.2085297, 1.1821017
2: -6.5866079, -5.1185770, -6.5569196, -5.1333632, -0.9454327, 0.9314923
3: -8.5712223, -6.8624277, -8.5564947, -6.8693738, -1.1766925, 1.1700850
4: -10.3501682, -8.8139935, -10.2892618, -8.8278732, -1.1661606, 1.1607456
5: -1.9970125, -0.7368398, -1.9742075, -0.7645016, -0.9242229, 0.9307587
6: -1.3459545, -0.2409604, -1.3388146, -0.2609118, -0.9312539, 0.9401288
7: -8.2069674, -6.5397167, -8.1461382, -6.5566187, -1.3326244, 1.3209591
8: -1.7012706, -0.6215019, -1.6973062, -0.6283474, -0.7441192, 0.7466202
9: -4.0687904, -2.7134535, -4.0291882, -2.7272739, -0.9136648, 0.8990607

Time for backsubstitution: 22.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 6177
type: B, layer: 1, pos: 145
type: A, layer: 1, pos: 145

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4573

## Relational analysis of NS_A2_B1_B1_A1_A1

### Relational analysis result of NS_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5340253, upper bound: 0.5376848
time: 4.36 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2

### Relational analysis result of NS_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5340252, upper bound: 0.5388388
time: 4.39 seconds

## BFS NS instance: NS_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: 9.1988010, 10.3688145, 9.2181997, 10.3427296, -0.9119382, 0.9130108
1: -16.9596748, -14.8035660, -16.9060307, -14.8359766, -1.2163167, 1.2005429
2: -6.5901194, -5.1138406, -6.5582213, -5.1325855, -0.9495382, 0.9376988
3: -8.5788469, -6.8582497, -8.5573215, -6.8677945, -1.1861801, 1.1743913
4: -10.3564415, -8.8084202, -10.2893906, -8.8256960, -1.1706862, 1.1661665
5: -2.0002658, -0.7335502, -1.9751343, -0.7640449, -0.9285154, 0.9343035
6: -1.3522712, -0.2327929, -1.3413566, -0.2607166, -0.9413762, 0.9508481
7: -8.2160616, -6.5347919, -8.1475458, -6.5548062, -1.3407006, 1.3277483
8: -1.7076387, -0.6134591, -1.6999626, -0.6282029, -0.7518737, 0.7579963
9: -4.0705266, -2.7120247, -4.0294724, -2.7267883, -0.9152532, 0.9006565

Time for backsubstitution: 22.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 6177
type: B, layer: 1, pos: 145
type: A, layer: 1, pos: 145

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 4573

## Relational analysis of NS_A2_B1_B1_A2_A1

### Relational analysis result of NS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5343186, upper bound: 0.5390237
time: 6.33 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2

### Relational analysis result of NS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5343185, upper bound: 0.5401481
time: 5.92 seconds

## BFS NS instance: NS_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: 9.2033863, 10.3675451, 9.2102766, 10.3442078, -0.9173670, 0.9148741
1: -16.9503632, -14.8099232, -16.9128151, -14.8200808, -1.2099798, 1.2016811
2: -6.5869627, -5.1111908, -6.5652690, -5.1171107, -0.9511597, 0.9475582
3: -8.5727654, -6.8575191, -8.5666924, -6.8583474, -1.1819863, 1.1854281
4: -10.3514118, -8.8133011, -10.2942162, -8.8237839, -1.1709146, 1.1649199
5: -2.0130868, -0.7361229, -2.0086179, -0.7455111, -0.9505739, 0.9407005
6: -1.3509259, -0.2402553, -1.3499229, -0.2514526, -0.9461646, 0.9433961
7: -8.2084198, -6.5364671, -8.1550417, -6.5490737, -1.3347139, 1.3306780
8: -1.7020454, -0.6207128, -1.6996036, -0.6240249, -0.7485361, 0.7501864
9: -4.0692534, -2.7059474, -4.0387836, -2.7116990, -0.9124677, 0.9141445

Time for backsubstitution: 22.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 6177
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 145
type: A, layer: 1, pos: 145

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 4569

## Relational analysis of NS_A2_B1_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5351511, upper bound: 0.5383376
time: 3.60 seconds

## Relational analysis of NS_A2_B1_B2_A1_B2

### Relational analysis result of NS_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5351511, upper bound: 0.5388371
time: 3.69 seconds

## BFS NS instance: NS_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: 9.1968384, 10.3700027, 9.2092724, 10.3452549, -0.9226608, 0.9180570
1: -16.9610424, -14.7959194, -16.9170761, -14.8197002, -1.2177784, 1.2160985
2: -6.5904799, -5.1064658, -6.5665708, -5.1163306, -0.9552758, 0.9537508
3: -8.5803938, -6.8533421, -8.5675230, -6.8567662, -1.1914701, 1.1897411
4: -10.3576832, -8.8077278, -10.2943449, -8.8216057, -1.1754274, 1.1703382
5: -2.0163341, -0.7328327, -2.0095456, -0.7450538, -0.9548516, 0.9442461
6: -1.3572391, -0.2320934, -1.3524611, -0.2512559, -0.9562864, 0.9531753
7: -8.2174988, -6.5315466, -8.1564388, -6.5472622, -1.3427896, 1.3374605
8: -1.7084136, -0.6126695, -1.7022624, -0.6238790, -0.7562888, 0.7615528
9: -4.0709887, -2.7045145, -4.0390692, -2.7112067, -0.9140551, 0.9156520

Time for backsubstitution: 22.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 6177
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 145
type: A, layer: 1, pos: 145

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 4569

## Relational analysis of NS_A2_B1_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5354408, upper bound: 0.5396410
time: 4.53 seconds

## Relational analysis of NS_A2_B1_B2_A2_B2

### Relational analysis result of NS_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5354408, upper bound: 0.5401475
time: 6.37 seconds

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

Time for backsubstitution: 22.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 6177
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 145
type: A, layer: 1, pos: 145

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 4573

## Relational analysis of NS_A2_B2_A1_B1_B1

### Relational analysis result of NS_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5341262, upper bound: 0.5396262
time: 4.50 seconds

## Relational analysis of NS_A2_B2_A1_B1_B2

### Relational analysis result of NS_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5341262, upper bound: 0.5396264
time: 4.57 seconds

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

Time for backsubstitution: 22.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 6177
type: B, layer: 1, pos: 145
type: A, layer: 1, pos: 145

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 4573

## Relational analysis of NS_A2_B2_A1_B2_B1

### Relational analysis result of NS_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5354990, upper bound: 0.5399506
time: 4.21 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2

### Relational analysis result of NS_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5354989, upper bound: 0.5399506
time: 6.29 seconds

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

Time for backsubstitution: 22.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 6177
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 145
type: B, layer: 1, pos: 145

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 4569

## Relational analysis of NS_A2_B2_A2_B1_B1

### Relational analysis result of NS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5352864, upper bound: 0.5402979
time: 4.21 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2

### Relational analysis result of NS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5352864, upper bound: 0.5407891
time: 4.16 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.85 + 547.90 = 604.75 seconds
