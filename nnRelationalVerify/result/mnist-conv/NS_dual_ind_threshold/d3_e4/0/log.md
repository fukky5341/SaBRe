## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.6088244805


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-13.1174488, -10.4751902, -13.1174488, -10.4751902, -1.8021035, 1.8021038)
1: (-7.1292858, -4.1849318, -7.1292858, -4.1849318, -2.3330483, 2.3330483)
2: (9.3677406, 11.2813492, 9.3677406, 11.2813492, -1.5663891, 1.5663891)
3: (-4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9216013, 1.9216008)
4: (-9.4387360, -6.7248478, -9.4387360, -6.7248478, -1.9660249, 1.9660249)
5: (-13.7978468, -11.1748791, -13.7978468, -11.1748791, -1.6303473, 1.6303473)
6: (-16.3375626, -12.7550831, -16.3375626, -12.7550831, -2.2865324, 2.2865324)
7: (-4.0563126, -1.3696806, -4.0563126, -1.3696806, -2.4632163, 2.4632158)
8: (-6.0375504, -3.6194944, -6.0375504, -3.6194944, -2.0286188, 2.0286188)
9: (-11.8428965, -9.3279085, -11.8428965, -9.3279085, -1.7399156, 1.7399158)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.17 + 37.94 = 60.11 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.6118839, upper bound: 0.6118833

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6117882, upper bound: 0.6060481
time: 6.23 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6118729, upper bound: 0.6118719
time: 10.04 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 16.36 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 16.36
Output dim: 2, lower bound: -0.6117882, upper bound: 0.6060481
NS_A2, status: Status.UNKNOWN, split count: 1, time: 16.36
Output dim: 2, lower bound: -0.6118729, upper bound: 0.6118719

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -13.0943003, -10.4755249, -13.1065807, -10.4753466, -1.7759724, 1.7884445
1: -7.1278410, -4.1993060, -7.1286287, -4.1918344, -2.3185043, 2.3113537
2: 9.3696833, 11.2674513, 9.3686314, 11.2748232, -1.5564294, 1.5496738
3: -4.8695822, -2.7414365, -4.8708601, -2.7387674, -1.9142666, 1.9118223
4: -9.4359198, -6.7293983, -9.4374094, -6.7269249, -1.9578218, 1.9571514
5: -13.7910442, -11.1777363, -13.7946110, -11.1761875, -1.6092191, 1.6107650
6: -16.3353348, -12.7582741, -16.3365364, -12.7566023, -2.2825241, 2.2806134
7: -4.0544090, -1.3716025, -4.0554242, -1.3705735, -2.4592528, 2.4594765
8: -6.0347066, -3.6268167, -6.0362368, -3.6229420, -2.0173368, 2.0154004
9: -11.8219442, -9.3285112, -11.8330145, -9.3281918, -1.7181113, 1.7291591

Time for backsubstitution: 22.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6198
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6198

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6111853, upper bound: 0.6021157
time: 9.08 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6117817, upper bound: 0.6060388
time: 5.91 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -13.1264172, -10.4119635, -13.1174059, -10.4751911, -1.8102484, 1.8183839
1: -7.1655445, -4.1722202, -7.1292839, -4.1849566, -2.3584220, 2.3481565
2: 9.3227787, 11.2848825, 9.3677444, 11.2813282, -1.5872390, 1.5632505
3: -4.8966122, -2.7341878, -4.8719616, -2.7364078, -1.9466472, 1.9276299
4: -9.4410915, -6.7106915, -9.4387341, -6.7248545, -1.9731879, 1.9805927
5: -13.8050451, -11.1576366, -13.7978373, -11.1748848, -1.6463673, 1.6372991
6: -16.3599205, -12.7376900, -16.3375587, -12.7550888, -2.3102832, 2.3105311
7: -4.0664325, -1.3625336, -4.0563083, -1.3696847, -2.4760466, 2.4712358
8: -6.0710554, -3.6127348, -6.0375452, -3.6195045, -2.0637965, 2.0330572
9: -11.8569736, -9.2782335, -11.8428593, -9.3279095, -1.7463279, 1.7589490

Time for backsubstitution: 21.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6198
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 6198

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6112705, upper bound: 0.6079443
time: 7.56 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6118665, upper bound: 0.6118674
time: 5.25 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 34.62 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 34.62
Output dim: 2, lower bound: -0.6111853, upper bound: 0.6021157
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 34.62
Output dim: 2, lower bound: -0.6117817, upper bound: 0.6060388
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 34.62
Output dim: 2, lower bound: -0.6112705, upper bound: 0.6079443
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 34.62
Output dim: 2, lower bound: -0.6118665, upper bound: 0.6118674

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -13.0925570, -10.4765682, -13.1036434, -10.4771299, -1.7718043, 1.7834723
1: -7.1243916, -4.2002764, -7.1227813, -4.1934886, -2.3138728, 2.3047719
2: 9.3770990, 11.2662783, 9.3812151, 11.2728186, -1.5452337, 1.5345402
3: -4.8683424, -2.7428393, -4.8687744, -2.7411346, -1.9077635, 1.9056749
4: -9.4323854, -6.7313271, -9.4314270, -6.7302184, -1.9501910, 1.9475889
5: -13.7902308, -11.1904621, -13.7932158, -11.1977682, -1.5867205, 1.5966635
6: -16.3342781, -12.7613230, -16.3347473, -12.7617559, -2.2725229, 2.2713261
7: -4.0396681, -1.3729491, -4.0304346, -1.3728769, -2.4353251, 2.4268708
8: -6.0320787, -3.6399794, -6.0317430, -3.6452599, -1.9921141, 1.9977226
9: -11.8144131, -9.3292885, -11.8202600, -9.3295155, -1.7092211, 1.7151899

Time for backsubstitution: 21.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6077342, upper bound: 0.6020765
time: 4.98 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6111815, upper bound: 0.6021127
time: 6.89 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -13.0942984, -10.4755268, -13.1100426, -10.4714012, -1.7786295, 1.7916257
1: -7.1278362, -4.1993065, -7.1292934, -4.1850090, -2.3251915, 2.3114834
2: 9.3696957, 11.2674513, 9.3661337, 11.2906618, -1.5657587, 1.5447121
3: -4.8695807, -2.7414377, -4.8723803, -2.7377543, -1.9132094, 1.9146156
4: -9.4359140, -6.7294006, -9.4386425, -6.7193499, -1.9654765, 1.9572196
5: -13.7910452, -11.1777563, -13.8199158, -11.1756687, -1.5956869, 1.6267762
6: -16.3353329, -12.7582798, -16.3424225, -12.7539597, -2.2837515, 2.2837355
7: -4.0543885, -1.3716018, -4.0591202, -1.3437414, -2.4791222, 2.4496841
8: -6.0347047, -3.6268358, -6.0646119, -3.6205878, -2.0072203, 2.0396411
9: -11.8219347, -9.3285141, -11.8353920, -9.3178616, -1.7277009, 1.7271767

Time for backsubstitution: 21.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6083402, upper bound: 0.6060012
time: 8.01 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6117780, upper bound: 0.6060359
time: 6.89 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -13.1247005, -10.4130058, -13.1144705, -10.4769716, -1.8061032, 1.8133992
1: -7.1620893, -4.1731935, -7.1234322, -4.1866007, -2.3531909, 2.3415799
2: 9.3301792, 11.2837095, 9.3803253, 11.2793245, -1.5743771, 1.5481195
3: -4.8953571, -2.7355886, -4.8698740, -2.7387736, -1.9398727, 1.9214811
4: -9.4375610, -6.7126126, -9.4327507, -6.7281446, -1.9655523, 1.9710312
5: -13.8042297, -11.1703615, -13.7964439, -11.1964712, -1.6238368, 1.6220806
6: -16.3588524, -12.7407341, -16.3357639, -12.7602444, -2.2999051, 2.3012385
7: -4.0516863, -1.3638797, -4.0313172, -1.3719852, -2.4521122, 2.4386296
8: -6.0684319, -3.6258864, -6.0330548, -3.6418238, -2.0385518, 2.0153904
9: -11.8494349, -9.2790098, -11.8300991, -9.3292322, -1.7374330, 1.7449756

Time for backsubstitution: 21.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6078208, upper bound: 0.6079028
time: 6.73 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6112667, upper bound: 0.6079433
time: 4.82 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -13.1264153, -10.4119663, -13.1208658, -10.4712458, -1.8129060, 1.8215702
1: -7.1655402, -4.1722202, -7.1299477, -4.1781411, -2.3604741, 2.3482881
2: 9.3227892, 11.2848787, 9.3652382, 11.2971668, -1.5881538, 1.5582869
3: -4.8966093, -2.7341895, -4.8734798, -2.7353933, -1.9456031, 1.9304214
4: -9.4410868, -6.7106938, -9.4399672, -6.7172823, -1.9808383, 1.9806647
5: -13.8050432, -11.1576548, -13.8231440, -11.1743650, -1.6328411, 1.6410468
6: -16.3599205, -12.7376928, -16.3434467, -12.7524462, -2.3094316, 2.3136570
7: -4.0664110, -1.3625355, -4.0600080, -1.3428533, -2.4926977, 2.4614458
8: -6.0710535, -3.6127529, -6.0659232, -3.6171532, -2.0537066, 2.0573378
9: -11.8569660, -9.2782345, -11.8452377, -9.3175812, -1.7559252, 1.7569876

Time for backsubstitution: 21.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6084249, upper bound: 0.6118272
time: 8.64 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6118627, upper bound: 0.6118641
time: 7.51 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 37.85 seconds
NS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 37.85
Output dim: 2, lower bound: -0.6077342, upper bound: 0.6020765
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 37.85
Output dim: 2, lower bound: -0.6111815, upper bound: 0.6021127
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 37.85
Output dim: 2, lower bound: -0.6083402, upper bound: 0.6060012
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 37.85
Output dim: 2, lower bound: -0.6117780, upper bound: 0.6060359
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 37.85
Output dim: 2, lower bound: -0.6078208, upper bound: 0.6079028
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 37.85
Output dim: 2, lower bound: -0.6112667, upper bound: 0.6079433
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 37.85
Output dim: 2, lower bound: -0.6084249, upper bound: 0.6118272
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 37.85
Output dim: 2, lower bound: -0.6118627, upper bound: 0.6118641

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -13.1469097, -10.4710140, -13.1036406, -10.4771442, -1.8008904, 1.7856729
1: -7.1480317, -4.1882467, -7.1227818, -4.1934972, -2.3306928, 2.3272066
2: 9.3623676, 11.2924824, 9.3812237, 11.2728176, -1.5621996, 1.5464182
3: -4.8847141, -2.7376719, -4.8687725, -2.7411375, -1.9244752, 1.9172273
4: -9.4527092, -6.7262573, -9.4314251, -6.7302322, -1.9672074, 1.9515519
5: -13.8291779, -11.1870470, -13.7932148, -11.1977835, -1.6096158, 1.5964248
6: -16.3611069, -12.7321968, -16.3347473, -12.7617626, -2.2948637, 2.3124542
7: -4.0710025, -1.3696856, -4.0304303, -1.3728859, -2.4662294, 2.4280896
8: -6.0751591, -3.6352429, -6.0317430, -3.6452699, -2.0260425, 1.9998302
9: -11.8375244, -9.3210793, -11.8202572, -9.3295174, -1.7354631, 1.7234495

Time for backsubstitution: 21.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6054403, upper bound: 0.6021118
time: 7.97 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6054403, upper bound: 0.6021147
time: 5.96 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -13.1486273, -10.4699669, -13.1100388, -10.4714184, -1.8065100, 1.7938187
1: -7.1514730, -4.1872716, -7.1292925, -4.1850157, -2.3420181, 2.3339190
2: 9.3549805, 11.2936487, 9.3661432, 11.2906609, -1.5786080, 1.5555456
3: -4.8859520, -2.7362618, -4.8723779, -2.7377582, -1.9299269, 1.9261518
4: -9.4562120, -6.7243218, -9.4386396, -6.7193632, -1.9824309, 1.9611421
5: -13.8299961, -11.1743374, -13.8199167, -11.1756811, -1.6186194, 1.6265551
6: -16.3621655, -12.7291632, -16.3424244, -12.7539606, -2.3043952, 2.3233807
7: -4.0856719, -1.3683288, -4.0591140, -1.3437500, -2.4950933, 2.4509130
8: -6.0777588, -3.6221070, -6.0646105, -3.6205988, -2.0411978, 2.0418303
9: -11.8449907, -9.3202982, -11.8353920, -9.3178635, -1.7538948, 1.7354383

Time for backsubstitution: 21.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6060376, upper bound: 0.6060357
time: 7.56 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6060376, upper bound: 0.6060352
time: 7.81 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -13.1790466, -10.4074669, -13.1144714, -10.4769897, -1.8281469, 1.8119764
1: -7.1858320, -4.1615849, -7.1234312, -4.1866078, -2.3541684, 2.3593154
2: 9.3153934, 11.3099041, 9.3803339, 11.2793217, -1.5869982, 1.5602517
3: -4.9118805, -2.7305236, -4.8698721, -2.7387762, -1.9492462, 1.9330664
4: -9.4578571, -6.7075438, -9.4327478, -6.7281585, -1.9818168, 1.9750633
5: -13.8430882, -11.1668396, -13.7964439, -11.1964855, -1.6321445, 1.6218555
6: -16.3857918, -12.7123327, -16.3357620, -12.7602472, -2.3103824, 2.3366928
7: -4.0835643, -1.3606162, -4.0313110, -1.3719945, -2.4820495, 2.4398532
8: -6.1117043, -3.6212697, -6.0330534, -3.6418338, -2.0524907, 2.0174537
9: -11.8722582, -9.2708607, -11.8300962, -9.3292370, -1.7636170, 1.7523746

Time for backsubstitution: 21.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6054385, upper bound: 0.6078547
time: 7.95 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6054385, upper bound: 0.6079410
time: 10.81 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -13.1252766, -10.4250689, -13.1204109, -10.4766903, -1.8061526, 1.8066926
1: -7.1645203, -4.1780958, -7.1295276, -4.1806397, -2.3499336, 2.3351684
2: 9.3305998, 11.2835798, 9.3684921, 11.2966394, -1.5781779, 1.5518274
3: -4.8958035, -2.7366655, -4.8731556, -2.7364304, -1.9417307, 1.9253588
4: -9.4388771, -6.7179155, -9.4390488, -6.7202787, -1.9706612, 1.9699678
5: -13.8045340, -11.1675892, -13.8229485, -11.1785030, -1.6275675, 1.6306841
6: -16.3592815, -12.7413254, -16.3431911, -12.7539873, -2.3058376, 2.3068027
7: -4.0623255, -1.3691781, -4.0583224, -1.3456066, -2.4836693, 2.4522433
8: -6.0695095, -3.6219668, -6.0652943, -3.6209774, -2.0474949, 2.0473912
9: -11.8541756, -9.2798891, -11.8440866, -9.3182793, -1.7518203, 1.7539392

Time for backsubstitution: 22.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6025977, upper bound: 0.6117430
time: 6.36 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6025977, upper bound: 0.6118281
time: 5.87 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -13.1807680, -10.4064226, -13.1208658, -10.4712601, -1.8337412, 1.8201530
1: -7.1892829, -4.1606050, -7.1299462, -4.1781502, -2.3614602, 2.3660321
2: 9.3080206, 11.3110714, 9.3652468, 11.2971649, -1.6007905, 1.5693789
3: -4.9131346, -2.7291133, -4.8734798, -2.7353954, -1.9549792, 1.9419799
4: -9.4613609, -6.7056141, -9.4399643, -6.7172947, -1.9928966, 1.9846625
5: -13.8439074, -11.1541328, -13.8231440, -11.1743803, -1.6411452, 1.6408215
6: -16.3868618, -12.7092991, -16.3434467, -12.7524509, -2.3199062, 2.3467278
7: -4.0982394, -1.3592606, -4.0600028, -1.3428612, -2.5085034, 2.4626751
8: -6.1142550, -3.6081443, -6.0659208, -3.6171637, -2.0676413, 2.0594797
9: -11.8797359, -9.2700768, -11.8452358, -9.3175831, -1.7820578, 1.7643931

Time for backsubstitution: 22.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6060358, upper bound: 0.6117777
time: 6.79 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6060358, upper bound: 0.6118649
time: 5.39 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 34.47 seconds
NS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 34.47
Output dim: 2, lower bound: -0.6054403, upper bound: 0.6021118
NS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 34.47
Output dim: 2, lower bound: -0.6054403, upper bound: 0.6021147
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 34.47
Output dim: 2, lower bound: -0.6060376, upper bound: 0.6060357
NS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 34.47
Output dim: 2, lower bound: -0.6060376, upper bound: 0.6060352
NS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 34.47
Output dim: 2, lower bound: -0.6054385, upper bound: 0.6078547
NS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 34.47
Output dim: 2, lower bound: -0.6054385, upper bound: 0.6079410
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 34.47
Output dim: 2, lower bound: -0.6025977, upper bound: 0.6117430
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 34.47
Output dim: 2, lower bound: -0.6025977, upper bound: 0.6118281
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 34.47
Output dim: 2, lower bound: -0.6060358, upper bound: 0.6117777
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 34.47
Output dim: 2, lower bound: -0.6060358, upper bound: 0.6118649

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -13.1252546, -10.4250727, -13.0973101, -10.4770260, -1.8069315, 1.7813993
1: -7.1643596, -4.1781011, -7.1280828, -4.1949272, -2.3308089, 2.3191748
2: 9.3310509, 11.2835741, 9.3704433, 11.2827635, -1.5624135, 1.5564337
3: -4.8957338, -2.7366786, -4.8707747, -2.7414608, -1.9336181, 1.9158850
4: -9.4388723, -6.7185926, -9.4362335, -6.7248201, -1.9576716, 1.9642739
5: -13.8045359, -11.1683969, -13.8161554, -11.1813536, -1.5983300, 1.6151121
6: -16.3590546, -12.7413445, -16.3409576, -12.7571602, -2.3004088, 2.2978082
7: -4.0623112, -1.3691859, -4.0564127, -1.3475213, -2.4808888, 2.4493890
8: -6.0694933, -3.6220307, -6.0624547, -3.6282864, -2.0335550, 2.0382969
9: -11.8539762, -9.2798882, -11.8231707, -9.3188801, -1.7583435, 1.7324023

Time for backsubstitution: 22.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6198

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5986647, upper bound: 0.6111429
time: 7.30 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.5986647, upper bound: 0.6086488
time: 5.37 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -13.1252899, -10.4250708, -13.1293983, -10.4134541, -1.8094070, 1.8018167
1: -7.1646338, -4.1780910, -7.1659069, -4.1678467, -2.3494854, 2.3397317
2: 9.3302755, 11.2835884, 9.3231945, 11.3001871, -1.5796282, 1.5638833
3: -4.8958530, -2.7366557, -4.8978577, -2.7341976, -1.9330926, 1.9355426
4: -9.4388790, -6.7174344, -9.4414043, -6.7056241, -1.9774933, 1.9694066
5: -13.8045359, -11.1670084, -13.8301458, -11.1606550, -1.6300721, 1.6483157
6: -16.3594437, -12.7413092, -16.3657188, -12.7365608, -2.3253303, 2.3214171
7: -4.0623364, -1.3691714, -4.0684481, -1.3384480, -2.4887209, 2.4630294
8: -6.0695229, -3.6219211, -6.0988102, -3.6141939, -2.0548005, 2.0669491
9: -11.8543158, -9.2798882, -11.8583202, -9.2685814, -1.7608557, 1.7503293

Time for backsubstitution: 22.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6198

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5986647, upper bound: 0.6112299
time: 5.31 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.5986647, upper bound: 0.6087319
time: 7.76 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -13.1807508, -10.4064226, -13.0977621, -10.4715977, -1.8301816, 1.7948604
1: -7.1891203, -4.1606131, -7.1285019, -4.1924858, -2.3423467, 2.3549237
2: 9.3084755, 11.3110619, 9.3672009, 11.2832890, -1.5850241, 1.5666232
3: -4.9130740, -2.7291269, -4.8710995, -2.7404294, -1.9468663, 1.9324117
4: -9.4613571, -6.7062869, -9.4371519, -6.7218370, -1.9847741, 1.9789667
5: -13.8439054, -11.1549387, -13.8163509, -11.1772299, -1.6241405, 1.6252511
6: -16.3866386, -12.7093229, -16.3412170, -12.7556362, -2.3144829, 2.3439226
7: -4.0982304, -1.3592710, -4.0580950, -1.3447785, -2.5057225, 2.4598203
8: -6.1142378, -3.6082106, -6.0630846, -3.6244736, -2.0562387, 2.0503962
9: -11.8795462, -9.2700806, -11.8243189, -9.3181858, -1.7826130, 1.7428577

Time for backsubstitution: 22.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6198

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6021120, upper bound: 0.6111807
time: 8.07 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6021120, upper bound: 0.6086810
time: 6.72 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -13.1807804, -10.4064217, -13.1298752, -10.4080362, -1.8368030, 1.8188448
1: -7.1893959, -4.1605997, -7.1663313, -4.1654005, -2.3757544, 2.3706124
2: 9.3076954, 11.3110790, 9.3199682, 11.3007183, -1.6038196, 1.5809851
3: -4.9131770, -2.7291026, -4.8981891, -2.7331667, -1.9534345, 1.9521532
4: -9.4613638, -6.7051344, -9.4423285, -6.7026472, -1.9996874, 1.9839911
5: -13.8439045, -11.1535530, -13.8303518, -11.1565514, -1.6436443, 1.6584249
6: -16.3870239, -12.7092857, -16.3659801, -12.7350569, -2.3394177, 2.3582788
7: -4.0982456, -1.3592553, -4.0701380, -1.3357055, -2.5135555, 2.4734478
8: -6.1142669, -3.6080980, -6.0994411, -3.6103840, -2.0749369, 2.0790346
9: -11.8798695, -9.2700796, -11.8594837, -9.2678967, -1.7880793, 1.7617190

Time for backsubstitution: 22.12 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 60.11 + 541.97 = 602.08 seconds
