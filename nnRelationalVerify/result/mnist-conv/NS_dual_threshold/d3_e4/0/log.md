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
execution time: IAR + RelationalAnalysis = 22.59 + 37.01 = 59.60 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.6118839, upper bound: 0.6118833

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5735
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 6198
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6117882, upper bound: 0.6060481
time: 6.16 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6118729, upper bound: 0.6118719
time: 9.95 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 16.23 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 16.23
Output dim: 2, lower bound: -0.6117882, upper bound: 0.6060481
NS_A2, status: Status.UNKNOWN, split count: 1, time: 16.23
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

Time for backsubstitution: 21.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6198
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 848

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6198

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6111853, upper bound: 0.6021157
time: 8.75 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6117817, upper bound: 0.6060388
time: 5.41 seconds

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

Time for backsubstitution: 21.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6198
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 848

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6198

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6112705, upper bound: 0.6079443
time: 7.88 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6118665, upper bound: 0.6118674
time: 5.38 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 35.21 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 35.21
Output dim: 2, lower bound: -0.6111853, upper bound: 0.6021157
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 35.21
Output dim: 2, lower bound: -0.6117817, upper bound: 0.6060388
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 35.21
Output dim: 2, lower bound: -0.6112705, upper bound: 0.6079443
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 35.21
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

Time for backsubstitution: 21.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 6218
type: A, layer: 1, pos: 848

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of NS_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6054440, upper bound: 0.6021160
time: 9.36 seconds

## Relational analysis of NS_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6054440, upper bound: 0.6021180
time: 5.22 seconds

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

Time for backsubstitution: 21.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 6218
type: A, layer: 1, pos: 848

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6060413, upper bound: 0.6060394
time: 8.53 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6060413, upper bound: 0.6060384
time: 4.66 seconds

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

Time for backsubstitution: 21.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 848

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6112302, upper bound: 0.6044936
time: 6.29 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6112672, upper bound: 0.6079401
time: 7.20 seconds

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

Time for backsubstitution: 21.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 848

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6118287, upper bound: 0.6084235
time: 6.70 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6118632, upper bound: 0.6118613
time: 11.77 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 40.41 seconds
NS_A1_B1_B1, status: Status.VERIFIED, split count: 3, time: 40.41
Output dim: 2, lower bound: -0.6054440, upper bound: 0.6021160
NS_A1_B1_B2, status: Status.VERIFIED, split count: 3, time: 40.41
Output dim: 2, lower bound: -0.6054440, upper bound: 0.6021180
NS_A1_B2_B1, status: Status.VERIFIED, split count: 3, time: 40.41
Output dim: 2, lower bound: -0.6060413, upper bound: 0.6060394
NS_A1_B2_B2, status: Status.VERIFIED, split count: 3, time: 40.41
Output dim: 2, lower bound: -0.6060413, upper bound: 0.6060384
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 40.41
Output dim: 2, lower bound: -0.6112302, upper bound: 0.6044936
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 40.41
Output dim: 2, lower bound: -0.6112672, upper bound: 0.6079401
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 40.41
Output dim: 2, lower bound: -0.6118287, upper bound: 0.6084235
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 40.41
Output dim: 2, lower bound: -0.6118632, upper bound: 0.6118613

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -13.1242285, -10.4184418, -13.1133833, -10.4901056, -1.7912412, 1.8057655
1: -7.1616688, -4.1756473, -7.1224241, -4.1925831, -2.3397651, 2.3316536
2: 9.3334160, 11.2831726, 9.3881721, 11.2780266, -1.5673270, 1.5382581
3: -4.8950262, -2.7366190, -4.8690844, -2.7412488, -1.9347787, 1.9177961
4: -9.4366493, -6.7156096, -9.4305630, -6.7353888, -1.9548645, 1.9608603
5: -13.8040228, -11.1744804, -13.7959576, -11.2064381, -1.6134496, 1.6168442
6: -16.3585930, -12.7422428, -16.3351498, -12.7639475, -2.2930908, 2.2979860
7: -4.0500059, -1.3666348, -4.0272679, -1.3786290, -2.4429455, 2.4300284
8: -6.0677986, -3.6297050, -6.0315161, -3.6510448, -2.0286007, 2.0097284
9: -11.8482819, -9.2796974, -11.8273287, -9.3309202, -1.7343960, 1.7408068

Time for backsubstitution: 21.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 848

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6078209, upper bound: 0.6044936
time: 7.24 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6078209, upper bound: 0.6044936
time: 6.07 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -13.1246967, -10.4130192, -13.1687794, -10.4713974, -1.8082759, 1.8284336
1: -7.1620898, -4.1732011, -7.1470981, -4.1747661, -2.3593986, 2.3539829
2: 9.3301878, 11.2837086, 9.3655405, 11.3055210, -1.5768921, 1.5650349
3: -4.8953567, -2.7355940, -4.8862619, -2.7336226, -1.9447002, 1.9381332
4: -9.4375563, -6.7126245, -9.4530754, -6.7230663, -1.9694977, 1.9854271
5: -13.8042278, -11.1703758, -13.8353539, -11.1930065, -1.6236095, 1.6305624
6: -16.3588543, -12.7407379, -16.3626175, -12.7311850, -2.3302245, 2.3172777
7: -4.0516815, -1.3638897, -4.0626273, -1.3687215, -2.4533396, 2.4695811
8: -6.0684309, -3.6258941, -6.0762200, -3.6370931, -2.0407138, 2.0466013
9: -11.8494339, -9.2790117, -11.8531523, -9.3210049, -1.7456975, 1.7652884

Time for backsubstitution: 21.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 848

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of NS_A2_B1_B2_B1

### Relational analysis result of NS_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6054389, upper bound: 0.6078545
time: 7.02 seconds

## Relational analysis of NS_A2_B1_B2_B2

### Relational analysis result of NS_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6054389, upper bound: 0.6079435
time: 6.62 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: -13.1259441, -10.4174023, -13.1197653, -10.4843731, -1.7980533, 1.8139254
1: -7.1651201, -4.1746759, -7.1289310, -4.1841216, -2.3470440, 2.3383570
2: 9.3260269, 11.2843437, 9.3730850, 11.2958813, -1.5811026, 1.5484154
3: -4.8962793, -2.7352209, -4.8726888, -2.7378821, -1.9404931, 1.9267406
4: -9.4401741, -6.7136898, -9.4377394, -6.7245111, -1.9701591, 1.9704628
5: -13.8048391, -11.1617756, -13.8226566, -11.1843386, -1.6224504, 1.6358063
6: -16.3596535, -12.7392063, -16.3428268, -12.7561512, -2.3026128, 2.3103988
7: -4.0647306, -1.3652897, -4.0559130, -1.3494911, -2.4834800, 2.4528270
8: -6.0704217, -3.6165705, -6.0643902, -3.6263776, -2.0437584, 2.0511086
9: -11.8558168, -9.2789211, -11.8424482, -9.3192635, -1.7528868, 1.7528126

Time for backsubstitution: 22.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of NS_A2_B2_B1_A1

### Relational analysis result of NS_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6084250, upper bound: 0.6084258
time: 5.15 seconds

## Relational analysis of NS_A2_B2_B1_A2

### Relational analysis result of NS_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6084250, upper bound: 0.6084237
time: 6.34 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: -13.1264143, -10.4119778, -13.1751690, -10.4656601, -1.8150728, 1.8365107
1: -7.1655393, -4.1722279, -7.1536160, -4.1662812, -2.3666859, 2.3607030
2: 9.3227978, 11.2848806, 9.3505306, 11.3233585, -1.5906720, 1.5752621
3: -4.8966088, -2.7341928, -4.8898702, -2.7302461, -1.9504235, 1.9469013
4: -9.4410849, -6.7107053, -9.4602365, -6.7121854, -1.9847732, 1.9950545
5: -13.8050432, -11.1576710, -13.8620577, -11.1709080, -1.6326118, 1.6495160
6: -16.3599167, -12.7376986, -16.3702908, -12.7234449, -2.3397245, 2.3272734
7: -4.0664067, -1.3625445, -4.0912299, -1.3395813, -2.4938841, 2.4922256
8: -6.0710530, -3.6127620, -6.1089945, -3.6124754, -2.0558414, 2.0713100
9: -11.8569641, -9.2782354, -11.8682203, -9.3093510, -1.7642040, 1.7772701

Time for backsubstitution: 22.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 848

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of NS_A2_B2_B2_B1

### Relational analysis result of NS_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6060362, upper bound: 0.6117767
time: 6.68 seconds

## Relational analysis of NS_A2_B2_B2_B2

### Relational analysis result of NS_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6060362, upper bound: 0.6118617
time: 6.96 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 36.23 seconds
NS_A2_B1_B1_A1, status: Status.VERIFIED, split count: 4, time: 36.23
Output dim: 2, lower bound: -0.6078209, upper bound: 0.6044936
NS_A2_B1_B1_A2, status: Status.VERIFIED, split count: 4, time: 36.23
Output dim: 2, lower bound: -0.6078209, upper bound: 0.6044936
NS_A2_B1_B2_B1, status: Status.VERIFIED, split count: 4, time: 36.23
Output dim: 2, lower bound: -0.6054389, upper bound: 0.6078545
NS_A2_B1_B2_B2, status: Status.VERIFIED, split count: 4, time: 36.23
Output dim: 2, lower bound: -0.6054389, upper bound: 0.6079435
NS_A2_B2_B1_A1, status: Status.VERIFIED, split count: 4, time: 36.23
Output dim: 2, lower bound: -0.6084250, upper bound: 0.6084258
NS_A2_B2_B1_A2, status: Status.VERIFIED, split count: 4, time: 36.23
Output dim: 2, lower bound: -0.6084250, upper bound: 0.6084237
NS_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 36.23
Output dim: 2, lower bound: -0.6060362, upper bound: 0.6117767
NS_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 36.23
Output dim: 2, lower bound: -0.6060362, upper bound: 0.6118617

## BFS NS instance: NS_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -13.1263943, -10.4119806, -13.1521206, -10.4660244, -1.8140743, 1.8111746
1: -7.1653795, -4.1722350, -7.1521487, -4.1804457, -2.3476200, 2.3491964
2: 9.3232527, 11.2848711, 9.3525219, 11.3094940, -1.5749128, 1.5766468
3: -4.8965397, -2.7342062, -4.8874731, -2.7352715, -1.9422944, 1.9376798
4: -9.4410810, -6.7113819, -9.4574404, -6.7167435, -1.9718328, 1.9896855
5: -13.8050404, -11.1584797, -13.8553028, -11.1738081, -1.6027803, 1.6339173
6: -16.3596916, -12.7377205, -16.3680344, -12.7265606, -2.3343234, 2.3245335
7: -4.0663910, -1.3625517, -4.0893803, -1.3415089, -2.4910941, 2.4893510
8: -6.0710349, -3.6128244, -6.1061354, -3.6197739, -2.0412188, 2.0623202
9: -11.8567667, -9.2782364, -11.8473978, -9.3099804, -1.7700562, 1.7558174

Time for backsubstitution: 21.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 848

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 4586

## Relational analysis of NS_A2_B2_B2_B1_B1

### Relational analysis result of NS_A2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6058926, upper bound: 0.6103989
time: 8.63 seconds

## Relational analysis of NS_A2_B2_B2_B1_B2

### Relational analysis result of NS_A2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6060350, upper bound: 0.6117749
time: 9.49 seconds

## BFS NS instance: NS_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -13.1264267, -10.4119787, -13.1842289, -10.4024811, -1.8183208, 1.8517246
1: -7.1656556, -4.1722255, -7.1900783, -4.1537781, -2.3810883, 2.3652658
2: 9.3224735, 11.2848854, 9.3052330, 11.3269176, -1.5933158, 1.5875192
3: -4.8966589, -2.7341826, -4.9147067, -2.7281067, -1.9483323, 1.9569366
4: -9.4410858, -6.7102237, -9.4625912, -6.6975555, -1.9916134, 1.9971380
5: -13.8050442, -11.1570950, -13.8692112, -11.1530218, -1.6350608, 1.6669995
6: -16.3600807, -12.7376842, -16.3929100, -12.7067032, -2.3588877, 2.3387680
7: -4.0664163, -1.3625388, -4.1019650, -1.3324347, -2.4989023, 2.5021186
8: -6.0710640, -3.6127167, -6.1426492, -3.6058021, -2.0631146, 2.0908346
9: -11.8571043, -9.2782345, -11.8822680, -9.2597504, -1.7733130, 1.7795570

Time for backsubstitution: 23.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 848

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4586

## Relational analysis of NS_A2_B2_B2_B2_B1

### Relational analysis result of NS_A2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6058926, upper bound: 0.6104863
time: 5.74 seconds

## Relational analysis of NS_A2_B2_B2_B2_B2

### Relational analysis result of NS_A2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6060350, upper bound: 0.6118630
time: 5.34 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 34.54 seconds
NS_A2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 34.54
Output dim: 2, lower bound: -0.6058926, upper bound: 0.6103989
NS_A2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 34.54
Output dim: 2, lower bound: -0.6060350, upper bound: 0.6117749
NS_A2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 34.54
Output dim: 2, lower bound: -0.6058926, upper bound: 0.6104863
NS_A2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 34.54
Output dim: 2, lower bound: -0.6060350, upper bound: 0.6118630

## BFS NS instance: NS_A2_B2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -13.1251955, -10.4172697, -13.1431427, -10.4762878, -1.8026228, 1.7954257
1: -7.1640873, -4.1769648, -7.1440225, -4.1896672, -2.3364251, 2.3347363
2: 9.3276854, 11.2836399, 9.3618116, 11.3021841, -1.5615833, 1.5657389
3: -4.8955131, -2.7368760, -4.8813162, -2.7403083, -1.9360988, 1.9291925
4: -9.4387112, -6.7151785, -9.4495335, -6.7242889, -1.9593716, 1.9717171
5: -13.8032312, -11.1601229, -13.8491621, -11.1774235, -1.5973325, 1.6264782
6: -16.3587189, -12.7424078, -16.3588028, -12.7386208, -2.3209729, 2.3108370
7: -4.0620618, -1.3630443, -4.0797834, -1.3433244, -2.4822078, 2.4765890
8: -6.0697899, -3.6186886, -6.0963178, -3.6311979, -2.0279522, 2.0435028
9: -11.8499861, -9.2783871, -11.8321247, -9.3152037, -1.7564397, 1.7405167

Time for backsubstitution: 23.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 848

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6198

## Relational analysis of NS_A2_B2_B2_B1_B1_A1

### Relational analysis result of NS_A2_B2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6019675, upper bound: 0.6098001
time: 6.93 seconds

## Relational analysis of NS_A2_B2_B2_B1_B1_A2

### Relational analysis result of NS_A2_B2_B2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6019674, upper bound: 0.6073025
time: 6.02 seconds

## BFS NS instance: NS_A2_B2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -13.1263943, -10.4119873, -13.1521187, -10.4660330, -1.8041170, 1.8061683
1: -7.1653786, -4.1722422, -7.1521454, -4.1804571, -2.3410959, 2.3474483
2: 9.3232527, 11.2848692, 9.3525238, 11.3094940, -1.5712171, 1.5685589
3: -4.8965378, -2.7342093, -4.8874726, -2.7352750, -1.9398694, 1.9366217
4: -9.4410782, -6.7113857, -9.4574375, -6.7167525, -1.9667115, 1.9863646
5: -13.8050423, -11.1584806, -13.8552990, -11.1738119, -1.6011686, 1.6324472
6: -16.3596897, -12.7377224, -16.3680344, -12.7265663, -2.3265243, 2.3196814
7: -4.0663886, -1.3625526, -4.0893731, -1.3415103, -2.4913487, 2.4885690
8: -6.0710354, -3.6128321, -6.1061358, -3.6197839, -2.0327930, 2.0568898
9: -11.8567629, -9.2782364, -11.8473883, -9.3099775, -1.7653539, 1.7470398

Time for backsubstitution: 21.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 4586

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6198

## Relational analysis of NS_A2_B2_B2_B1_B2_A1

### Relational analysis result of NS_A2_B2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6021113, upper bound: 0.6111793
time: 5.07 seconds

## Relational analysis of NS_A2_B2_B2_B1_B2_A2

### Relational analysis result of NS_A2_B2_B2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6021113, upper bound: 0.6086790
time: 6.26 seconds

## BFS NS instance: NS_A2_B2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -13.1252279, -10.4172668, -13.1749344, -10.4130898, -1.8065782, 1.8357558
1: -7.1643615, -4.1769547, -7.1811552, -4.1630092, -2.3698583, 2.3504705
2: 9.3269081, 11.2836561, 9.3147564, 11.3186903, -1.5780268, 1.5762944
3: -4.8956327, -2.7368507, -4.9083776, -2.7331643, -1.9421086, 1.9482837
4: -9.4387169, -6.7140193, -9.4542284, -6.7051549, -1.9790998, 1.9785242
5: -13.8032331, -11.1587372, -13.8620758, -11.1566858, -1.6296358, 1.6589911
6: -16.3591080, -12.7423754, -16.3818703, -12.7188158, -2.3455734, 2.3238630
7: -4.0620880, -1.3630292, -4.0922666, -1.3343678, -2.4898529, 2.4893029
8: -6.0698195, -3.6185808, -6.1327572, -3.6177735, -2.0494871, 2.0719862
9: -11.8503227, -9.2783861, -11.8667297, -9.2650146, -1.7610729, 1.7639439

Time for backsubstitution: 22.76 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 59.60 + 541.77 = 601.38 seconds
