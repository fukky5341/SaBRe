## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.281490264


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-12.2602692, -10.6615553, -12.2602692, -10.6615553, -1.1638608, 1.1638610)
1: (3.3816395, 4.2802763, 3.3816395, 4.2802763, -0.5360075, 0.5360075)
2: (-4.7594166, -3.9467888, -4.7594166, -3.9467888, -0.5615702, 0.5615700)
3: (-12.5689964, -11.2200060, -12.5689964, -11.2200060, -0.8178110, 0.8178110)
4: (-2.1814775, -1.1082797, -2.1814775, -1.1082797, -0.7643485, 0.7643486)
5: (-9.8950491, -8.8726807, -9.8950491, -8.8726807, -0.5886670, 0.5886672)
6: (-7.8550801, -6.6118288, -7.8550801, -6.6118288, -0.8692248, 0.8692250)
7: (-2.6614103, -2.0481133, -2.6614103, -2.0481133, -0.3831897, 0.3831897)
8: (-3.6533647, -2.6237564, -3.6533647, -2.6237564, -0.6626787, 0.6626787)
9: (-12.3033600, -11.2095757, -12.3033600, -11.2095757, -0.7445683, 0.7445687)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.33 + 34.11 = 56.44 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.2843336, upper bound: 0.2843319

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 5815
type: A, layer: 1, pos: 6193
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2809320, upper bound: 0.2842797
time: 3.97 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2843234, upper bound: 0.2843236
time: 3.56 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.61 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 7.61
Output dim: 1, lower bound: -0.2809320, upper bound: 0.2842797
NS_A2, status: Status.UNKNOWN, split count: 1, time: 7.61
Output dim: 1, lower bound: -0.2843234, upper bound: 0.2843236

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -12.2566128, -10.6660509, -12.2584133, -10.6637945, -1.1574161, 1.1560495
1: 3.3941870, 4.2790804, 3.3878903, 4.2797050, -0.5223285, 0.5279497
2: -4.7570295, -3.9528306, -4.7581630, -3.9497879, -0.5572194, 0.5543010
3: -12.5673180, -11.2399158, -12.5681963, -11.2307301, -0.8070064, 0.7986674
4: -2.1786361, -1.1251497, -2.1798735, -1.1166773, -0.7538729, 0.7462382
5: -9.8549080, -8.8744802, -9.8751173, -8.8739586, -0.5487707, 0.5673336
6: -7.8242140, -6.6131554, -7.8397293, -6.6124611, -0.8372519, 0.8506219
7: -2.6614103, -2.0566189, -2.6614103, -2.0524929, -0.3746436, 0.3712261
8: -3.6503315, -2.6554260, -3.6517224, -2.6394753, -0.6432962, 0.6295264
9: -12.2851992, -11.2112570, -12.2942438, -11.2106705, -0.7164712, 0.7220800

Time for backsubstitution: 19.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 901

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 5815

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2758145, upper bound: 0.2840262
time: 3.39 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2809252, upper bound: 0.2842766
time: 3.55 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -12.2632561, -10.6611099, -12.2602692, -10.6615610, -1.1680565, 1.1630223
1: 3.3802824, 4.2914476, 3.3816509, 4.2802758, -0.5305270, 0.5391240
2: -4.7639046, -3.9451995, -4.7594156, -3.9467945, -0.5667387, 0.5626214
3: -12.5861626, -11.2193661, -12.5689945, -11.2200222, -0.8214369, 0.8113990
4: -2.1983910, -1.1079626, -2.1814775, -1.1082890, -0.7762887, 0.7570059
5: -9.8970680, -8.8315592, -9.8950367, -8.8726826, -0.5663214, 0.5943046
6: -7.8568144, -6.5845189, -7.8550687, -6.6118283, -0.8540559, 0.8799009
7: -2.6678953, -2.0473022, -2.6614103, -2.0481203, -0.3863180, 0.3804188
8: -3.6862607, -2.6237860, -3.6533632, -2.6237745, -0.6692147, 0.6477579
9: -12.3055668, -11.1960287, -12.3033533, -11.2095757, -0.7403207, 0.7482436

Time for backsubstitution: 20.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 901

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 5815

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2791878, upper bound: 0.2840693
time: 3.59 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2843180, upper bound: 0.2843204
time: 3.74 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 27.50 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 27.50
Output dim: 1, lower bound: -0.2758145, upper bound: 0.2840262
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 27.50
Output dim: 1, lower bound: -0.2809252, upper bound: 0.2842766
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 27.50
Output dim: 1, lower bound: -0.2791878, upper bound: 0.2840693
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 27.50
Output dim: 1, lower bound: -0.2843180, upper bound: 0.2843204

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -12.2517338, -10.6946659, -12.1878099, -10.7224007, -1.0956264, 1.0543487
1: 3.3968010, 4.2719736, 3.4088454, 4.2642565, -0.5032534, 0.4999350
2: -4.7555079, -3.9634695, -4.7327061, -3.9715309, -0.5342877, 0.5193375
3: -12.5368519, -11.2434378, -12.5072165, -11.2916002, -0.7077967, 0.7359343
4: -2.1760983, -1.1259422, -2.1681695, -1.1218863, -0.7409921, 0.7324990
5: -9.8468704, -8.8754568, -9.8584661, -8.8929510, -0.5222907, 0.5484744
6: -7.8192291, -6.6291666, -7.7897081, -6.6463513, -0.7939923, 0.7836859
7: -2.6610959, -2.0586338, -2.6601472, -2.0571985, -0.3690907, 0.3666296
8: -3.6478987, -2.6704264, -3.6204443, -2.6705756, -0.6117899, 0.5852171
9: -12.2797785, -11.2152147, -12.2787485, -11.2287388, -0.6906588, 0.7020100

Time for backsubstitution: 21.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6193
type: A, layer: 1, pos: 5815
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6193

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2754766, upper bound: 0.2797667
time: 3.28 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2758108, upper bound: 0.2840245
time: 3.47 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -12.2566118, -10.6660681, -12.2584124, -10.6638269, -1.1221824, 1.1104729
1: 3.3941875, 4.2790742, 3.3878932, 4.2796898, -0.5150266, 0.5172635
2: -4.7570286, -3.9528391, -4.7581615, -3.9498048, -0.5423324, 0.5415295
3: -12.5672913, -11.2399168, -12.5681391, -11.2307320, -0.7571170, 0.7312961
4: -2.1786356, -1.1251497, -2.1798706, -1.1166790, -0.7534382, 0.7449692
5: -9.8549023, -8.8744812, -9.8751040, -8.8739605, -0.5410740, 0.5523677
6: -7.8242111, -6.6131659, -7.8397255, -6.6124811, -0.8265829, 0.8283508
7: -2.6614099, -2.0566192, -2.6614094, -2.0524945, -0.3743684, 0.3709040
8: -3.6503305, -2.6554351, -3.6517205, -2.6394963, -0.6281379, 0.6102347
9: -12.2851954, -11.2112579, -12.2942333, -11.2106733, -0.7180564, 0.7215455

Time for backsubstitution: 20.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6193
type: A, layer: 1, pos: 5815
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 6193

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2807221, upper bound: 0.2801545
time: 3.56 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2809228, upper bound: 0.2842738
time: 3.29 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -12.2583752, -10.6897182, -12.1896648, -10.7201672, -1.1062655, 1.0615618
1: 3.3828955, 4.2843399, 3.4026499, 4.2648292, -0.5114723, 0.5099988
2: -4.7623839, -3.9558229, -4.7339602, -3.9685259, -0.5438181, 0.5276715
3: -12.5556955, -11.2228880, -12.5080175, -11.2808914, -0.7222140, 0.7486691
4: -2.1958218, -1.1087556, -2.1697092, -1.1134973, -0.7632868, 0.7431195
5: -9.8890390, -8.8325434, -9.8783884, -8.8916836, -0.5398194, 0.5754423
6: -7.8518467, -6.6005287, -7.8049889, -6.6457181, -0.8108070, 0.8077743
7: -2.6675804, -2.0493186, -2.6601472, -2.0528264, -0.3807703, 0.3758363
8: -3.6838260, -2.6387863, -3.6220851, -2.6548738, -0.6377091, 0.6034980
9: -12.3001442, -11.1999998, -12.2878571, -11.2276592, -0.7145023, 0.7281858

Time for backsubstitution: 21.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6193
type: A, layer: 1, pos: 5815
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 6193

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2788679, upper bound: 0.2798104
time: 3.54 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2791843, upper bound: 0.2840660
time: 3.46 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -12.2632542, -10.6611271, -12.2602663, -10.6615944, -1.1306434, 1.1176751
1: 3.3802829, 4.2914400, 3.3816547, 4.2802615, -0.5233243, 0.5273352
2: -4.7639046, -3.9452078, -4.7594142, -3.9468119, -0.5491769, 0.5498548
3: -12.5861359, -11.2193699, -12.5689411, -11.2200270, -0.7715480, 0.7440569
4: -2.1983895, -1.1079626, -2.1814737, -1.1082897, -0.7706106, 0.7557380
5: -9.8970623, -8.8315601, -9.8950243, -8.8726826, -0.5587366, 0.5793320
6: -7.8568134, -6.5845280, -7.8550644, -6.6118488, -0.8435698, 0.8506577
7: -2.6678948, -2.0473027, -2.6614094, -2.0481219, -0.3860429, 0.3800966
8: -3.6862602, -2.6237950, -3.6533608, -2.6237926, -0.6540562, 0.6285163
9: -12.3055611, -11.1960297, -12.3033409, -11.2095785, -0.7418928, 0.7456237

Time for backsubstitution: 22.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6193
type: A, layer: 1, pos: 5815
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 6193

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2841149, upper bound: 0.2802048
time: 4.37 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2843156, upper bound: 0.2843162
time: 4.21 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 30.70 seconds
NS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 30.70
Output dim: 1, lower bound: -0.2754766, upper bound: 0.2797667
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 30.70
Output dim: 1, lower bound: -0.2758108, upper bound: 0.2840245
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 30.70
Output dim: 1, lower bound: -0.2807221, upper bound: 0.2801545
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 30.70
Output dim: 1, lower bound: -0.2809228, upper bound: 0.2842738
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 30.70
Output dim: 1, lower bound: -0.2788679, upper bound: 0.2798104
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 30.70
Output dim: 1, lower bound: -0.2791843, upper bound: 0.2840660
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 30.70
Output dim: 1, lower bound: -0.2841149, upper bound: 0.2802048
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 30.70
Output dim: 1, lower bound: -0.2843156, upper bound: 0.2843162

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -12.2517281, -10.6947384, -12.1878090, -10.7224369, -1.0494647, 1.0155833
1: 3.3968043, 4.2719660, 3.4088459, 4.2642536, -0.4971843, 0.4885923
2: -4.7555065, -3.9634764, -4.7327051, -3.9715343, -0.5140730, 0.4908179
3: -12.5368376, -11.2434406, -12.5072079, -11.2916031, -0.6889184, 0.7199559
4: -2.1760945, -1.1259561, -2.1681678, -1.1218905, -0.7368441, 0.7294123
5: -9.8468666, -8.8754578, -9.8584652, -8.8929520, -0.5198271, 0.5468869
6: -7.8192229, -6.6291852, -7.7897038, -6.6463623, -0.7908707, 0.7690461
7: -2.6610951, -2.0586424, -2.6601470, -2.0572023, -0.3654722, 0.3508110
8: -3.6478968, -2.6704359, -3.6204429, -2.6705809, -0.6074224, 0.5629805
9: -12.2797785, -11.2152424, -12.2787476, -11.2287483, -0.6906517, 0.6948366

Time for backsubstitution: 20.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 901

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 6193

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2715560, upper bound: 0.2836858
time: 7.72 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2715560, upper bound: 0.2836876
time: 3.26 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -12.2566090, -10.6661377, -12.2584076, -10.6638632, -1.0754588, 1.0740044
1: 3.3941908, 4.2790656, 3.3878946, 4.2796865, -0.5067148, 0.5066397
2: -4.7570267, -3.9528465, -4.7581606, -3.9498084, -0.5182986, 0.5118359
3: -12.5672779, -11.2399206, -12.5681324, -11.2307339, -0.7392625, 0.7141809
4: -2.1786318, -1.1251631, -2.1798697, -1.1166854, -0.7446434, 0.7397647
5: -9.8548985, -8.8744822, -9.8751030, -8.8739614, -0.5361981, 0.5511305
6: -7.8242044, -6.6131854, -7.8397212, -6.6124911, -0.8137004, 0.8135207
7: -2.6614099, -2.0566278, -2.6614096, -2.0524991, -0.3709474, 0.3553048
8: -3.6503282, -2.6554461, -3.6517200, -2.6395006, -0.6245005, 0.5892648
9: -12.2851915, -11.2112846, -12.2942305, -11.2106867, -0.7114582, 0.7143719

Time for backsubstitution: 21.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 901

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 6155

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2809228, upper bound: 0.2809260
time: 3.59 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2809228, upper bound: 0.2842738
time: 3.37 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -12.2583714, -10.6897888, -12.1896629, -10.7202034, -1.0579276, 1.0227966
1: 3.3828988, 4.2843328, 3.4026523, 4.2648249, -0.5055165, 0.4986627
2: -4.7623825, -3.9558299, -4.7339582, -3.9685297, -0.5209270, 0.4991589
3: -12.5556812, -11.2228918, -12.5080109, -11.2808943, -0.7033358, 0.7327273
4: -2.1958172, -1.1087689, -2.1697073, -1.1135020, -0.7540307, 0.7401249
5: -9.8890352, -8.8325443, -9.8783865, -8.8916836, -0.5374783, 0.5738592
6: -7.8518400, -6.6005487, -7.8049846, -6.6457291, -0.8079140, 0.7913165
7: -2.6675797, -2.0493274, -2.6601470, -2.0528302, -0.3722919, 0.3600174
8: -3.6838241, -2.6387987, -3.6220832, -2.6548786, -0.6333432, 0.5812619
9: -12.3001451, -11.2000265, -12.2878551, -11.2276707, -0.7108653, 0.7192196

Time for backsubstitution: 21.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 901

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 6193

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2749487, upper bound: 0.2837314
time: 4.06 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2749487, upper bound: 0.2840666
time: 3.54 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -12.1866255, -10.7345371, -12.2520752, -10.6977606, -1.0116467, 1.0373983
1: 3.3988485, 4.2742414, 3.3842869, 4.2721167, -0.4931984, 0.5043225
2: -4.7288733, -3.9790728, -4.7572722, -3.9638190, -0.4913538, 0.5131494
3: -12.5495996, -11.2586622, -12.5507984, -11.2253799, -0.7316476, 0.6771358
4: -2.1719847, -1.1271420, -2.1774104, -1.1171508, -0.7247384, 0.7292953
5: -9.8867216, -8.8416214, -9.8902388, -8.8737116, -0.5456873, 0.5621505
6: -7.8259220, -6.6135311, -7.8502789, -6.6256957, -0.7889061, 0.8063805
7: -2.6567488, -2.0650964, -2.6606448, -2.0565531, -0.3627348, 0.3604457
8: -3.6701880, -2.6396847, -3.6499891, -2.6300316, -0.6243336, 0.6034150
9: -12.2791767, -11.2217903, -12.3000975, -11.2224083, -0.6989739, 0.7157426

Time for backsubstitution: 21.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 901

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 6155

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2840711, upper bound: 0.2768093
time: 3.59 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2840716, upper bound: 0.2768092
time: 3.46 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -12.2632484, -10.6611967, -12.2602634, -10.6616306, -1.0839233, 1.0812058
1: 3.3802857, 4.2914324, 3.3816557, 4.2802582, -0.5150223, 0.5167177
2: -4.7639036, -3.9452143, -4.7594137, -3.9468162, -0.5251431, 0.5201611
3: -12.5861206, -11.2193727, -12.5689335, -11.2200289, -0.7536938, 0.7269466
4: -2.1983867, -1.1079764, -2.1814733, -1.1082971, -0.7618020, 0.7505064
5: -9.8970594, -8.8315611, -9.8950233, -8.8726826, -0.5538634, 0.5780995
6: -7.8568068, -6.5845466, -7.8550615, -6.6118588, -0.8307061, 0.8358381
7: -2.6678946, -2.0473113, -2.6614096, -2.0481265, -0.3777698, 0.3644973
8: -3.6862569, -2.6238050, -3.6533594, -2.6237974, -0.6504204, 0.6075468
9: -12.3055592, -11.1960573, -12.3033400, -11.2095919, -0.7303228, 0.7336134

Time for backsubstitution: 22.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 901

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 6155

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2842718, upper bound: 0.2809242
time: 4.20 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2842723, upper bound: 0.2809907
time: 3.67 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 30.13 seconds
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.13
Output dim: 1, lower bound: -0.2715560, upper bound: 0.2836858
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.13
Output dim: 1, lower bound: -0.2715560, upper bound: 0.2836876
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 30.13
Output dim: 1, lower bound: -0.2809228, upper bound: 0.2809260
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.13
Output dim: 1, lower bound: -0.2809228, upper bound: 0.2842738
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.13
Output dim: 1, lower bound: -0.2749487, upper bound: 0.2837314
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.13
Output dim: 1, lower bound: -0.2749487, upper bound: 0.2840666
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.13
Output dim: 1, lower bound: -0.2840711, upper bound: 0.2768093
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.13
Output dim: 1, lower bound: -0.2840716, upper bound: 0.2768092
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.13
Output dim: 1, lower bound: -0.2842718, upper bound: 0.2809242
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.13
Output dim: 1, lower bound: -0.2842723, upper bound: 0.2809907

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -12.2517281, -10.6947384, -12.1139164, -10.7958794, -0.9748194, 0.7280817
1: 3.3968043, 4.2719660, 3.4255977, 4.2473812, -0.4871731, 0.4625134
2: -4.7555065, -3.9634764, -4.6984801, -4.0049658, -0.4797288, 0.4153411
3: -12.5368376, -11.2434406, -12.4705381, -11.3292208, -0.4866037, 0.6719718
4: -2.1760945, -1.1259561, -2.1436648, -1.1410406, -0.7107584, 0.6854439
5: -9.8468666, -8.8754578, -9.8486919, -8.9024277, -0.4831612, 0.5311815
6: -7.8192229, -6.6291852, -7.7624292, -6.6744337, -0.7715044, 0.7673478
7: -2.6610951, -2.0586424, -2.6488917, -2.0731769, -0.3535470, 0.3421643
8: -3.6478968, -2.6704359, -3.6059823, -2.6868544, -0.5660919, 0.4293126
9: -12.2797785, -11.2152424, -12.2536736, -11.2498646, -0.7040744, 0.6248749

Time for backsubstitution: 22.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5815
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 5815

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2715560, upper bound: 0.2791350
time: 6.10 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2715559, upper bound: 0.2836852
time: 4.72 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -12.2517281, -10.6947384, -12.1878052, -10.7224722, -1.0494640, 0.9989944
1: 3.3968043, 4.2719660, 3.4088478, 4.2642498, -0.4950836, 0.4867139
2: -4.7555065, -3.9634764, -4.7327046, -3.9715374, -0.5122247, 0.4859059
3: -12.5368376, -11.2434406, -12.5072012, -11.2916050, -0.6951113, 0.7199558
4: -2.1760945, -1.1259561, -2.1681662, -1.1218953, -0.7368431, 0.7322655
5: -9.8468666, -8.8754578, -9.8584633, -8.8929520, -0.5267742, 0.5468860
6: -7.8192229, -6.6291852, -7.7897000, -6.6463723, -0.7837751, 0.7653296
7: -2.6610951, -2.0586424, -2.6601465, -2.0572062, -0.3533499, 0.3508104
8: -3.6478968, -2.6704359, -3.6204433, -2.6705856, -0.6074219, 0.5931612
9: -12.2797785, -11.2152424, -12.2787476, -11.2287598, -0.6834860, 0.6948359

Time for backsubstitution: 22.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5815
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 5815

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2715560, upper bound: 0.2794739
time: 4.01 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2715560, upper bound: 0.2840250
time: 3.83 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -12.2566090, -10.6661377, -12.2632504, -10.6611786, -1.0756886, 1.0769415
1: 3.3941908, 4.2790656, 3.3802853, 4.2914295, -0.5079248, 0.5086865
2: -4.7570267, -3.9528465, -4.7639027, -3.9452205, -0.5206466, 0.5143657
3: -12.5672779, -11.2399206, -12.5861006, -11.2193718, -0.7419982, 0.7158439
4: -2.1786318, -1.1251631, -2.1983869, -1.1079705, -0.7457082, 0.7450573
5: -9.8548985, -8.8744822, -9.8970566, -8.8315611, -0.5370795, 0.5543823
6: -7.8242044, -6.6131854, -7.8568058, -6.5845480, -0.8151610, 0.8168600
7: -2.6614099, -2.0566278, -2.6678941, -2.0473080, -0.3717302, 0.3600178
8: -3.6503282, -2.6554461, -3.6862555, -2.6238108, -0.6245005, 0.5936630
9: -12.2851915, -11.2112846, -12.3055515, -11.1960440, -0.7102299, 0.7175200

Time for backsubstitution: 21.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5815
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 5815

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2758109, upper bound: 0.2791614
time: 3.60 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2758109, upper bound: 0.2791629
time: 3.79 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -12.2583714, -10.6897888, -12.1157694, -10.7936630, -0.9832733, 0.7353148
1: 3.3828988, 4.2843328, 3.4194040, 4.2479515, -0.4954877, 0.4725118
2: -4.7623825, -3.9558299, -4.6997337, -4.0019684, -0.4865808, 0.4236787
3: -12.5556812, -11.2228918, -12.4713392, -11.3185081, -0.5010223, 0.6847433
4: -2.1958172, -1.1087689, -2.1452112, -1.1326528, -0.7279451, 0.6960840
5: -9.8890352, -8.8325443, -9.8686113, -8.9011555, -0.5007989, 0.5581521
6: -7.8518400, -6.6005487, -7.7777271, -6.6738014, -0.7886460, 0.7894661
7: -2.6675797, -2.0493274, -2.6488917, -2.0688152, -0.3603516, 0.3495684
8: -3.6838241, -2.6387987, -3.6076226, -2.6711512, -0.5920128, 0.4475929
9: -12.3001451, -11.2000265, -12.2627811, -11.2487993, -0.7229133, 0.6424797

Time for backsubstitution: 22.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5815
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 5815

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2749473, upper bound: 0.2791785
time: 3.85 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2749487, upper bound: 0.2837308
time: 3.75 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -12.2583714, -10.6897888, -12.1896610, -10.7202396, -1.0579267, 1.0062084
1: 3.3828988, 4.2843328, 3.4026546, 4.2648211, -0.5033027, 0.4967881
2: -4.7623825, -3.9558299, -4.7339582, -3.9685333, -0.5209262, 0.4942468
3: -12.5556812, -11.2228918, -12.5080051, -11.2808971, -0.7095290, 0.7327273
4: -2.1958172, -1.1087689, -2.1697049, -1.1135068, -0.7540298, 0.7429850
5: -9.8890352, -8.8325443, -9.8783855, -8.8916836, -0.5444270, 0.5738581
6: -7.8518400, -6.6005487, -7.8049812, -6.6457391, -0.8005893, 0.7876081
7: -2.6675797, -2.0493274, -2.6601465, -2.0528347, -0.3650297, 0.3600167
8: -3.6838241, -2.6387987, -3.6220818, -2.6548839, -0.6333427, 0.6114424
9: -12.3001451, -11.2000265, -12.2878561, -11.2276821, -0.7073302, 0.7162242

Time for backsubstitution: 22.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5815
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 5815

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2749487, upper bound: 0.2794968
time: 4.07 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2749487, upper bound: 0.2837314
time: 3.62 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -12.1866255, -10.7345371, -12.2484245, -10.7022352, -1.0054181, 1.0311456
1: 3.3988485, 4.2742414, 3.3968101, 4.2709208, -0.4908851, 0.4915230
2: -4.7288733, -3.9790728, -4.7548857, -3.9698548, -0.4850075, 0.5092043
3: -12.5495996, -11.2586622, -12.5491199, -11.2452755, -0.7126076, 0.6733878
4: -2.1719847, -1.1271420, -2.1745663, -1.1340120, -0.7077048, 0.7278585
5: -9.8867216, -8.8416214, -9.8501110, -8.8755131, -0.5447757, 0.5225625
6: -7.8259220, -6.6135311, -7.8194213, -6.6270204, -0.7842867, 0.7765431
7: -2.6567488, -2.0650964, -2.6606448, -2.0650406, -0.3523263, 0.3585347
8: -3.6701880, -2.6396847, -3.6469579, -2.6616840, -0.5926175, 0.5953270
9: -12.2791767, -11.2217903, -12.2819424, -11.2240772, -0.6850355, 0.6934881

Time for backsubstitution: 22.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5815
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 5815

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2788242, upper bound: 0.2715559
time: 3.63 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2788242, upper bound: 0.2715559
time: 3.97 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -12.1866255, -10.7345371, -12.2550640, -10.6973124, -1.0121527, 1.0407641
1: 3.3988485, 4.2742414, 3.3829241, 4.2832861, -0.4940085, 0.5062935
2: -4.7288733, -3.9790728, -4.7617607, -3.9622197, -0.4930675, 0.5147551
3: -12.5495996, -11.2586622, -12.5679646, -11.2247238, -0.7328632, 0.6787007
4: -2.1719847, -1.1271420, -2.1943357, -1.1168244, -0.7167351, 0.7349279
5: -9.8867216, -8.8416214, -9.8922768, -8.8325872, -0.5461556, 0.5400897
6: -7.8259220, -6.6135311, -7.8520083, -6.5983839, -0.7900519, 0.8107035
7: -2.6567488, -2.0650964, -2.6671305, -2.0557342, -0.3590572, 0.3604453
8: -3.6701880, -2.6396847, -3.6828861, -2.6300440, -0.6243336, 0.6063516
9: -12.2791767, -11.2217903, -12.3023100, -11.2088575, -0.6999931, 0.7111771

Time for backsubstitution: 22.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5815
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 5815

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2788260, upper bound: 0.2715559
time: 3.45 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2788260, upper bound: 0.2716212
time: 4.37 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -12.2632484, -10.6611967, -12.2566061, -10.6661186, -1.0776877, 1.0749435
1: 3.3802857, 4.2914324, 3.3941908, 4.2790613, -0.5126839, 0.5039386
2: -4.7639036, -3.9452143, -4.7570267, -3.9528513, -0.5187939, 0.5162181
3: -12.5861206, -11.2193727, -12.5672569, -11.2399206, -0.7346387, 0.7232058
4: -2.1983867, -1.1079764, -2.1786332, -1.1251566, -0.7447689, 0.7460213
5: -9.8970594, -8.8315611, -9.8548946, -8.8744822, -0.5529615, 0.5385053
6: -7.8568068, -6.5845466, -7.8242064, -6.6131859, -0.8260660, 0.8059764
7: -2.6678946, -2.0473113, -2.6614096, -2.0566247, -0.3673617, 0.3625971
8: -3.6862569, -2.6238050, -3.6503291, -2.6554499, -0.6187043, 0.5994594
9: -12.3055592, -11.1960573, -12.2851858, -11.2112741, -0.7163942, 0.7113589

Time for backsubstitution: 22.27 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.44 + 560.06 = 616.50 seconds
