## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.14146995799999998


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (6.6721687, 7.3738494, 6.6721687, 7.3738494, -0.2816482, 0.2816482)
1: (-12.1324730, -11.1050491, -12.1324730, -11.1050491, -0.3865297, 0.3865297)
2: (-2.5288877, -1.9969044, -2.5288877, -1.9969044, -0.3325222, 0.3325224)
3: (-10.6231375, -9.7707710, -10.6231375, -9.7707710, -0.4566672, 0.4566669)
4: (-6.4830627, -5.5876026, -6.4830627, -5.5876026, -0.4396453, 0.4396451)
5: (-8.3051805, -7.6046185, -8.3051805, -7.6046185, -0.3318262, 0.3318262)
6: (-3.3040340, -2.5555944, -3.3040340, -2.5555944, -0.3250239, 0.3250240)
7: (-10.1047630, -9.2861776, -10.1047630, -9.2861776, -0.2660110, 0.2660111)
8: (-2.0143652, -1.3882971, -2.0143652, -1.3882971, -0.3044350, 0.3044350)
9: (-3.0801594, -2.3345079, -3.0801594, -2.3345079, -0.3198457, 0.3198457)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 21.98 + 33.84 = 55.82 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.1443570, upper bound: 0.1443571

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4596
type: B, layer: 1, pos: 4596
type: A, layer: 1, pos: 6112
type: B, layer: 1, pos: 6112
type: A, layer: 1, pos: 469
type: B, layer: 1, pos: 469
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4596

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1443540, upper bound: 0.1424736
time: 2.98 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1443540, upper bound: 0.1443536
time: 4.06 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.26 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 7.26
Output dim: 0, lower bound: -0.1443540, upper bound: 0.1424736
NS_A2, status: Status.UNKNOWN, split count: 1, time: 7.26
Output dim: 0, lower bound: -0.1443540, upper bound: 0.1443536

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 6.6730919, 7.3656626, 6.6725993, 7.3698215, -0.2775550, 0.2734268
1: -12.1261539, -11.1076469, -12.1293640, -11.1062450, -0.3775806, 0.3807876
2: -2.5162001, -2.0002832, -2.5226436, -1.9984753, -0.3197110, 0.3260489
3: -10.6084776, -9.7748032, -10.6159286, -9.7726326, -0.4418390, 0.4491975
4: -6.4809051, -5.5883307, -6.4820023, -5.5879388, -0.4346256, 0.4351058
5: -8.2968674, -7.6060839, -8.3010883, -7.6052957, -0.3228159, 0.3269718
6: -3.3018732, -2.5633788, -3.3030381, -2.5594232, -0.3211207, 0.3171899
7: -10.1046171, -9.2878056, -10.1046944, -9.2869987, -0.2631614, 0.2628496
8: -2.0074844, -1.3901715, -2.0109792, -1.3892007, -0.2973757, 0.3006899
9: -3.0774653, -2.3349581, -3.0788345, -2.3347192, -0.3158644, 0.3172126

Time for backsubstitution: 20.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6112
type: B, layer: 1, pos: 6112
type: B, layer: 1, pos: 4596
type: A, layer: 1, pos: 469
type: B, layer: 1, pos: 469
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6112

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1443524, upper bound: 0.1413820
time: 3.05 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1443524, upper bound: 0.1424718
time: 3.05 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 6.6633687, 7.3738375, 6.6721697, 7.3738427, -0.2892545, 0.2799847
1: -12.1325092, -11.0990486, -12.1324625, -11.1050501, -0.3848910, 0.3934677
2: -2.5288482, -1.9838848, -2.5288675, -1.9969063, -0.3314018, 0.3385465
3: -10.6232271, -9.7561655, -10.6231203, -9.7707739, -0.4552307, 0.4666157
4: -6.4833403, -5.5868025, -6.4830618, -5.5876026, -0.4392786, 0.4405334
5: -8.3053093, -7.5968733, -8.3051682, -7.6046185, -0.3312364, 0.3381317
6: -3.3118382, -2.5555823, -3.3040318, -2.5556033, -0.3315060, 0.3242049
7: -10.1056242, -9.2860069, -10.1047649, -9.2861805, -0.2652807, 0.2660714
8: -2.0143452, -1.3813105, -2.0143547, -1.3882971, -0.3040035, 0.3103374
9: -3.0802166, -2.3320312, -3.0801549, -2.3345070, -0.3190397, 0.3228207

Time for backsubstitution: 21.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6112
type: B, layer: 1, pos: 6112
type: B, layer: 1, pos: 4596
type: A, layer: 1, pos: 469
type: B, layer: 1, pos: 469
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6112

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1443524, upper bound: 0.1432623
time: 2.96 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1443524, upper bound: 0.1443521
time: 3.10 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 27.82 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 27.82
Output dim: 0, lower bound: -0.1443524, upper bound: 0.1413820
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 27.82
Output dim: 0, lower bound: -0.1443524, upper bound: 0.1424718
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 27.82
Output dim: 0, lower bound: -0.1443524, upper bound: 0.1432623
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 27.82
Output dim: 0, lower bound: -0.1443524, upper bound: 0.1443521

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: 6.6731348, 7.3625135, 6.6726222, 7.3682656, -0.2759812, 0.2702591
1: -12.1229963, -11.1076498, -12.1278048, -11.1062469, -0.3744338, 0.3792360
2: -2.5161994, -2.0004659, -2.5226433, -1.9985647, -0.3187017, 0.3249693
3: -10.6083202, -9.7777939, -10.6158476, -9.7741117, -0.4402509, 0.4461005
4: -6.4757404, -5.5883384, -6.4794493, -5.5879421, -0.4292459, 0.4324496
5: -8.2967319, -7.6060858, -8.3010216, -7.6052942, -0.3217168, 0.3260148
6: -3.2986050, -2.5634181, -3.3014233, -2.5594442, -0.3178172, 0.3155403
7: -10.1045265, -9.2890320, -10.1046495, -9.2876053, -0.2624576, 0.2615129
8: -2.0074854, -1.3908653, -2.0109782, -1.3895421, -0.2970192, 0.2999680
9: -3.0773683, -2.3363810, -3.0787854, -2.3354177, -0.3151307, 0.3157883

Time for backsubstitution: 22.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4596
type: B, layer: 1, pos: 6112
type: A, layer: 1, pos: 469
type: B, layer: 1, pos: 469
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4596

## Relational analysis of NS_A1_A1_B1

### Relational analysis result of NS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1424724, upper bound: 0.1413820
time: 3.18 seconds

## Relational analysis of NS_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1424724, upper bound: 0.1413818
time: 4.85 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: 6.6696339, 7.3656683, 6.6726012, 7.3698177, -0.2810388, 0.2714489
1: -12.1262741, -11.1042004, -12.1293621, -11.1062450, -0.3757281, 0.3842428
2: -2.5163352, -1.9998889, -2.5226436, -1.9984753, -0.3201268, 0.3257754
3: -10.6122856, -9.7746363, -10.6159277, -9.7726355, -0.4457521, 0.4476438
4: -6.4814138, -5.5824199, -6.4819970, -5.5879388, -0.4321060, 0.4410398
5: -8.2973099, -7.6057773, -8.3010883, -7.6052957, -0.3224151, 0.3275294
6: -3.3018999, -2.5595603, -3.3030357, -2.5594244, -0.3190305, 0.3210393
7: -10.1062689, -9.2878094, -10.1046944, -9.2870007, -0.2648576, 0.2620763
8: -2.0083523, -1.3899498, -2.0109792, -1.3892012, -0.2982461, 0.3005826
9: -3.0792685, -2.3348284, -3.0788345, -2.3347189, -0.3177359, 0.3164711

Time for backsubstitution: 21.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4596
type: B, layer: 1, pos: 6112
type: B, layer: 1, pos: 469
type: A, layer: 1, pos: 469
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4596

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1424725, upper bound: 0.1424718
time: 3.09 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1424724, upper bound: 0.1424718
time: 2.89 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: 6.6634111, 7.3706865, 6.6721897, 7.3722873, -0.2871835, 0.2768171
1: -12.1293564, -11.0990486, -12.1309052, -11.1050491, -0.3817456, 0.3914298
2: -2.5288482, -1.9840658, -2.5288672, -1.9969952, -0.3303924, 0.3373673
3: -10.6230698, -9.7591572, -10.6230421, -9.7722483, -0.4536443, 0.4635191
4: -6.4781828, -5.5868106, -6.4805112, -5.5876064, -0.4339046, 0.4378779
5: -8.3051796, -7.5968742, -8.3051004, -7.6046209, -0.3301399, 0.3370843
6: -3.3085682, -2.5556235, -3.3024185, -2.5556235, -0.3282032, 0.3225552
7: -10.1055298, -9.2872295, -10.1047182, -9.2867870, -0.2645760, 0.2647347
8: -2.0143442, -1.3820028, -2.0143542, -1.3886414, -0.3036480, 0.3096160
9: -3.0801201, -2.3334541, -3.0801075, -2.3352098, -0.3183059, 0.3213964

Time for backsubstitution: 21.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4596
type: B, layer: 1, pos: 6112
type: A, layer: 1, pos: 469
type: B, layer: 1, pos: 469
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4596

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1424718, upper bound: 0.1432617
time: 3.00 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1424718, upper bound: 0.1432626
time: 3.00 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: 6.6599121, 7.3738422, 6.6721668, 7.3738403, -0.2893218, 0.2780066
1: -12.1326332, -11.0956020, -12.1324596, -11.1050501, -0.3830390, 0.3934690
2: -2.5289836, -1.9834847, -2.5288665, -1.9969060, -0.3318172, 0.3381171
3: -10.6270390, -9.7560024, -10.6231194, -9.7707729, -0.4591439, 0.4650614
4: -6.4838595, -5.5808921, -6.4830570, -5.5876036, -0.4367671, 0.4464676
5: -8.3057537, -7.5965648, -8.3051662, -7.6046190, -0.3308401, 0.3377764
6: -3.3118637, -2.5517619, -3.3040299, -2.5556040, -0.3294230, 0.3280549
7: -10.1072721, -9.2860088, -10.1047649, -9.2861805, -0.2669765, 0.2652979
8: -2.0152125, -1.3810883, -2.0143547, -1.3882980, -0.3048730, 0.3102314
9: -3.0820224, -2.3319016, -3.0801532, -2.3345091, -0.3209112, 0.3220792

Time for backsubstitution: 21.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4596
type: B, layer: 1, pos: 6112
type: B, layer: 1, pos: 469
type: A, layer: 1, pos: 469
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 4596

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1424718, upper bound: 0.1443515
time: 3.18 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1424718, upper bound: 0.1443523
time: 3.49 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 28.78 seconds
NS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 28.78
Output dim: 0, lower bound: -0.1424724, upper bound: 0.1413820
NS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 28.78
Output dim: 0, lower bound: -0.1424724, upper bound: 0.1413818
NS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 28.78
Output dim: 0, lower bound: -0.1424725, upper bound: 0.1424718
NS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 28.78
Output dim: 0, lower bound: -0.1424724, upper bound: 0.1424718
NS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 28.78
Output dim: 0, lower bound: -0.1424718, upper bound: 0.1432617
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 28.78
Output dim: 0, lower bound: -0.1424718, upper bound: 0.1432626
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 28.78
Output dim: 0, lower bound: -0.1424718, upper bound: 0.1443515
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 28.78
Output dim: 0, lower bound: -0.1424718, upper bound: 0.1443523

## BFS NS instance: NS_A1_A1_B1

### Backsubstitution after applying NS history:
0: 6.6731348, 7.3625135, 6.6731148, 7.3641076, -0.2718203, 0.2702274
1: -12.1229963, -11.1076498, -12.1245918, -11.1076469, -0.3735366, 0.3751314
2: -2.5161994, -2.0004659, -2.5161989, -2.0003738, -0.3185856, 0.3185151
3: -10.6083202, -9.7777939, -10.6084013, -9.7762794, -0.4401312, 0.4386210
4: -6.4757404, -5.5883384, -6.4783511, -5.5883350, -0.4278564, 0.4305782
5: -8.2967319, -7.6060858, -8.2967987, -7.6060839, -0.3214293, 0.3215709
6: -3.2986050, -2.5634181, -3.3002605, -2.5633984, -0.3138528, 0.3155067
7: -10.1045265, -9.2890320, -10.1045723, -9.2884140, -0.2612770, 0.2606442
8: -2.0074854, -1.3908653, -2.0074840, -1.3905158, -0.2968349, 0.2964692
9: -3.0773683, -2.3363810, -3.0774164, -2.3356595, -0.3146781, 0.3139877

Time for backsubstitution: 22.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6112
type: B, layer: 1, pos: 469
type: A, layer: 1, pos: 469
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 6112

## Relational analysis of NS_A1_A1_B1_B1

### Relational analysis result of NS_A1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1413816, upper bound: 0.1413820
time: 3.35 seconds

## Relational analysis of NS_A1_A1_B1_B2

### Relational analysis result of NS_A1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1413816, upper bound: 0.1413820
time: 3.23 seconds

## BFS NS instance: NS_A1_A1_B2

### Backsubstitution after applying NS history:
0: 6.6731348, 7.3625135, 6.6633897, 7.3722839, -0.2800038, 0.2778771
1: -12.1229963, -11.1076498, -12.1309509, -11.0990467, -0.3803320, 0.3815773
2: -2.5161994, -2.0004659, -2.5288479, -1.9839742, -0.3247302, 0.3260653
3: -10.6083202, -9.7777939, -10.6231499, -9.7576437, -0.4498386, 0.4510314
4: -6.4757404, -5.5883384, -6.4807901, -5.5868082, -0.4301395, 0.4339032
5: -8.2967319, -7.6060858, -8.3052387, -7.5968723, -0.3281982, 0.3301857
6: -3.2986050, -2.5634181, -3.3102226, -2.5556023, -0.3216529, 0.3215292
7: -10.1045265, -9.2890320, -10.1055765, -9.2866144, -0.2628794, 0.2616258
8: -2.0074854, -1.3908653, -2.0143442, -1.3816528, -0.3029923, 0.3032286
9: -3.0773683, -2.3363810, -3.0801687, -2.3327336, -0.3173490, 0.3167272

Time for backsubstitution: 22.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6112
type: B, layer: 1, pos: 469
type: A, layer: 1, pos: 469
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6112

## Relational analysis of NS_A1_A1_B2_B1

### Relational analysis result of NS_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1413816, upper bound: 0.1413818
time: 5.41 seconds

## Relational analysis of NS_A1_A1_B2_B2

### Relational analysis result of NS_A1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1413816, upper bound: 0.1413819
time: 4.53 seconds

## BFS NS instance: NS_A1_A2_B1

### Backsubstitution after applying NS history:
0: 6.6696339, 7.3656683, 6.6730900, 7.3656597, -0.2768776, 0.2714167
1: -12.1262741, -11.1042004, -12.1261482, -11.1076469, -0.3748305, 0.3801382
2: -2.5163352, -1.9998889, -2.5161996, -2.0002837, -0.3200114, 0.3193216
3: -10.6122856, -9.7746363, -10.6084766, -9.7748032, -0.4456310, 0.4401646
4: -6.4814138, -5.5824199, -6.4808998, -5.5883307, -0.4307160, 0.4391687
5: -8.2973099, -7.6057773, -8.2968683, -7.6060839, -0.3221281, 0.3230855
6: -3.3018999, -2.5595603, -3.3018727, -2.5633788, -0.3150673, 0.3210061
7: -10.1062689, -9.2878094, -10.1046190, -9.2878075, -0.2636771, 0.2612071
8: -2.0083523, -1.3899498, -2.0074844, -1.3901739, -0.2980611, 0.2970839
9: -3.0792685, -2.3348284, -3.0774646, -2.3349600, -0.3172835, 0.3146708

Time for backsubstitution: 22.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6112
type: B, layer: 1, pos: 469
type: A, layer: 1, pos: 469
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6112

## Relational analysis of NS_A1_A2_B1_B1

### Relational analysis result of NS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1413816, upper bound: 0.1424715
time: 3.16 seconds

## Relational analysis of NS_A1_A2_B1_B2

### Relational analysis result of NS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1413816, upper bound: 0.1424718
time: 3.56 seconds

## BFS NS instance: NS_A1_A2_B2

### Backsubstitution after applying NS history:
0: 6.6696339, 7.3656683, 6.6633692, 7.3738337, -0.2826165, 0.2790734
1: -12.1262741, -11.1042004, -12.1325035, -11.0990486, -0.3816259, 0.3853195
2: -2.5163352, -1.9998889, -2.5288482, -1.9838841, -0.3254075, 0.3268106
3: -10.6122856, -9.7746363, -10.6232290, -9.7561665, -0.4521244, 0.4525721
4: -6.4814138, -5.5824199, -6.4833384, -5.5868025, -0.4329991, 0.4424920
5: -8.2973099, -7.6057773, -8.3053083, -7.5968733, -0.3289783, 0.3308787
6: -3.3018999, -2.5595603, -3.3118339, -2.5555823, -0.3228729, 0.3238075
7: -10.1062689, -9.2878094, -10.1056242, -9.2860069, -0.2652794, 0.2621894
8: -2.0083523, -1.3899498, -2.0143452, -1.3813119, -0.3034409, 0.3038440
9: -3.0792685, -2.3348284, -3.0802166, -2.3320327, -0.3199542, 0.3174100

Time for backsubstitution: 21.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6112
type: B, layer: 1, pos: 469
type: A, layer: 1, pos: 469
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6112

## Relational analysis of NS_A1_A2_B2_B1

### Relational analysis result of NS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1413816, upper bound: 0.1424714
time: 5.80 seconds

## Relational analysis of NS_A1_A2_B2_B2

### Relational analysis result of NS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1413816, upper bound: 0.1424718
time: 4.28 seconds

## BFS NS instance: NS_A2_A1_B1

### Backsubstitution after applying NS history:
0: 6.6634111, 7.3706865, 6.6731148, 7.3641076, -0.2789795, 0.2784107
1: -12.1293564, -11.0990486, -12.1245918, -11.1076469, -0.3799829, 0.3819275
2: -2.5288482, -1.9840658, -2.5161989, -2.0003738, -0.3261580, 0.3246374
3: -10.6230698, -9.7591572, -10.6084013, -9.7762794, -0.4520965, 0.4487753
4: -6.4781828, -5.5868106, -6.4783511, -5.5883350, -0.4311838, 0.4328613
5: -8.3051796, -7.5968742, -8.2967987, -7.6060839, -0.3300431, 0.3283420
6: -3.3085682, -2.5556235, -3.3002605, -2.5633984, -0.3203881, 0.3227949
7: -10.1055298, -9.2872295, -10.1045723, -9.2884140, -0.2622585, 0.2622465
8: -2.0143442, -1.3820028, -2.0074840, -1.3905158, -0.3035064, 0.3027151
9: -3.0801201, -2.3334541, -3.0774164, -2.3356595, -0.3174175, 0.3166580

Time for backsubstitution: 21.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6112
type: B, layer: 1, pos: 469
type: A, layer: 1, pos: 469
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 6112

## Relational analysis of NS_A2_A1_B1_B1

### Relational analysis result of NS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1413810, upper bound: 0.1432616
time: 3.50 seconds

## Relational analysis of NS_A2_A1_B1_B2

### Relational analysis result of NS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1413810, upper bound: 0.1432615
time: 3.93 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: 6.6634111, 7.3706865, 6.6633897, 7.3722839, -0.2846351, 0.2830422
1: -12.1293564, -11.0990486, -12.1309509, -11.0990467, -0.3909287, 0.3914883
2: -2.5288482, -1.9840658, -2.5288479, -1.9839742, -0.3363583, 0.3373675
3: -10.6230698, -9.7591572, -10.6231499, -9.7576437, -0.4631608, 0.4635720
4: -6.4781828, -5.5868106, -6.4807901, -5.5868082, -0.4357095, 0.4384296
5: -8.3051796, -7.5968742, -8.3052387, -7.5968723, -0.3363740, 0.3366895
6: -3.3085682, -2.5556235, -3.3102226, -2.5556023, -0.3279045, 0.3285360
7: -10.1055298, -9.2872295, -10.1055765, -9.2866144, -0.2654121, 0.2647789
8: -2.0143442, -1.3820028, -2.0143442, -1.3816528, -0.3094718, 0.3096157
9: -3.0801201, -2.3334541, -3.0801687, -2.3327336, -0.3221036, 0.3214133

Time for backsubstitution: 22.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6112
type: B, layer: 1, pos: 469
type: A, layer: 1, pos: 469
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 6112

## Relational analysis of NS_A2_A1_B2_B1

### Relational analysis result of NS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1413810, upper bound: 0.1432624
time: 5.86 seconds

## Relational analysis of NS_A2_A1_B2_B2

### Relational analysis result of NS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1413810, upper bound: 0.1432626
time: 3.31 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: 6.6599121, 7.3738422, 6.6730900, 7.3656597, -0.2811173, 0.2796001
1: -12.1326332, -11.0956020, -12.1261482, -11.1076469, -0.3812768, 0.3847800
2: -2.5289836, -1.9834847, -2.5161996, -2.0002837, -0.3268373, 0.3253872
3: -10.6270390, -9.7560024, -10.6084766, -9.7748032, -0.4543827, 0.4503169
4: -6.4838595, -5.5808921, -6.4808998, -5.5883307, -0.4340467, 0.4414527
5: -8.3057537, -7.5965648, -8.2968683, -7.6060839, -0.3308227, 0.3290360
6: -3.3118637, -2.5517619, -3.3018727, -2.5633788, -0.3216074, 0.3250730
7: -10.1072721, -9.2860088, -10.1046190, -9.2878075, -0.2646587, 0.2628095
8: -2.0152125, -1.3810883, -2.0074844, -1.3901739, -0.3039556, 0.3033308
9: -3.0820224, -2.3319016, -3.0774646, -2.3349600, -0.3200229, 0.3173409

Time for backsubstitution: 21.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6112
type: B, layer: 1, pos: 469
type: A, layer: 1, pos: 469
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6112

## Relational analysis of NS_A2_A2_B1_B1

### Relational analysis result of NS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1413810, upper bound: 0.1443513
time: 3.15 seconds

## Relational analysis of NS_A2_A2_B1_B2

### Relational analysis result of NS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1413810, upper bound: 0.1443517
time: 3.07 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: 6.6599121, 7.3738422, 6.6633692, 7.3738337, -0.2893215, 0.2842317
1: -12.1326332, -11.0956020, -12.1325035, -11.0990486, -0.3922231, 0.3935273
2: -2.5289836, -1.9834847, -2.5288482, -1.9838841, -0.3370359, 0.3381171
3: -10.6270390, -9.7560024, -10.6232290, -9.7561665, -0.4654472, 0.4651136
4: -6.4838595, -5.5808921, -6.4833384, -5.5868025, -0.4385734, 0.4470177
5: -8.3057537, -7.5965648, -8.3053083, -7.5968733, -0.3371561, 0.3381474
6: -3.3118637, -2.5517619, -3.3118339, -2.5555823, -0.3291183, 0.3308141
7: -10.1072721, -9.2860088, -10.1056242, -9.2860069, -0.2678134, 0.2653420
8: -2.0152125, -1.3810883, -2.0143452, -1.3813119, -0.3099204, 0.3102316
9: -3.0820224, -2.3319016, -3.0802166, -2.3320327, -0.3246889, 0.3220959

Time for backsubstitution: 22.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6112
type: B, layer: 1, pos: 469
type: A, layer: 1, pos: 469
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 6112

## Relational analysis of NS_A2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1413810, upper bound: 0.1443520
time: 4.10 seconds

## Relational analysis of NS_A2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1413810, upper bound: 0.1443525
time: 3.85 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 30.33 seconds
NS_A1_A1_B1_B1, status: Status.VERIFIED, split count: 4, time: 30.33
Output dim: 0, lower bound: -0.1413816, upper bound: 0.1413820
NS_A1_A1_B1_B2, status: Status.VERIFIED, split count: 4, time: 30.33
Output dim: 0, lower bound: -0.1413816, upper bound: 0.1413820
NS_A1_A1_B2_B1, status: Status.VERIFIED, split count: 4, time: 30.33
Output dim: 0, lower bound: -0.1413816, upper bound: 0.1413818
NS_A1_A1_B2_B2, status: Status.VERIFIED, split count: 4, time: 30.33
Output dim: 0, lower bound: -0.1413816, upper bound: 0.1413819
NS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 30.33
Output dim: 0, lower bound: -0.1413816, upper bound: 0.1424715
NS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 30.33
Output dim: 0, lower bound: -0.1413816, upper bound: 0.1424718
NS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 30.33
Output dim: 0, lower bound: -0.1413816, upper bound: 0.1424714
NS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 30.33
Output dim: 0, lower bound: -0.1413816, upper bound: 0.1424718
NS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 30.33
Output dim: 0, lower bound: -0.1413810, upper bound: 0.1432616
NS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 30.33
Output dim: 0, lower bound: -0.1413810, upper bound: 0.1432615
NS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 30.33
Output dim: 0, lower bound: -0.1413810, upper bound: 0.1432624
NS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 30.33
Output dim: 0, lower bound: -0.1413810, upper bound: 0.1432626
NS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 30.33
Output dim: 0, lower bound: -0.1413810, upper bound: 0.1443513
NS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 30.33
Output dim: 0, lower bound: -0.1413810, upper bound: 0.1443517
NS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 30.33
Output dim: 0, lower bound: -0.1413810, upper bound: 0.1443520
NS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 30.33
Output dim: 0, lower bound: -0.1413810, upper bound: 0.1443525

## BFS NS instance: NS_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: 6.6696339, 7.3656683, 6.6731348, 7.3625135, -0.2737212, 0.2733803
1: -12.1262741, -11.1042004, -12.1229963, -11.1076498, -0.3768053, 0.3769934
2: -2.5163352, -1.9998889, -2.5161994, -2.0004659, -0.3183322, 0.3186915
3: -10.6122856, -9.7746363, -10.6083202, -9.7777939, -0.4425743, 0.4416878
4: -6.4814138, -5.5824199, -6.4757404, -5.5883384, -0.4333830, 0.4337950
5: -8.2973099, -7.6057773, -8.2967319, -7.6060858, -0.3215799, 0.3214638
6: -3.3018999, -2.5595603, -3.2986050, -2.5634181, -0.3171411, 0.3177165
7: -10.1062689, -9.2878094, -10.1045265, -9.2890320, -0.2623706, 0.2618467
8: -2.0083523, -1.3899498, -2.0074854, -1.3908653, -0.2973402, 0.2973449
9: -3.0792685, -2.3348284, -3.0773683, -2.3363810, -0.3158801, 0.3154948

Time for backsubstitution: 21.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 469
type: B, layer: 1, pos: 469
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 469

## Relational analysis of NS_A1_A2_B1_B1_A1

### Relational analysis result of NS_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1406168, upper bound: 0.1424714
time: 3.04 seconds

## Relational analysis of NS_A1_A2_B1_B1_A2

### Relational analysis result of NS_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1413803, upper bound: 0.1424712
time: 4.71 seconds

## BFS NS instance: NS_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: 6.6696339, 7.3656683, 6.6696339, 7.3656683, -0.2715502, 0.2715502
1: -12.1262741, -11.1042004, -12.1262741, -11.1042004, -0.3748894, 0.3748894
2: -2.5163352, -1.9998889, -2.5163352, -1.9998889, -0.3200629, 0.3200631
3: -10.6122856, -9.7746363, -10.6122856, -9.7746363, -0.4413764, 0.4413764
4: -6.4814138, -5.5824199, -6.4814138, -5.5824199, -0.4321446, 0.4321446
5: -8.2973099, -7.6057773, -8.2973099, -7.6057773, -0.3232334, 0.3232334
6: -3.3018999, -2.5595603, -3.3018999, -2.5595603, -0.3153734, 0.3153735
7: -10.1062689, -9.2878094, -10.1062689, -9.2878094, -0.2614677, 0.2614678
8: -2.0083523, -1.3899498, -2.0083523, -1.3899498, -0.2975123, 0.2975123
9: -3.0792685, -2.3348284, -3.0792685, -2.3348284, -0.3150795, 0.3150792

Time for backsubstitution: 21.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 469
type: B, layer: 1, pos: 469
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 469

## Relational analysis of NS_A1_A2_B1_B2_A1

### Relational analysis result of NS_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1406168, upper bound: 0.1424719
time: 3.34 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2

### Relational analysis result of NS_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1413803, upper bound: 0.1424716
time: 5.38 seconds

## BFS NS instance: NS_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: 6.6696339, 7.3656683, 6.6634111, 7.3706865, -0.2794514, 0.2789931
1: -12.1262741, -11.1042004, -12.1293564, -11.0990486, -0.3829061, 0.3821733
2: -2.5163352, -1.9998889, -2.5288482, -1.9840658, -0.3244181, 0.3261793
3: -10.6122856, -9.7746363, -10.6230698, -9.7591572, -0.4490640, 0.4522305
4: -6.4814138, -5.5824199, -6.4781828, -5.5868106, -0.4356661, 0.4371219
5: -8.2973099, -7.6057773, -8.3051796, -7.5968742, -0.3284252, 0.3298531
6: -3.3018999, -2.5595603, -3.3085682, -2.5556235, -0.3228176, 0.3205172
7: -10.1062689, -9.2878094, -10.1055298, -9.2872295, -0.2639731, 0.2628281
8: -2.0083523, -1.3899498, -2.0143442, -1.3820028, -0.3027197, 0.3037410
9: -3.0792685, -2.3348284, -3.0801201, -2.3334541, -0.3185503, 0.3182342

Time for backsubstitution: 22.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 469
type: B, layer: 1, pos: 469
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 469

## Relational analysis of NS_A1_A2_B2_B1_A1

### Relational analysis result of NS_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1424959, upper bound: 0.1424708
time: 3.32 seconds

## Relational analysis of NS_A1_A2_B2_B1_A2

### Relational analysis result of NS_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1432594, upper bound: 0.1424708
time: 3.27 seconds

## BFS NS instance: NS_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: 6.6696339, 7.3656683, 6.6599121, 7.3738422, -0.2797337, 0.2791402
1: -12.1262741, -11.1042004, -12.1326332, -11.0956020, -0.3816850, 0.3813355
2: -2.5163352, -1.9998889, -2.5289836, -1.9834847, -0.3254614, 0.3272955
3: -10.6122856, -9.7746363, -10.6270390, -9.7560024, -0.4521687, 0.4528260
4: -6.4814138, -5.5824199, -6.4838595, -5.5808921, -0.4344277, 0.4354749
5: -8.2973099, -7.6057773, -8.3057537, -7.5965648, -0.3295279, 0.3310165
6: -3.3018999, -2.5595603, -3.3118637, -2.5517619, -0.3229890, 0.3219160
7: -10.1062689, -9.2878094, -10.1072721, -9.2860088, -0.2630703, 0.2624495
8: -2.0083523, -1.3899498, -2.0152125, -1.3810883, -0.3035965, 0.3038481
9: -3.0792685, -2.3348284, -3.0820224, -2.3319016, -0.3177497, 0.3178189

Time for backsubstitution: 22.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 469
type: B, layer: 1, pos: 469
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 469

## Relational analysis of NS_A1_A2_B2_B2_A1

### Relational analysis result of NS_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1424959, upper bound: 0.1424712
time: 3.21 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2

### Relational analysis result of NS_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1432594, upper bound: 0.1424709
time: 7.25 seconds

## BFS NS instance: NS_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: 6.6634111, 7.3706865, 6.6731348, 7.3625135, -0.2778687, 0.2784021
1: -12.1293564, -11.0990486, -12.1229963, -11.1076498, -0.3799820, 0.3803315
2: -2.5288482, -1.9840658, -2.5161994, -2.0004659, -0.3258438, 0.3244162
3: -10.6230698, -9.7591572, -10.6083202, -9.7777939, -0.4509947, 0.4487379
4: -6.4781828, -5.5868106, -6.4757404, -5.5883384, -0.4311800, 0.4301362
5: -8.3051796, -7.5968742, -8.2967319, -7.6060858, -0.3298502, 0.3280060
6: -3.3085682, -2.5556235, -3.2986050, -2.5634181, -0.3203738, 0.3216393
7: -10.1055298, -9.2872295, -10.1045265, -9.2890320, -0.2615955, 0.2622163
8: -2.0143442, -1.3820028, -2.0074854, -1.3908653, -0.3032284, 0.3027148
9: -3.0801201, -2.3334541, -3.0773683, -2.3363810, -0.3167057, 0.3166368

Time for backsubstitution: 22.16 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 55.82 + 565.72 = 621.54 seconds
