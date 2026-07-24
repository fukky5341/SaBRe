## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.15950791595
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.7245874, 2.7245874)
1: (-7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764)
2: (-6.1131477, -4.0248523, -6.1131477, -4.0248523, -2.0882955, 2.0882955)
3: (-6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959)
4: (-6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728)
5: (-6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860)
6: (-11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779)
7: (2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937)
8: (-4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.3597069, 2.3597069)
9: (-2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474)

## BASE Result
execution time: IAR + LP analysis = 13.91 + 33.22 = 47.13 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -1.6885415, upper bound: 1.6885395


# Binary Search by BASE starts (time budget: 3552.87 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.0716936588287354
rel_dist={7: [-1.3781560099392798, 1.3781554825854307]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=1.974029779434204
rel_dist={7: [-1.163764129436149, 1.1637638135586528]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=1.9068553447723389
rel_dist={7: [-0.9954076049845186, 0.9954053351782006]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=1.9404423236846924
rel_dist={7: [-1.0863711031322327, 1.086370443966616]}

## Binary Search Result
Binary search time: 196.73 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual_ind) starts
Time budget: 3356.14 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 484
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 466

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4397735, upper bound: 1.4304536
time: 4.17 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4397735, upper bound: 1.4397728
time: 4.10 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8.46 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 8.46
Output dim: 7, lower bound: -1.4397735, upper bound: 1.4304536
IS_A2, status: Status.UNKNOWN, split count: 1, time: 8.46
Output dim: 7, lower bound: -1.4397735, upper bound: 1.4397728

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -7.1095839, -4.4137192, -7.1165104, -4.3992352, -2.6222820, 2.6134820
1: -7.2805634, -5.0668259, -7.2985964, -5.0622454, -2.2183180, 2.2317705
2: -6.0828629, -4.0292816, -6.1053562, -4.0259476, -1.8863406, 1.9064281
3: -6.1397429, -3.5712752, -6.1596360, -3.5659003, -2.5738425, 2.5883608
4: -6.4862461, -4.0831118, -6.4929533, -4.0640020, -2.4222441, 2.4098415
5: -6.5159121, -4.3084583, -6.5210662, -4.3032651, -2.2126470, 2.2126079
6: -11.4689255, -8.7266827, -11.4801722, -8.7087135, -2.7602119, 2.7534895
7: 2.7540634, 4.7761955, 2.7493360, 4.8082967, -2.0542333, 2.0268595
8: -4.3826246, -2.0528831, -4.4007831, -2.0487394, -2.2366147, 2.2526460
9: -2.7611513, -1.0621337, -2.7847159, -1.0571579, -1.7039934, 1.7225822

Time for backsubstitution: 14.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6178
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6178

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4384763, upper bound: 1.4203533
time: 4.09 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4397645, upper bound: 1.4304438
time: 4.07 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -7.1395717, -4.3879652, -7.1188116, -4.3942299, -2.6575646, 2.6409068
1: -7.3117166, -5.0573659, -7.3048601, -5.0606899, -2.2510266, 2.2474942
2: -6.1161394, -3.9974980, -6.1131439, -4.0248532, -1.9210391, 1.9469953
3: -6.1823025, -3.5602818, -6.1664610, -3.5639718, -2.6183307, 2.6061792
4: -6.5086150, -4.0486188, -6.4951229, -4.0573568, -2.4512582, 2.4465041
5: -6.5314770, -4.2962823, -6.5228043, -4.3014207, -2.2300563, 2.2265220
6: -11.5104237, -8.7007751, -11.4839725, -8.7025032, -2.8079205, 2.7831974
7: 2.7215409, 4.8289042, 2.7477772, 4.8194575, -2.0979166, 2.0811269
8: -4.4139233, -2.0308938, -4.4071074, -2.0474067, -2.2702231, 2.2874858
9: -2.7983041, -1.0427648, -2.7929342, -1.0555943, -1.7427098, 1.7501694

Time for backsubstitution: 14.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6178
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6178

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4384763, upper bound: 1.4295407
time: 4.08 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4397645, upper bound: 1.4397637
time: 4.03 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 22.56 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 22.56
Output dim: 7, lower bound: -1.4384763, upper bound: 1.4203533
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 22.56
Output dim: 7, lower bound: -1.4397645, upper bound: 1.4304438
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 22.56
Output dim: 7, lower bound: -1.4384763, upper bound: 1.4295407
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 22.56
Output dim: 7, lower bound: -1.4397645, upper bound: 1.4397637

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -7.1075058, -4.4142962, -7.1073470, -4.4020815, -2.5913157, 2.5752814
1: -7.2675619, -5.0686350, -7.2419939, -5.0705452, -2.1970167, 2.1733589
2: -6.0764112, -4.0306711, -6.0769548, -4.0322857, -1.8730278, 1.8766103
3: -6.1261911, -3.5729399, -6.1001792, -3.5736089, -2.5525823, 2.5272393
4: -6.4814434, -4.0844402, -6.4720173, -4.0699067, -2.4115367, 2.3875771
5: -6.5145559, -4.3120146, -6.5149708, -4.3179975, -2.1965585, 2.2029562
6: -11.4664211, -8.7273588, -11.4696989, -8.7118206, -2.7546005, 2.7423401
7: 2.7618763, 4.7745938, 2.7836185, 4.8018084, -2.0399320, 1.9909754
8: -4.3812647, -2.0584893, -4.3953733, -2.0734482, -2.2051129, 2.2399001
9: -2.7595549, -1.0666767, -2.7776461, -1.0773144, -1.6822405, 1.7109693

Time for backsubstitution: 14.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 484
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6178

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4295411, upper bound: 1.4203532
time: 4.36 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4295411, upper bound: 1.4203531
time: 4.22 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -7.1095819, -4.4137211, -7.1330585, -4.3808761, -2.6186914, 2.6457658
1: -7.2805586, -5.0668278, -7.3172588, -4.9779396, -2.3026190, 2.2504311
2: -6.0828533, -4.0292826, -6.1127005, -3.9822850, -1.9167435, 1.9097104
3: -6.1397257, -3.5712771, -6.1912374, -3.4924839, -2.6472418, 2.6199603
4: -6.4862423, -4.0831132, -6.5035505, -4.0347729, -2.4514694, 2.4204373
5: -6.5159101, -4.3084612, -6.5634313, -4.2917161, -2.2241940, 2.2549701
6: -11.4689226, -8.7266836, -11.5041847, -8.6795559, -2.7893667, 2.7775011
7: 2.7540696, 4.7761941, 2.7368786, 4.8328104, -2.0787408, 2.0393155
8: -4.3826246, -2.0528975, -4.4459128, -2.0405588, -2.2408924, 2.2898016
9: -2.7611499, -1.0621386, -2.8175559, -1.0520136, -1.7091362, 1.7554173

Time for backsubstitution: 14.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 484
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 484

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4287673, upper bound: 1.4290808
time: 4.17 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4397560, upper bound: 1.4304359
time: 4.05 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -7.1374698, -4.3885665, -7.1096125, -4.3970861, -2.6265550, 2.6026635
1: -7.2987614, -5.0592084, -7.2482710, -5.0689421, -2.2298193, 2.1890626
2: -6.1096973, -3.9989083, -6.0847449, -4.0312195, -1.9077182, 1.9170845
3: -6.1688304, -3.5619378, -6.1070814, -3.5716782, -2.5971522, 2.5451436
4: -6.5037889, -4.0499392, -6.4741917, -4.0632706, -2.4405184, 2.4242525
5: -6.5301013, -4.2998447, -6.5166931, -4.3161211, -2.2139802, 2.2168484
6: -11.5079632, -8.7014627, -11.4735203, -8.7056122, -2.8023510, 2.7720575
7: 2.7293704, 4.8273215, 2.7820554, 4.8129535, -2.0835831, 2.0452662
8: -4.4125814, -2.0365047, -4.4016571, -2.0721107, -2.2387376, 2.2747388
9: -2.7967238, -1.0473062, -2.7858524, -1.0757493, -1.7209746, 1.7385463

Time for backsubstitution: 14.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 484
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6178

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4295411, upper bound: 1.4295410
time: 4.15 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4295411, upper bound: 1.4295425
time: 4.59 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -7.1395707, -4.3879652, -7.1353798, -4.3758745, -2.6539936, 2.6731181
1: -7.3117118, -5.0573664, -7.3234844, -4.9762707, -2.3354411, 2.2661180
2: -6.1161289, -3.9974985, -6.1204910, -3.9811733, -1.9515581, 1.9502292
3: -6.1822863, -3.5602820, -6.1979280, -3.4905536, -2.6917326, 2.6376460
4: -6.5086107, -4.0486197, -6.5056705, -4.0280910, -2.4805198, 2.4570508
5: -6.5314751, -4.2962847, -6.5651608, -4.2898512, -2.2416239, 2.2688761
6: -11.5104218, -8.7007751, -11.5080681, -8.6733398, -2.8370819, 2.8072929
7: 2.7215462, 4.8289037, 2.7353396, 4.8440409, -2.1224947, 2.0935640
8: -4.4139209, -2.0309086, -4.4523387, -2.0392413, -2.2745023, 2.3125732
9: -2.7983015, -1.0427696, -2.8258305, -1.0504498, -1.7478516, 1.7830609

Time for backsubstitution: 14.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 484
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 484

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4287672, upper bound: 1.4386577
time: 3.89 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4397560, upper bound: 1.4397557
time: 4.52 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 22.88 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 22.88
Output dim: 7, lower bound: -1.4295411, upper bound: 1.4203532
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 22.88
Output dim: 7, lower bound: -1.4295411, upper bound: 1.4203531
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 22.88
Output dim: 7, lower bound: -1.4287673, upper bound: 1.4290808
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.88
Output dim: 7, lower bound: -1.4397560, upper bound: 1.4304359
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 22.88
Output dim: 7, lower bound: -1.4295411, upper bound: 1.4295410
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 22.88
Output dim: 7, lower bound: -1.4295411, upper bound: 1.4295425
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 22.88
Output dim: 7, lower bound: -1.4287672, upper bound: 1.4386577
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.88
Output dim: 7, lower bound: -1.4397560, upper bound: 1.4397557

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7.1004982, -4.4165354, -7.1073470, -4.4020815, -2.5661521, 2.5570979
1: -7.2239199, -5.0750461, -7.2419939, -5.0705452, -2.1533747, 2.1669478
2: -6.0544543, -4.0355406, -6.0769548, -4.0322857, -1.8512621, 1.8714240
3: -6.0800323, -3.5789900, -6.1001792, -3.5736089, -2.5064235, 2.5211892
4: -6.4652915, -4.0889955, -6.4720173, -4.0699067, -2.3953848, 2.3830218
5: -6.5098515, -4.3232841, -6.5149708, -4.3179975, -2.1918540, 2.1916866
6: -11.4582014, -8.7297783, -11.4696989, -8.7118206, -2.7463808, 2.7399206
7: 2.7883801, 4.7697563, 2.7836185, 4.8018084, -2.0134282, 1.9861379
8: -4.3772411, -2.0776005, -4.3953733, -2.0734482, -2.2006497, 2.2166591
9: -2.7541146, -1.0822948, -2.7776461, -1.0773144, -1.6768003, 1.6953512

Time for backsubstitution: 14.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4203608, upper bound: 1.4203533
time: 4.61 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4203608, upper bound: 1.4203530
time: 5.05 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.1260247, -4.3957896, -7.1073470, -4.4020815, -2.5916176, 2.5765915
1: -7.2993093, -4.9826632, -7.2419939, -5.0705452, -2.2287641, 2.2593307
2: -6.0901985, -3.9857919, -6.0769548, -4.0322857, -1.8874283, 1.9039218
3: -6.1716847, -3.4985342, -6.1001792, -3.5736089, -2.5980759, 2.6016450
4: -6.4969745, -4.0540309, -6.4720173, -4.0699067, -2.4270678, 2.4179864
5: -6.5582848, -4.2971525, -6.5149708, -4.3179975, -2.2402873, 2.2178183
6: -11.4922695, -8.6975517, -11.4696989, -8.7118206, -2.7804489, 2.7721472
7: 2.7416191, 4.7999220, 2.7836185, 4.8018084, -2.0601892, 2.0163035
8: -4.4265437, -2.0446544, -4.3953733, -2.0734482, -2.2454000, 2.2489219
9: -2.7932529, -1.0570203, -2.7776461, -1.0773144, -1.7159386, 1.7206258

Time for backsubstitution: 14.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4203608, upper bound: 1.4203531
time: 4.60 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4203608, upper bound: 1.4203531
time: 4.53 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.0874295, -4.4695120, -7.1282406, -4.3941245, -2.6133318, 2.4968345
1: -7.2669110, -5.0985832, -7.3142543, -4.9855986, -2.2178748, 2.2156711
2: -6.0739012, -4.0397863, -6.1108217, -3.9848199, -1.8253565, 1.8880706
3: -6.1154642, -3.6083899, -6.1853523, -3.5009282, -2.6145360, 2.5769625
4: -6.4708724, -4.1351161, -6.5002861, -4.0470839, -2.4237885, 2.3651700
5: -6.4900002, -4.3296204, -6.5568094, -4.2967477, -2.1932526, 2.2271891
6: -11.4296236, -8.7332878, -11.4949875, -8.6810904, -2.7485332, 2.7450790
7: 2.7827232, 4.7659335, 2.7436993, 4.8305326, -2.0478094, 2.0222342
8: -4.3769441, -2.0667491, -4.4446945, -2.0438251, -2.2153606, 2.2615080
9: -2.7491832, -1.0777196, -2.8141997, -1.0557536, -1.6934296, 1.7364801

Time for backsubstitution: 14.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4191108, upper bound: 1.4290808
time: 4.45 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4191109, upper bound: 1.4290805
time: 4.19 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.1739674, -4.4076023, -7.1330557, -4.3808846, -2.6650119, 2.6450300
1: -7.3089833, -5.0584164, -7.3172569, -4.9779472, -2.3310361, 2.2588406
2: -6.1151609, -4.0233784, -6.1126990, -3.9822879, -1.9393113, 1.9068170
3: -6.1835012, -3.5623147, -6.1912327, -3.4924951, -2.6910062, 2.6289179
4: -6.5303016, -4.0690274, -6.5035472, -4.0347848, -2.4955168, 2.4345198
5: -6.5301943, -4.2948461, -6.5634260, -4.2917204, -2.2384739, 2.2685800
6: -11.4851284, -8.7016354, -11.5041752, -8.6795578, -2.8055706, 2.8025398
7: 2.7370927, 4.7971935, 2.7368853, 4.8328085, -2.0957158, 2.0603082
8: -4.4010677, -2.0444937, -4.4459114, -2.0405617, -2.2730341, 2.2909379
9: -2.7744975, -1.0535952, -2.8175538, -1.0520163, -1.7224813, 1.7639586

Time for backsubstitution: 14.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4304389, upper bound: 1.4304362
time: 4.39 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4304389, upper bound: 1.4304359
time: 4.37 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -7.1302719, -4.3909101, -7.1096125, -4.3970861, -2.6010995, 2.5843866
1: -7.2549419, -5.0657506, -7.2482710, -5.0689421, -2.1859999, 2.1825204
2: -6.0877600, -4.0038648, -6.0847449, -4.0312195, -1.8859653, 1.9116821
3: -6.1228518, -3.5679712, -6.1070814, -3.5716782, -2.5511737, 2.5391102
4: -6.4875026, -4.0545297, -6.4741917, -4.0632706, -2.4242320, 2.4196620
5: -6.5252895, -4.3112545, -6.5166931, -4.3161211, -2.2091684, 2.2054386
6: -11.4998178, -8.7039356, -11.4735203, -8.7056122, -2.7942057, 2.7695847
7: 2.7559633, 4.8221107, 2.7820554, 4.8129535, -2.0569901, 2.0400553
8: -4.4081459, -2.0556450, -4.4016571, -2.0721107, -2.2338185, 2.2514660
9: -2.7911782, -1.0629236, -2.7858524, -1.0757493, -1.7154289, 1.7229289

Time for backsubstitution: 14.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4203579, upper bound: 1.4295408
time: 4.43 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4203579, upper bound: 1.4295409
time: 4.48 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.1559477, -4.3700466, -7.1096125, -4.3970861, -2.6267457, 2.6043508
1: -7.3301582, -4.9731579, -7.2482710, -5.0689421, -2.2612162, 2.2751131
2: -6.1234574, -3.9540708, -6.0847449, -4.0312195, -1.9220667, 1.9295466
3: -6.2140942, -3.4875479, -6.1070814, -3.5716782, -2.6424160, 2.6195335
4: -6.5191393, -4.0193810, -6.4741917, -4.0632706, -2.4558687, 2.4548106
5: -6.5737514, -4.2848196, -6.5166931, -4.3161211, -2.2576303, 2.2318735
6: -11.5340376, -8.6715927, -11.4735203, -8.7056122, -2.8284254, 2.8019276
7: 2.7091510, 4.8529577, 2.7820554, 4.8129535, -2.1038024, 2.0709023
8: -4.4579649, -2.0227385, -4.4016571, -2.0721107, -2.2795548, 2.2837243
9: -2.8306656, -1.0376561, -2.7858524, -1.0757493, -1.7549163, 1.7481964

Time for backsubstitution: 14.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4203579, upper bound: 1.4295406
time: 4.46 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4203579, upper bound: 1.4295411
time: 4.43 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -7.1172113, -4.4442253, -7.1305504, -4.3891187, -2.6482148, 2.5816240
1: -7.2980137, -5.0895939, -7.3204808, -4.9839268, -2.3140869, 2.2308869
2: -6.1071815, -4.0081663, -6.1186128, -3.9837093, -1.9246612, 1.9288862
3: -6.1579351, -3.5972159, -6.1920395, -3.4989913, -2.6589439, 2.5948236
4: -6.4931512, -4.1009483, -6.5024056, -4.0403938, -2.4527574, 2.4014573
5: -6.5044780, -4.3176546, -6.5585327, -4.2948823, -2.2095957, 2.2408781
6: -11.4710178, -8.7074261, -11.4988680, -8.6748714, -2.7961464, 2.7914419
7: 2.7505209, 4.8187766, 2.7421622, 4.8417664, -2.0912454, 2.0766144
8: -4.4083481, -2.0449057, -4.4511213, -2.0425091, -2.2508283, 2.2841568
9: -2.7855072, -1.0584073, -2.8224781, -1.0541898, -1.7313174, 1.7640707

Time for backsubstitution: 14.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4191080, upper bound: 1.4386575
time: 4.19 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4191079, upper bound: 1.4386579
time: 4.10 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.2037201, -4.3817492, -7.1353760, -4.3758831, -2.6890111, 2.6723239
1: -7.3401566, -5.0489388, -7.3234811, -4.9762793, -2.3638773, 2.2745423
2: -6.1484365, -3.9915972, -6.1204910, -3.9811764, -1.9741855, 1.9473963
3: -6.2259502, -3.5511432, -6.1979227, -3.4905658, -2.7353845, 2.6467795
4: -6.5526733, -4.0343285, -6.5056677, -4.0281034, -2.5245700, 2.4713392
5: -6.5455351, -4.2827592, -6.5651536, -4.2898560, -2.2556791, 2.2823944
6: -11.5267038, -8.6757145, -11.5080605, -8.6733418, -2.8533621, 2.8323460
7: 2.7045541, 4.8498664, 2.7353458, 4.8440394, -2.1394854, 2.1145205
8: -4.4324212, -2.0225267, -4.4523373, -2.0392437, -2.3063345, 2.3137412
9: -2.8115811, -1.0342251, -2.8258276, -1.0504520, -1.7611291, 1.7916025

Time for backsubstitution: 14.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4304360, upper bound: 1.4397556
time: 3.95 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4304360, upper bound: 1.4397560
time: 4.40 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 22.80 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 7, lower bound: -1.4203608, upper bound: 1.4203533
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 7, lower bound: -1.4203608, upper bound: 1.4203530
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 7, lower bound: -1.4203608, upper bound: 1.4203531
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 7, lower bound: -1.4203608, upper bound: 1.4203531
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 7, lower bound: -1.4191108, upper bound: 1.4290808
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 7, lower bound: -1.4191109, upper bound: 1.4290805
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 7, lower bound: -1.4304389, upper bound: 1.4304362
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 7, lower bound: -1.4304389, upper bound: 1.4304359
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 7, lower bound: -1.4203579, upper bound: 1.4295408
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 7, lower bound: -1.4203579, upper bound: 1.4295409
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 7, lower bound: -1.4203579, upper bound: 1.4295406
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 7, lower bound: -1.4203579, upper bound: 1.4295411
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 7, lower bound: -1.4191080, upper bound: 1.4386575
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 7, lower bound: -1.4191079, upper bound: 1.4386579
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 7, lower bound: -1.4304360, upper bound: 1.4397556
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 7, lower bound: -1.4304360, upper bound: 1.4397560

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -7.1004982, -4.4165354, -7.1004982, -4.4165354, -2.5507355, 2.5507352
1: -7.2239199, -5.0750461, -7.2239199, -5.0750461, -2.1488738, 2.1488738
2: -6.0544543, -4.0355406, -6.0544543, -4.0355406, -1.8469074, 1.8469076
3: -6.0800323, -3.5789900, -6.0800323, -3.5789900, -2.5010424, 2.5010424
4: -6.4652915, -4.0889955, -6.4652915, -4.0889955, -2.3762960, 2.3762960
5: -6.5098515, -4.3232841, -6.5098515, -4.3232841, -2.1865673, 2.1865673
6: -11.4582014, -8.7297783, -11.4582014, -8.7297783, -2.7284231, 2.7284231
7: 2.7883801, 4.7697563, 2.7883801, 4.7697563, -1.9813762, 1.9813762
8: -4.3772411, -2.0776005, -4.3772411, -2.0776005, -2.1948118, 2.1948121
9: -2.7541146, -1.0822948, -2.7541146, -1.0822948, -1.6718198, 1.6718198

Time for backsubstitution: 14.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 484
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 484

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4092064, upper bound: 1.4191963
time: 4.59 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4203523, upper bound: 1.4203502
time: 4.28 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -7.1004982, -4.4165354, -7.1302719, -4.3909101, -2.5764418, 2.5803490
1: -7.2239199, -5.0750461, -7.2549419, -5.0657506, -2.1581693, 2.1798959
2: -6.0544543, -4.0355406, -6.0877600, -4.0038648, -1.8786530, 1.8830175
3: -6.0800323, -3.5789900, -6.1228518, -3.5679712, -2.5120611, 2.5438619
4: -6.4652915, -4.0889955, -6.4875026, -4.0545297, -2.4107618, 2.3985071
5: -6.5098515, -4.3232841, -6.5252895, -4.3112545, -2.1985970, 2.2020054
6: -11.4582014, -8.7297783, -11.4998178, -8.7039356, -2.7542658, 2.7700396
7: 2.7883801, 4.7697563, 2.7559633, 4.8221107, -2.0337305, 2.0137930
8: -4.3772411, -2.0776005, -4.4081459, -2.0556450, -2.2160320, 2.2260299
9: -2.7541146, -1.0822948, -2.7911782, -1.0629236, -1.6911911, 1.7088834

Time for backsubstitution: 14.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 484
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 484

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4092065, upper bound: 1.4191956
time: 5.23 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4203523, upper bound: 1.4203495
time: 4.57 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.1260247, -4.3957896, -7.1004982, -4.4165354, -2.5762014, 2.5702291
1: -7.2993093, -4.9826632, -7.2239199, -5.0750461, -2.2242632, 2.2412567
2: -6.0901985, -3.9857919, -6.0544543, -4.0355406, -1.8830740, 1.8835273
3: -6.1716847, -3.4985342, -6.0800323, -3.5789900, -2.5926948, 2.5814981
4: -6.4969745, -4.0540309, -6.4652915, -4.0889955, -2.4079790, 2.4112606
5: -6.5582848, -4.2971525, -6.5098515, -4.3232841, -2.2350006, 2.2126989
6: -11.4922695, -8.6975517, -11.4582014, -8.7297783, -2.7624912, 2.7606497
7: 2.7416191, 4.7999220, 2.7883801, 4.7697563, -2.0281372, 2.0115418
8: -4.4265437, -2.0446544, -4.3772411, -2.0776005, -2.2402349, 2.2270751
9: -2.7932529, -1.0570203, -2.7541146, -1.0822948, -1.7109581, 1.6970943

Time for backsubstitution: 14.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 484
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 484

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4180279, upper bound: 1.4191914
time: 4.27 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4292395, upper bound: 1.4203453
time: 4.63 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.1260247, -4.3957896, -7.1302719, -4.3909101, -2.6019077, 2.5998430
1: -7.2993093, -4.9826632, -7.2549419, -5.0657506, -2.2335587, 2.2722788
2: -6.0901985, -3.9857919, -6.0877600, -4.0038648, -1.9085989, 1.9083488
3: -6.1716847, -3.4985342, -6.1228518, -3.5679712, -2.6037135, 2.6243176
4: -6.4969745, -4.0540309, -6.4875026, -4.0545297, -2.4424448, 2.4334717
5: -6.5582848, -4.2971525, -6.5252895, -4.3112545, -2.2470303, 2.2281370
6: -11.4922695, -8.6975517, -11.4998178, -8.7039356, -2.7883339, 2.8022661
7: 2.7416191, 4.7999220, 2.7559633, 4.8221107, -2.0804915, 2.0439587
8: -4.4265437, -2.0446544, -4.4081459, -2.0556450, -2.2501569, 2.2582929
9: -2.7932529, -1.0570203, -2.7911782, -1.0629236, -1.7303294, 1.7341579

Time for backsubstitution: 14.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 484
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 484

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4180279, upper bound: 1.4191911
time: 4.46 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4292395, upper bound: 1.4203448
time: 4.57 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -7.0874295, -4.4695120, -7.1212578, -4.4086094, -2.5979772, 2.4899750
1: -7.2669110, -5.0985832, -7.2963314, -4.9903088, -2.2026412, 2.1977482
2: -6.0739012, -4.0397863, -6.0883207, -3.9882071, -1.8193381, 1.8638408
3: -6.1154642, -3.6083899, -6.1658249, -3.5063295, -2.6091347, 2.5574350
4: -6.4708724, -4.1351161, -6.4937277, -4.0663462, -2.4045262, 2.3586116
5: -6.4900002, -4.3296204, -6.5516963, -4.3020115, -2.1826224, 2.2220759
6: -11.4296236, -8.7332878, -11.4832153, -8.6990757, -2.7305479, 2.7342129
7: 2.7827232, 4.7659335, 2.7483976, 4.7982149, -2.0154917, 2.0175359
8: -4.3769441, -2.0667491, -4.4262862, -2.0479188, -2.2096157, 2.2427306
9: -2.7491832, -1.0777196, -2.7904439, -1.0607300, -1.6884532, 1.7127243

Time for backsubstitution: 14.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6178

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4092015, upper bound: 1.4279971
time: 4.24 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4092015, upper bound: 1.4220907
time: 4.47 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.0874295, -4.4695120, -7.1511459, -4.3828568, -2.6241684, 2.5062461
1: -7.2669110, -5.0985832, -7.3272238, -4.9807863, -2.2131100, 2.2286406
2: -6.0739012, -4.0397863, -6.1215811, -3.9564888, -1.8324804, 1.8994648
3: -6.1154642, -3.6083899, -6.2082758, -3.4953034, -2.6201608, 2.5998859
4: -6.4708724, -4.1351161, -6.5158892, -4.0316582, -2.4392142, 2.3807731
5: -6.4900002, -4.3296204, -6.5670643, -4.2896719, -2.1994481, 2.2374439
6: -11.4296236, -8.7332878, -11.5249577, -8.6731119, -2.7565117, 2.7750669
7: 2.7827232, 4.7659335, 2.7159374, 4.8512778, -2.0685546, 2.0499961
8: -4.3769441, -2.0667491, -4.4577332, -2.0260148, -2.2308326, 2.2677572
9: -2.7491832, -1.0777196, -2.8278742, -1.0413669, -1.7078162, 1.7501546

Time for backsubstitution: 14.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6178

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4092015, upper bound: 1.4279966
time: 4.54 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4092015, upper bound: 1.4191904
time: 6.17 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.1739674, -4.4076023, -7.1260462, -4.3953643, -2.6520782, 2.6384940
1: -7.3089833, -5.0584164, -7.2993340, -4.9826484, -2.3263350, 2.2409177
2: -6.1151609, -4.0233784, -6.0901980, -3.9856782, -1.9346399, 1.8823235
3: -6.1835012, -3.5623147, -6.1717052, -3.4978774, -2.6856239, 2.6093905
4: -6.5303016, -4.0690274, -6.4969878, -4.0540156, -2.4762859, 2.4279604
5: -6.5301943, -4.2948461, -6.5582981, -4.2969809, -2.2332134, 2.2634521
6: -11.4851284, -8.7016354, -11.4923897, -8.6975412, -2.7875872, 2.7907543
7: 2.7370927, 4.7971935, 2.7415898, 4.8004980, -2.0634053, 2.0556037
8: -4.4010677, -2.0444937, -4.4275432, -2.0446548, -2.2671714, 2.2720311
9: -2.7744975, -1.0535952, -2.7938085, -1.0569927, -1.7175049, 1.7402133

Time for backsubstitution: 14.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6178

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4203475, upper bound: 1.4292375
time: 4.41 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4203475, upper bound: 1.4232646
time: 4.39 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.1739674, -4.4076023, -7.1559706, -4.3696213, -2.6717041, 2.6549187
1: -7.3089833, -5.0584164, -7.3301830, -4.9731426, -2.3358407, 2.2717667
2: -6.1151609, -4.0233784, -6.1234570, -3.9539552, -1.9476297, 1.9183683
3: -6.1835012, -3.5623147, -6.2141147, -3.4868898, -2.6966114, 2.6517999
4: -6.5303016, -4.0690274, -6.5191517, -4.0193667, -2.5109348, 2.4501243
5: -6.5301943, -4.2948461, -6.5737629, -4.2846479, -2.2455463, 2.2789168
6: -11.4851284, -8.7016354, -11.5341549, -8.6715841, -2.8135443, 2.8325195
7: 2.7370927, 4.7971935, 2.7091217, 4.8535328, -2.1164401, 2.0880718
8: -4.4010677, -2.0444937, -4.4589643, -2.0227389, -2.2852592, 2.2971060
9: -2.7744975, -1.0535952, -2.8312201, -1.0376278, -1.7368697, 1.7776250

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6178

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4203475, upper bound: 1.4292370
time: 4.69 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4203475, upper bound: 1.4203468
time: 4.16 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -7.1302719, -4.3909101, -7.1004982, -4.4165354, -2.5803490, 2.5764418
1: -7.2549419, -5.0657506, -7.2239199, -5.0750461, -2.1798959, 2.1581693
2: -6.0877600, -4.0038648, -6.0544543, -4.0355406, -1.8830178, 1.8786529
3: -6.1228518, -3.5679712, -6.0800323, -3.5789900, -2.5438619, 2.5120611
4: -6.4875026, -4.0545297, -6.4652915, -4.0889955, -2.3985071, 2.4107618
5: -6.5252895, -4.3112545, -6.5098515, -4.3232841, -2.2020054, 2.1985970
6: -11.4998178, -8.7039356, -11.4582014, -8.7297783, -2.7700396, 2.7542658
7: 2.7559633, 4.8221107, 2.7883801, 4.7697563, -2.0137930, 2.0337305
8: -4.4081459, -2.0556450, -4.3772411, -2.0776005, -2.2260294, 2.2160320
9: -2.7911782, -1.0629236, -2.7541146, -1.0822948, -1.7088834, 1.6911911

Time for backsubstitution: 14.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 484
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 484

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4092036, upper bound: 1.4286878
time: 4.52 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4203494, upper bound: 1.4295382
time: 4.32 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -7.1302719, -4.3909101, -7.1302719, -4.3909101, -2.6040292, 2.6040292
1: -7.2549419, -5.0657506, -7.2549419, -5.0657506, -2.1891913, 2.1891913
2: -6.0877600, -4.0038648, -6.0877600, -4.0038648, -1.9056253, 1.9056251
3: -6.1228518, -3.5679712, -6.1228518, -3.5679712, -2.5548806, 2.5548806
4: -6.4875026, -4.0545297, -6.4875026, -4.0545297, -2.4329729, 2.4329729
5: -6.5252895, -4.3112545, -6.5252895, -4.3112545, -2.2140350, 2.2140350
6: -11.4998178, -8.7039356, -11.4998178, -8.7039356, -2.7958822, 2.7958822
7: 2.7559633, 4.8221107, 2.7559633, 4.8221107, -2.0661473, 2.0661473
8: -4.4081459, -2.0556450, -4.4081459, -2.0556450, -2.2569170, 2.2569177
9: -2.7911782, -1.0629236, -2.7911782, -1.0629236, -1.7282547, 1.7282547

Time for backsubstitution: 14.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 484
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 484

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4092036, upper bound: 1.4286875
time: 4.71 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4203494, upper bound: 1.4295385
time: 4.53 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.1559477, -4.3700466, -7.1004982, -4.4165354, -2.6059952, 2.5964062
1: -7.3301582, -4.9731579, -7.2239199, -5.0750461, -2.2551122, 2.2507620
2: -6.1234574, -3.9540708, -6.0544543, -4.0355406, -1.9191191, 1.8965174
3: -6.2140942, -3.4875479, -6.0800323, -3.5789900, -2.6351042, 2.5924845
4: -6.5191393, -4.0193810, -6.4652915, -4.0889955, -2.4301438, 2.4459105
5: -6.5737514, -4.2848196, -6.5098515, -4.3232841, -2.2504673, 2.2250319
6: -11.5340376, -8.6715927, -11.4582014, -8.7297783, -2.8042593, 2.7866087
7: 2.7091510, 4.8529577, 2.7883801, 4.7697563, -2.0606053, 2.0645776
8: -4.4579649, -2.0227385, -4.3772411, -2.0776005, -2.2653141, 2.2482896
9: -2.8306656, -1.0376561, -2.7541146, -1.0822948, -1.7483708, 1.7164586

Time for backsubstitution: 14.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 484
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 484

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4180250, upper bound: 1.4286827
time: 4.28 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4292366, upper bound: 1.4295330
time: 4.68 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.1559477, -4.3700466, -7.1302719, -4.3909101, -2.6296759, 2.6239934
1: -7.3301582, -4.9731579, -7.2549419, -5.0657506, -2.2644076, 2.2817841
2: -6.1234574, -3.9540708, -6.0877600, -4.0038648, -1.9417267, 1.9297975
3: -6.2140942, -3.4875479, -6.1228518, -3.5679712, -2.6461229, 2.6353040
4: -6.5191393, -4.0193810, -6.4875026, -4.0545297, -2.4646096, 2.4681215
5: -6.5737514, -4.2848196, -6.5252895, -4.3112545, -2.2624969, 2.2404699
6: -11.5340376, -8.6715927, -11.4998178, -8.7039356, -2.8301020, 2.8282251
7: 2.7091510, 4.8529577, 2.7559633, 4.8221107, -2.1129596, 2.0969944
8: -4.4579649, -2.0227385, -4.4081459, -2.0556450, -2.2877259, 2.2891762
9: -2.8306656, -1.0376561, -2.7911782, -1.0629236, -1.7677420, 1.7535222

Time for backsubstitution: 14.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 484
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 484

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4180250, upper bound: 1.4286829
time: 4.39 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4292366, upper bound: 1.4295335
time: 4.95 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -7.1172113, -4.4442253, -7.1212578, -4.4086094, -2.6275511, 2.5662687
1: -7.2980137, -5.0895939, -7.2963314, -4.9903088, -2.3077049, 2.2067375
2: -6.1071815, -4.0081663, -6.0883207, -3.9882071, -1.9099984, 1.8962443
3: -6.1579351, -3.5972159, -6.1658249, -3.5063295, -2.6516056, 2.5686090
4: -6.4931512, -4.1009483, -6.4937277, -4.0663462, -2.4268050, 2.3927794
5: -6.5044780, -4.3176546, -6.5516963, -4.3020115, -2.2006106, 2.2340417
6: -11.4710178, -8.7074261, -11.4832153, -8.6990757, -2.7719421, 2.7757893
7: 2.7505209, 4.8187766, 2.7483976, 4.7982149, -2.0476940, 2.0703790
8: -4.4083481, -2.0449057, -4.4262862, -2.0479188, -2.2430139, 2.2526999
9: -2.7855072, -1.0584073, -2.7904439, -1.0607300, -1.7247772, 1.7320366

Time for backsubstitution: 14.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6178

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4091987, upper bound: 1.4375203
time: 3.92 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4091986, upper bound: 1.4286825
time: 4.56 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.1172113, -4.4442253, -7.1511459, -4.3828568, -2.6512575, 2.5892534
1: -7.2980137, -5.0895939, -7.3272238, -4.9807863, -2.3172274, 2.2376299
2: -6.1071815, -4.0081663, -6.1215811, -3.9564888, -1.9314351, 1.9219542
3: -6.1579351, -3.5972159, -6.2082758, -3.4953034, -2.6626318, 2.6110599
4: -6.4931512, -4.1009483, -6.5158892, -4.0316582, -2.4614930, 2.4149408
5: -6.5044780, -4.3176546, -6.5670643, -4.2896719, -2.2148061, 2.2494097
6: -11.4710178, -8.7074261, -11.5249577, -8.6731119, -2.7979059, 2.8175316
7: 2.7505209, 4.8187766, 2.7159374, 4.8512778, -2.1007569, 2.1028392
8: -4.4083481, -2.0449057, -4.4577332, -2.0260148, -2.2739010, 2.2900538
9: -2.7855072, -1.0584073, -2.8278742, -1.0413669, -1.7441403, 1.7694669

Time for backsubstitution: 14.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6178

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4091987, upper bound: 1.4375203
time: 4.26 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4091986, upper bound: 1.4286828
time: 4.47 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.2037201, -4.3817492, -7.1260462, -4.3953643, -2.6684413, 2.6578395
1: -7.3401566, -5.0489388, -7.2993340, -4.9826484, -2.3575082, 2.2503953
2: -6.1484365, -3.9915972, -6.0901980, -3.9856782, -1.9595308, 1.9143984
3: -6.2259502, -3.5511432, -6.1717052, -3.4978774, -2.7280729, 2.6205621
4: -6.5526733, -4.0343285, -6.4969878, -4.0540156, -2.4986577, 2.4626594
5: -6.5455351, -4.2827592, -6.5582981, -4.2969809, -2.2485542, 2.2755389
6: -11.5267038, -8.6757145, -11.4923897, -8.6975412, -2.8291626, 2.8166752
7: 2.7045541, 4.8498664, 2.7415898, 4.8004980, -2.0959439, 2.1082766
8: -4.4324212, -2.0225267, -4.4275432, -2.0446548, -2.2985115, 2.2820041
9: -2.8115811, -1.0342251, -2.7938085, -1.0569927, -1.7545885, 1.7595834

Time for backsubstitution: 14.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6178

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4203446, upper bound: 1.4384675
time: 4.61 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4203446, upper bound: 1.4295328
time: 4.47 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.2037201, -4.3817492, -7.1559706, -4.3696213, -2.6937773, 2.6800268
1: -7.3401566, -5.0489388, -7.3301830, -4.9731426, -2.3670139, 2.2812443
2: -6.1484365, -3.9915972, -6.1234570, -3.9539552, -1.9809539, 1.9412308
3: -6.2259502, -3.5511432, -6.2141147, -3.4868898, -2.7390604, 2.6629715
4: -6.5526733, -4.0343285, -6.5191517, -4.0193667, -2.5333066, 2.4848232
5: -6.5455351, -4.2827592, -6.5737629, -4.2846479, -2.2608871, 2.2910037
6: -11.5267038, -8.6757145, -11.5341549, -8.6715841, -2.8551197, 2.8584404
7: 2.7045541, 4.8498664, 2.7091217, 4.8535328, -2.1489787, 2.1407447
8: -4.4324212, -2.0225267, -4.4589643, -2.0227389, -2.3225207, 2.3195684
9: -2.8115811, -1.0342251, -2.8312201, -1.0376278, -1.7739533, 1.7969950

Time for backsubstitution: 14.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6178

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4203446, upper bound: 1.4384675
time: 4.64 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4203446, upper bound: 1.4295335
time: 4.31 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 23.22 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.22
Output dim: 7, lower bound: -1.4092064, upper bound: 1.4191963
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.22
Output dim: 7, lower bound: -1.4203523, upper bound: 1.4203502
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.22
Output dim: 7, lower bound: -1.4092065, upper bound: 1.4191956
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.22
Output dim: 7, lower bound: -1.4203523, upper bound: 1.4203495
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.22
Output dim: 7, lower bound: -1.4180279, upper bound: 1.4191914
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.22
Output dim: 7, lower bound: -1.4292395, upper bound: 1.4203453
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.22
Output dim: 7, lower bound: -1.4180279, upper bound: 1.4191911
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.22
Output dim: 7, lower bound: -1.4292395, upper bound: 1.4203448
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.22
Output dim: 7, lower bound: -1.4092015, upper bound: 1.4279971
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.22
Output dim: 7, lower bound: -1.4092015, upper bound: 1.4220907
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.22
Output dim: 7, lower bound: -1.4092015, upper bound: 1.4279966
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.22
Output dim: 7, lower bound: -1.4092015, upper bound: 1.4191904
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.22
Output dim: 7, lower bound: -1.4203475, upper bound: 1.4292375
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.22
Output dim: 7, lower bound: -1.4203475, upper bound: 1.4232646
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.22
Output dim: 7, lower bound: -1.4203475, upper bound: 1.4292370
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.22
Output dim: 7, lower bound: -1.4203475, upper bound: 1.4203468
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.22
Output dim: 7, lower bound: -1.4092036, upper bound: 1.4286878
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.22
Output dim: 7, lower bound: -1.4203494, upper bound: 1.4295382
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.22
Output dim: 7, lower bound: -1.4092036, upper bound: 1.4286875
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.22
Output dim: 7, lower bound: -1.4203494, upper bound: 1.4295385
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.22
Output dim: 7, lower bound: -1.4180250, upper bound: 1.4286827
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.22
Output dim: 7, lower bound: -1.4292366, upper bound: 1.4295330
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.22
Output dim: 7, lower bound: -1.4180250, upper bound: 1.4286829
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.22
Output dim: 7, lower bound: -1.4292366, upper bound: 1.4295335
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.22
Output dim: 7, lower bound: -1.4091987, upper bound: 1.4375203
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.22
Output dim: 7, lower bound: -1.4091986, upper bound: 1.4286825
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.22
Output dim: 7, lower bound: -1.4091987, upper bound: 1.4375203
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.22
Output dim: 7, lower bound: -1.4091986, upper bound: 1.4286828
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.22
Output dim: 7, lower bound: -1.4203446, upper bound: 1.4384675
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.22
Output dim: 7, lower bound: -1.4203446, upper bound: 1.4295328
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.22
Output dim: 7, lower bound: -1.4203446, upper bound: 1.4384675
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.22
Output dim: 7, lower bound: -1.4203446, upper bound: 1.4295335

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7.0785074, -4.4720697, -7.0958042, -4.4297113, -2.5411263, 2.4352081
1: -7.2102599, -5.1066265, -7.2208481, -5.0826674, -2.0776196, 2.1142216
2: -6.0455213, -4.0458918, -6.0525951, -4.0380917, -1.7434268, 1.8254912
3: -6.0556769, -3.6161091, -6.0740995, -3.5874519, -2.4682250, 2.4579904
4: -6.4499207, -4.1408119, -6.4619875, -4.1013026, -2.3486180, 2.3211756
5: -6.4842033, -4.3443871, -6.5033002, -4.3283248, -2.1492572, 2.1589131
6: -11.4190016, -8.7362728, -11.4489994, -8.7312908, -2.6877108, 2.6927018
7: 2.8168187, 4.7595048, 2.7951646, 4.7674751, -1.9506564, 1.9643402
8: -4.3715687, -2.0913925, -4.3759947, -2.0808630, -2.1700463, 2.1665545
9: -2.7423232, -1.0978405, -2.7509060, -1.0860480, -1.6562752, 1.6530654

Time for backsubstitution: 14.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5746

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4073607, upper bound: 1.4191956
time: 4.20 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4092029, upper bound: 1.4191957
time: 4.43 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.1649351, -4.4104891, -7.1004944, -4.4165430, -2.6047072, 2.5501831
1: -7.2524037, -5.0667162, -7.2239170, -5.0750537, -2.1773500, 2.1572008
2: -6.0867262, -4.0297132, -6.0544543, -4.0355430, -1.8847053, 1.8439457
3: -6.1237803, -3.5700240, -6.0800257, -3.5790014, -2.5447788, 2.5100017
4: -6.5093751, -4.0749931, -6.4652910, -4.0890074, -2.4203677, 2.3902979
5: -6.5240593, -4.3097110, -6.5098448, -4.3232884, -2.2007709, 2.2001338
6: -11.4744501, -8.7047443, -11.4581909, -8.7297802, -2.7446699, 2.7534466
7: 2.7714758, 4.7907190, 2.7883873, 4.7697544, -1.9982786, 2.0023317
8: -4.3956261, -2.0691853, -4.3772407, -2.0776024, -2.2268791, 2.1958911
9: -2.7674413, -1.0737039, -2.7541118, -1.0822972, -1.6851441, 1.6804079

Time for backsubstitution: 14.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5746

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4184658, upper bound: 1.4203495
time: 4.42 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4203489, upper bound: 1.4203498
time: 4.30 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 23.03 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 23.03
Output dim: 7, lower bound: -1.4073607, upper bound: 1.4191956
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 23.03
Output dim: 7, lower bound: -1.4092029, upper bound: 1.4191957
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 23.03
Output dim: 7, lower bound: -1.4184658, upper bound: 1.4203495
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 23.03
Output dim: 7, lower bound: -1.4203489, upper bound: 1.4203498
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 7, lower bound: -1.4092065, upper bound: 1.4191956
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 7, lower bound: -1.4203523, upper bound: 1.4203495
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 7, lower bound: -1.4180279, upper bound: 1.4191914
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 7, lower bound: -1.4292395, upper bound: 1.4203453
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 7, lower bound: -1.4180279, upper bound: 1.4191911
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 7, lower bound: -1.4292395, upper bound: 1.4203448
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 7, lower bound: -1.4092015, upper bound: 1.4279971
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 7, lower bound: -1.4092015, upper bound: 1.4220907
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 7, lower bound: -1.4092015, upper bound: 1.4279966
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 7, lower bound: -1.4092015, upper bound: 1.4191904
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 7, lower bound: -1.4203475, upper bound: 1.4292375
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 7, lower bound: -1.4203475, upper bound: 1.4232646
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 7, lower bound: -1.4203475, upper bound: 1.4292370
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 7, lower bound: -1.4203475, upper bound: 1.4203468
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 7, lower bound: -1.4092036, upper bound: 1.4286878
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 7, lower bound: -1.4203494, upper bound: 1.4295382
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 7, lower bound: -1.4092036, upper bound: 1.4286875
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 7, lower bound: -1.4203494, upper bound: 1.4295385
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 7, lower bound: -1.4180250, upper bound: 1.4286827
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 7, lower bound: -1.4292366, upper bound: 1.4295330
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 7, lower bound: -1.4180250, upper bound: 1.4286829
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 7, lower bound: -1.4292366, upper bound: 1.4295335
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 7, lower bound: -1.4091987, upper bound: 1.4375203
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 7, lower bound: -1.4091986, upper bound: 1.4286825
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 7, lower bound: -1.4091987, upper bound: 1.4375203
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 7, lower bound: -1.4091986, upper bound: 1.4286828
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 7, lower bound: -1.4203446, upper bound: 1.4384675
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 7, lower bound: -1.4203446, upper bound: 1.4295328
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 7, lower bound: -1.4203446, upper bound: 1.4384675
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.03
Output dim: 7, lower bound: -1.4203446, upper bound: 1.4295335
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.0716936588287354
rel_dist={7: [-1.4397835588004604, 1.43978321464756]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 484
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 466

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2399046, upper bound: 1.2342033
time: 5.15 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2399045, upper bound: 1.2399037
time: 5.77 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 11.10 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 11.10
Output dim: 7, lower bound: -1.2399046, upper bound: 1.2342033
IS_A2, status: Status.UNKNOWN, split count: 1, time: 11.10
Output dim: 7, lower bound: -1.2399045, upper bound: 1.2399037

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -7.1095839, -4.4137192, -7.1143208, -4.4039063, -2.4136324, 2.4077082
1: -7.2805634, -5.0668259, -7.2927589, -5.0636978, -2.2168655, 2.2259331
2: -6.0828629, -4.0292816, -6.0980883, -4.0269942, -1.7556643, 1.7692211
3: -6.1397429, -3.5712752, -6.1532335, -3.5676692, -2.4270267, 2.4403393
4: -6.4862461, -4.0831118, -6.4908619, -4.0701885, -2.4160576, 2.4077501
5: -6.5159121, -4.3084583, -6.5194330, -4.3049421, -2.0990944, 2.0965016
6: -11.4689255, -8.7266827, -11.4765635, -8.7145176, -2.6940379, 2.6882510
7: 2.7540634, 4.7761955, 2.7508283, 4.7978978, -1.9756148, 1.9573643
8: -4.3826246, -2.0528831, -4.3949118, -2.0500278, -2.0577159, 2.0685294
9: -2.7611513, -1.0621337, -2.7770782, -1.0586865, -1.7024648, 1.7149445

Time for backsubstitution: 14.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6178
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6178

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2398871, upper bound: 1.2292463
time: 4.43 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2398996, upper bound: 1.2341974
time: 5.29 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -7.1395717, -4.3879652, -7.1188102, -4.3942356, -2.4538755, 2.4368649
1: -7.3117166, -5.0573659, -7.3048553, -5.0606899, -2.2510266, 2.2474895
2: -6.1161394, -3.9974980, -6.1131401, -4.0248537, -1.7899680, 1.8149612
3: -6.1823025, -3.5602818, -6.1664557, -3.5639729, -2.4786258, 2.4727552
4: -6.5086150, -4.0486188, -6.4951205, -4.0573606, -2.4512544, 2.4465017
5: -6.5314770, -4.2962823, -6.5228024, -4.3014226, -2.1351528, 2.1232841
6: -11.5104237, -8.7007751, -11.4839706, -8.7025070, -2.7475724, 2.7211208
7: 2.7215409, 4.8289042, 2.7477779, 4.8194513, -2.0328965, 2.0112240
8: -4.4139233, -2.0308938, -4.4071021, -2.0474052, -2.0931807, 2.1095049
9: -2.7983041, -1.0427648, -2.7929301, -1.0555948, -1.7427093, 1.7501653

Time for backsubstitution: 14.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6178
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6178

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2398871, upper bound: 1.2349294
time: 4.51 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2398995, upper bound: 1.2398984
time: 5.73 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.61 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 24.61
Output dim: 7, lower bound: -1.2398871, upper bound: 1.2292463
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 24.61
Output dim: 7, lower bound: -1.2398996, upper bound: 1.2341974
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 24.61
Output dim: 7, lower bound: -1.2398871, upper bound: 1.2349294
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 24.61
Output dim: 7, lower bound: -1.2398995, upper bound: 1.2398984

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -7.1051064, -4.4149904, -7.1051850, -4.4067430, -2.3743005, 2.3634171
1: -7.2528334, -5.0707264, -7.2361417, -5.0719891, -2.1808443, 2.1654153
2: -6.0691261, -4.0322661, -6.0696826, -4.0333061, -1.7350378, 1.7376063
3: -6.1108451, -3.5748754, -6.0937033, -3.5753796, -2.3893356, 2.3766518
4: -6.4760313, -4.0859728, -6.4699202, -4.0760860, -2.3999453, 2.3839474
5: -6.5129929, -4.3159733, -6.5133419, -4.3197289, -2.0750113, 2.0804770
6: -11.4636059, -8.7281456, -11.4660454, -8.7176180, -2.6833897, 2.6711450
7: 2.7706959, 4.7726946, 2.7851133, 4.7914267, -1.9526513, 1.9209332
8: -4.3796315, -2.0648308, -4.3895092, -2.0747395, -2.0243893, 2.0481429
9: -2.7576790, -1.0718303, -2.7700167, -1.0788443, -1.6788347, 1.6981864

Time for backsubstitution: 14.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 484
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 484

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2317655, upper bound: 1.2271451
time: 4.42 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2398814, upper bound: 1.2292411
time: 5.06 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -7.1095819, -4.4137211, -7.1308460, -4.3855448, -2.4100218, 2.4279146
1: -7.2805557, -5.0668287, -7.3114567, -4.9794750, -2.2694318, 2.2446280
2: -6.0828466, -4.0292830, -6.1054287, -3.9833484, -1.7832196, 1.7687414
3: -6.1397152, -3.5712771, -6.1849632, -3.4942548, -2.5023365, 2.4617534
4: -6.4862413, -4.0831146, -6.5015035, -4.0409946, -2.4452467, 2.4183888
5: -6.5159101, -4.3084641, -6.5618048, -4.2934380, -2.1127722, 2.1414182
6: -11.4689207, -8.7266836, -11.5004730, -8.6853638, -2.7233772, 2.7123885
7: 2.7540748, 4.7761927, 2.7383509, 4.8223472, -2.0020592, 1.9686689
8: -4.3826246, -2.0529070, -4.4399176, -2.0418310, -2.0547218, 2.1014714
9: -2.7611494, -1.0621409, -2.8098636, -1.0535429, -1.7076066, 1.7477226

Time for backsubstitution: 14.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 484
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 484

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2318131, upper bound: 1.2320883
time: 4.63 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2398939, upper bound: 1.2341923
time: 5.12 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -7.1350570, -4.3892870, -7.1096110, -4.3970904, -2.4144468, 2.3924489
1: -7.2840843, -5.0613365, -7.2482662, -5.0689435, -2.2151408, 2.1869297
2: -6.1024213, -4.0005293, -6.0847406, -4.0312204, -1.7693324, 1.7832322
3: -6.1535754, -3.5638642, -6.1070781, -3.5716786, -2.4412208, 2.4096718
4: -6.4983473, -4.0514627, -6.4741917, -4.0632763, -2.4350710, 2.4227290
5: -6.5285168, -4.3038139, -6.5166945, -4.3161211, -2.1110823, 2.1072874
6: -11.5052242, -8.7022648, -11.4735184, -8.7056170, -2.7367525, 2.7041378
7: 2.7382097, 4.8254461, 2.7820566, 4.8129458, -2.0093451, 1.9748237
8: -4.4109678, -2.0428495, -4.4016519, -2.0721126, -2.0598950, 2.0891070
9: -2.7948709, -1.0524592, -2.7858484, -1.0757507, -1.7191201, 1.7333891

Time for backsubstitution: 14.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 484
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 484

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2317657, upper bound: 1.2328271
time: 4.35 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2398814, upper bound: 1.2349246
time: 4.42 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -7.1395688, -4.3879662, -7.1353774, -4.3758793, -2.4503002, 2.4569902
1: -7.3117089, -5.0573692, -7.3234797, -4.9762726, -2.3015928, 2.2661104
2: -6.1161218, -3.9974995, -6.1204886, -3.9811733, -1.8177309, 1.8144156
3: -6.1822753, -3.5602820, -6.1979227, -3.4905558, -2.5538998, 2.4941804
4: -6.5086098, -4.0486193, -6.5056682, -4.0280972, -2.4805126, 2.4570489
5: -6.5314741, -4.2962875, -6.5651579, -4.2898531, -2.1488976, 2.1682322
6: -11.5104218, -8.7007761, -11.5080662, -8.6733446, -2.7769108, 2.7455616
7: 2.7215505, 4.8289037, 2.7353406, 4.8440347, -2.0483203, 2.0224521
8: -4.4139204, -2.0309162, -4.4523325, -2.0392423, -2.0901899, 2.1307178
9: -2.7983017, -1.0427722, -2.8258266, -1.0504501, -1.7478516, 1.7830545

Time for backsubstitution: 14.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 484
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 484

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2318131, upper bound: 1.2377824
time: 4.71 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2398939, upper bound: 1.2398933
time: 5.15 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.15 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.15
Output dim: 7, lower bound: -1.2317655, upper bound: 1.2271451
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.15
Output dim: 7, lower bound: -1.2398814, upper bound: 1.2292411
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.15
Output dim: 7, lower bound: -1.2318131, upper bound: 1.2320883
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.15
Output dim: 7, lower bound: -1.2398939, upper bound: 1.2341923
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.15
Output dim: 7, lower bound: -1.2317657, upper bound: 1.2328271
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.15
Output dim: 7, lower bound: -1.2398814, upper bound: 1.2349246
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.15
Output dim: 7, lower bound: -1.2318131, upper bound: 1.2377824
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.15
Output dim: 7, lower bound: -1.2398939, upper bound: 1.2398933

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7.0830383, -4.4706306, -7.0949373, -4.4346266, -2.3466325, 2.2347050
1: -7.2391829, -5.1023750, -7.2295318, -5.0881100, -1.9684205, 2.1073678
2: -6.0601783, -4.0427194, -6.0656333, -4.0386877, -1.6115437, 1.7076778
3: -6.0865507, -3.6119905, -6.0811758, -3.5932651, -2.4097285, 2.3271258
4: -6.4606628, -4.1378598, -6.4627185, -4.1020870, -2.3585758, 2.3248587
5: -6.4872437, -4.3371167, -6.4995089, -4.3303289, -1.9708686, 2.0264616
6: -11.4243326, -8.7346935, -11.4466629, -8.7208557, -2.6174903, 2.5255265
7: 2.7992291, 4.7624378, 2.7994840, 4.7865667, -1.9117630, 1.8823249
8: -4.3739552, -2.0786481, -4.3868513, -2.0816317, -1.9928875, 2.0155718
9: -2.7458348, -1.0873929, -2.7631092, -1.0867130, -1.6591219, 1.6757163

Time for backsubstitution: 14.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2260906, upper bound: 1.2271451
time: 4.43 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2260906, upper bound: 1.2271472
time: 5.27 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.1695175, -4.4089069, -7.1051798, -4.4067593, -2.4169822, 2.3573446
1: -7.2812848, -5.0623569, -7.2361374, -5.0720024, -2.2092824, 2.1737804
2: -6.1014118, -4.0264044, -6.0696826, -4.0333118, -1.7688470, 1.7346745
3: -6.1546092, -3.5659127, -6.0936947, -3.5753999, -2.4318037, 2.3874133
4: -6.5201006, -4.0719280, -6.4699168, -4.0761075, -2.4439931, 2.3979888
5: -6.5272417, -4.3023767, -6.5133305, -4.3197365, -2.0717154, 2.0989072
6: -11.4799166, -8.7031059, -11.4660311, -8.7176218, -2.6957936, 2.6924458
7: 2.7537532, 4.7936764, 2.7851250, 4.7914233, -1.9633853, 1.9310613
8: -4.3980474, -2.0564208, -4.3895092, -2.0747437, -2.0540462, 2.0492604
9: -2.7710185, -1.0632646, -2.7700129, -1.0788503, -1.6921682, 1.7067482

Time for backsubstitution: 14.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2342050, upper bound: 1.2292411
time: 4.22 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2342050, upper bound: 1.2292410
time: 4.52 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.0874290, -4.4695115, -7.1204104, -4.4135933, -2.3841796, 2.2782118
1: -7.2669082, -5.0985842, -7.3050036, -4.9956794, -2.0501370, 2.1645305
2: -6.0738926, -4.0397868, -6.1013374, -3.9887018, -1.6671333, 1.7396181
3: -6.1154532, -3.6083899, -6.1723795, -3.5121467, -2.4939442, 2.4122286
4: -6.4708691, -4.1351166, -6.4943957, -4.0670724, -2.4037967, 2.3592792
5: -6.4899998, -4.3296237, -6.5478497, -4.3040237, -1.9998968, 2.0883906
6: -11.4296217, -8.7332888, -11.4810200, -8.6886454, -2.6583123, 2.5626197
7: 2.7827277, 4.7659330, 2.7527905, 4.8174777, -1.9621344, 1.9300706
8: -4.3769436, -2.0667567, -4.4373012, -2.0487337, -2.0226626, 2.0685186
9: -2.7491817, -1.0777223, -2.8026929, -1.0613831, -1.6849103, 1.7249706

Time for backsubstitution: 14.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2261376, upper bound: 1.2320880
time: 4.99 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2261376, upper bound: 1.2320881
time: 4.70 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.1739669, -4.4076018, -7.1308413, -4.3855581, -2.4509811, 2.4216628
1: -7.3089809, -5.0584173, -7.3114524, -4.9794888, -2.2833176, 2.2530351
2: -6.1151533, -4.0233793, -6.1054268, -3.9833531, -1.8032532, 1.7658432
3: -6.1834917, -3.5623150, -6.1849537, -3.4942751, -2.5260024, 2.4725020
4: -6.5303001, -4.0690274, -6.5014982, -4.0410166, -2.4892836, 2.4324708
5: -6.5301943, -4.2948475, -6.5617924, -4.2934465, -2.1095433, 2.1521668
6: -11.4851284, -8.7016344, -11.5004578, -8.6853666, -2.7359390, 2.7336884
7: 2.7370973, 4.7971916, 2.7383635, 4.8223438, -2.0126967, 1.9784024
8: -4.4010682, -2.0445027, -4.4399157, -2.0418377, -2.0844126, 2.1026065
9: -2.7744975, -1.0535977, -2.8098600, -1.0535491, -1.7209485, 1.7562623

Time for backsubstitution: 14.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2342083, upper bound: 1.2341924
time: 4.70 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2342083, upper bound: 1.2341922
time: 4.62 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -7.1127729, -4.4453974, -7.0993009, -4.4249644, -2.3855679, 2.2992327
1: -7.2703643, -5.0934491, -7.2416563, -5.0850353, -2.1853290, 2.1278031
2: -6.0934734, -4.0111094, -6.0806904, -4.0365801, -1.7380018, 1.7532399
3: -6.1291828, -3.6008005, -6.0946484, -3.5895357, -2.4661584, 2.3603506
4: -6.4828911, -4.1036711, -6.4669867, -4.0892372, -2.3936539, 2.3633156
5: -6.5016899, -4.3251276, -6.5028362, -4.3267202, -2.0021620, 2.0531549
6: -11.4657984, -8.7088623, -11.4541101, -8.7088528, -2.6708870, 2.6511698
7: 2.7670627, 4.8153224, 2.7964344, 4.8080978, -1.9654121, 1.9351985
8: -4.4054003, -2.0568190, -4.3990002, -2.0790071, -2.0292563, 2.0563149
9: -2.7822101, -1.0680823, -2.7789247, -1.0836189, -1.6985912, 1.7108424

Time for backsubstitution: 14.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2260749, upper bound: 1.2328269
time: 4.17 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2260750, upper bound: 1.2328270
time: 7.70 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.1992226, -4.3831062, -7.1096058, -4.3971043, -2.4461703, 2.3863301
1: -7.3125544, -5.0529480, -7.2482615, -5.0689569, -2.2435975, 2.1953135
2: -6.1347046, -3.9946680, -6.0847416, -4.0312243, -1.8034768, 1.7803798
3: -6.1972280, -3.5547252, -6.1070690, -3.5716991, -2.4836245, 2.4206004
4: -6.5424185, -4.0372119, -6.4741869, -4.0632977, -2.4791207, 2.4369750
5: -6.5425410, -4.2902260, -6.5166821, -4.3161306, -2.1077499, 2.1257739
6: -11.5216141, -8.6772089, -11.4735031, -8.7056179, -2.7490754, 2.7254300
7: 2.7212520, 4.8463936, 2.7820683, 4.8129430, -2.0200903, 1.9847634
8: -4.4294457, -2.0344634, -4.4016490, -2.0721159, -2.0892577, 2.0902090
9: -2.8081429, -1.0438921, -2.7858448, -1.0757558, -1.7323871, 1.7419527

Time for backsubstitution: 14.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2341893, upper bound: 1.2349238
time: 4.50 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2341893, upper bound: 1.2349242
time: 4.43 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -7.1172104, -4.4442263, -7.1249027, -4.4039168, -2.4218981, 2.3547459
1: -7.2980118, -5.0895944, -7.3170280, -4.9924645, -2.2754481, 2.1849854
2: -6.1071739, -4.0081673, -6.1163979, -3.9865308, -1.7847495, 1.7852119
3: -6.1579237, -3.5972176, -6.1854391, -3.5084171, -2.5593700, 2.4448633
4: -6.4931498, -4.1009483, -6.4985590, -4.0541334, -2.4390163, 2.3976107
5: -6.5044794, -4.3176560, -6.5511813, -4.3004341, -2.0313530, 2.1150365
6: -11.4710169, -8.7074261, -11.4885921, -8.6766253, -2.7118258, 2.6893330
7: 2.7505250, 4.8187771, 2.7497866, 4.8391771, -2.0048935, 1.9825833
8: -4.4083505, -2.0449162, -4.4497261, -2.0461454, -2.0595317, 2.0976198
9: -2.7855067, -1.0584114, -2.8186722, -1.0582904, -1.7258370, 1.7602608

Time for backsubstitution: 14.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2261218, upper bound: 1.2377820
time: 4.96 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2261218, upper bound: 1.2377825
time: 4.83 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.2037191, -4.3817501, -7.1353717, -4.3758936, -2.4801626, 2.4506803
1: -7.3401527, -5.0489402, -7.3234754, -4.9762859, -2.3154964, 2.2745352
2: -6.1484289, -3.9915967, -6.1204858, -3.9811780, -1.8378239, 1.8115828
3: -6.2259407, -3.5511441, -6.1979141, -3.4905748, -2.5778041, 2.5051105
4: -6.5526705, -4.0343275, -6.5056643, -4.0281181, -2.5245523, 2.4713368
5: -6.5455341, -4.2827616, -6.5651474, -4.2898612, -2.1456313, 2.1800075
6: -11.5267048, -8.6757164, -11.5080519, -8.6733465, -2.7893724, 2.7668519
7: 2.7045584, 4.8498669, 2.7353530, 4.8440304, -2.0590727, 2.0320516
8: -4.4324207, -2.0225363, -4.4523306, -2.0392466, -2.1195707, 2.1318853
9: -2.8115802, -1.0342265, -2.8258219, -1.0504553, -1.7611248, 1.7915953

Time for backsubstitution: 14.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2341926, upper bound: 1.2398929
time: 4.96 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2341926, upper bound: 1.2398935
time: 4.95 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 24.34 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.34
Output dim: 7, lower bound: -1.2260906, upper bound: 1.2271451
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.34
Output dim: 7, lower bound: -1.2260906, upper bound: 1.2271472
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.34
Output dim: 7, lower bound: -1.2342050, upper bound: 1.2292411
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.34
Output dim: 7, lower bound: -1.2342050, upper bound: 1.2292410
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.34
Output dim: 7, lower bound: -1.2261376, upper bound: 1.2320880
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.34
Output dim: 7, lower bound: -1.2261376, upper bound: 1.2320881
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.34
Output dim: 7, lower bound: -1.2342083, upper bound: 1.2341924
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.34
Output dim: 7, lower bound: -1.2342083, upper bound: 1.2341922
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.34
Output dim: 7, lower bound: -1.2260749, upper bound: 1.2328269
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.34
Output dim: 7, lower bound: -1.2260750, upper bound: 1.2328270
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.34
Output dim: 7, lower bound: -1.2341893, upper bound: 1.2349238
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.34
Output dim: 7, lower bound: -1.2341893, upper bound: 1.2349242
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.34
Output dim: 7, lower bound: -1.2261218, upper bound: 1.2377820
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.34
Output dim: 7, lower bound: -1.2261218, upper bound: 1.2377825
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.34
Output dim: 7, lower bound: -1.2341926, upper bound: 1.2398929
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.34
Output dim: 7, lower bound: -1.2341926, upper bound: 1.2398935

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -7.0830383, -4.4706306, -7.0903182, -4.4444299, -2.3362498, 2.2300403
1: -7.2391829, -5.1023750, -7.2173038, -5.0911636, -1.9566183, 2.0956073
2: -6.0601783, -4.0427194, -6.0504031, -4.0409298, -1.6073008, 1.6912475
3: -6.0865507, -3.6119905, -6.0674067, -3.5969045, -2.4012866, 2.3083534
4: -6.4606628, -4.1378598, -6.4580936, -4.1150370, -2.3456259, 2.3202338
5: -6.4872437, -4.3371167, -6.4960408, -4.3338861, -1.9627805, 2.0187268
6: -11.4243326, -8.7346935, -11.4387341, -8.7330160, -2.6045942, 2.5182285
7: 2.7992291, 4.7624378, 2.8027439, 4.7648878, -1.8887572, 1.8774087
8: -4.3739552, -2.0786481, -4.3745737, -2.0844908, -1.9889827, 2.0008130
9: -2.7458348, -1.0873929, -2.7473683, -1.0901634, -1.6556715, 1.6599754

Time for backsubstitution: 14.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2260845, upper bound: 1.2228837
time: 4.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2260843, upper bound: 1.2271385
time: 4.33 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -7.0830383, -4.4706306, -7.1199770, -4.4187851, -2.3622761, 2.2595177
1: -7.2391829, -5.1023750, -7.2483482, -5.0818624, -1.9671683, 2.1244690
2: -6.0601783, -4.0427194, -6.0837107, -4.0092487, -1.6392224, 1.7250830
3: -6.0865507, -3.6119905, -6.1103373, -3.5857980, -2.4190917, 2.3552532
4: -6.4606628, -4.1378598, -6.4802999, -4.0804915, -2.3801713, 2.3424401
5: -6.4872437, -4.3371167, -6.5112634, -4.3218541, -1.9792912, 2.0389414
6: -11.4243326, -8.7346935, -11.4803991, -8.7071657, -2.6321154, 2.5589600
7: 2.7992291, 4.7624378, 2.7703466, 4.8173051, -1.9413908, 1.9104798
8: -4.3739552, -2.0786481, -4.4055319, -2.0625610, -2.0101852, 2.0321169
9: -2.7458348, -1.0873929, -2.7841997, -1.0707964, -1.6750385, 1.6968068

Time for backsubstitution: 14.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2260845, upper bound: 1.2228862
time: 4.55 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2260843, upper bound: 1.2271382
time: 4.42 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.1695175, -4.4089069, -7.1004934, -4.4165487, -2.4091768, 2.3529902
1: -7.2812848, -5.0623569, -7.2239132, -5.0750589, -2.2062259, 2.1615562
2: -6.1014118, -4.0264044, -6.0544538, -4.0355463, -1.7655840, 1.7180827
3: -6.1546092, -3.5659127, -6.0800228, -3.5790098, -2.4269361, 2.3686528
4: -6.5201006, -4.0719280, -6.4652872, -4.0890179, -2.4310827, 2.3933592
5: -6.5272417, -4.3023767, -6.5098391, -4.3232937, -2.0616682, 2.0913668
6: -11.4799166, -8.7031059, -11.4581842, -8.7297812, -2.6828690, 2.6855249
7: 2.7537532, 4.7936764, 2.7883921, 4.7697535, -1.9404132, 1.9261384
8: -4.3980474, -2.0564208, -4.3772392, -2.0776043, -2.0500598, 2.0344839
9: -2.7710185, -1.0632646, -2.7541101, -1.0822996, -1.6887189, 1.6908455

Time for backsubstitution: 14.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2341984, upper bound: 1.2250056
time: 4.62 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2341983, upper bound: 1.2292343
time: 4.63 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.1695175, -4.4089069, -7.1302667, -4.3909240, -2.4244919, 2.3826041
1: -7.2812848, -5.0623569, -7.2549381, -5.0657640, -2.2155209, 2.1925812
2: -6.1014118, -4.0264044, -6.0877571, -4.0038705, -1.7784579, 1.7519847
3: -6.1546092, -3.5659127, -6.1228428, -3.5679901, -2.4427805, 2.4154975
4: -6.5201006, -4.0719280, -6.4875002, -4.0545506, -2.4655499, 2.4155722
5: -6.5272417, -4.3023767, -6.5252781, -4.3112626, -2.0811241, 2.1115587
6: -11.4799166, -8.7031059, -11.4998035, -8.7039375, -2.7104697, 2.7259259
7: 2.7537532, 4.7936764, 2.7559752, 4.8221083, -1.9947724, 1.9534092
8: -4.3980474, -2.0564208, -4.4081435, -2.0556488, -2.0702076, 2.0657020
9: -2.7710185, -1.0632646, -2.7911742, -1.0629299, -1.7080886, 1.7279096

Time for backsubstitution: 14.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2341984, upper bound: 1.2250060
time: 4.40 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2341983, upper bound: 1.2292345
time: 4.72 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -7.0874290, -4.4695115, -7.1156540, -4.4234133, -2.3737736, 2.2735388
1: -7.2669082, -5.0985842, -7.2928839, -4.9988594, -2.0408270, 2.1528375
2: -6.0738926, -4.0397868, -6.0861077, -3.9910243, -1.6629877, 1.7232194
3: -6.1154532, -3.6083899, -6.1590648, -3.5157871, -2.4864635, 2.3940229
4: -6.4708691, -4.1351166, -6.4898849, -4.0801220, -2.3907471, 2.3547683
5: -6.4899998, -4.3296237, -6.5443707, -4.3075657, -1.9916821, 2.0806124
6: -11.4296217, -8.7332888, -11.4729671, -8.7008228, -2.6454086, 2.5551791
7: 2.7827277, 4.7659330, 2.7560139, 4.7956209, -1.9387445, 1.9252083
8: -4.3769436, -2.0667567, -4.4248476, -2.0515509, -2.0187464, 2.0570838
9: -2.7491817, -1.0777223, -2.7866340, -1.0648301, -1.6802983, 1.7089117

Time for backsubstitution: 14.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2261307, upper bound: 1.2278092
time: 4.30 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2261308, upper bound: 1.2320810
time: 5.17 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.0874290, -4.4695115, -7.1455021, -4.3976502, -2.3999791, 2.2898111
1: -7.2669082, -5.0985842, -7.3238068, -4.9893208, -2.0512605, 2.1816800
2: -6.0738926, -4.0397868, -6.1193714, -3.9593134, -1.6761298, 1.7571596
3: -6.1154532, -3.6083899, -6.2015581, -3.5047150, -2.5025845, 2.4410005
4: -6.4708691, -4.1351166, -6.5120440, -4.0453935, -2.4254756, 2.3769274
5: -6.4899998, -4.3296237, -6.5596237, -4.2952232, -2.0085049, 2.1009190
6: -11.4296217, -8.7332888, -11.5146770, -8.6748562, -2.6730433, 2.5960088
7: 2.7827277, 4.7659330, 2.7235646, 4.8487196, -1.9812541, 1.9575083
8: -4.3769436, -2.0667567, -4.4563227, -2.0296621, -2.0399461, 2.0778680
9: -2.7491817, -1.0777223, -2.8240850, -1.0454684, -1.7001817, 1.7463627

Time for backsubstitution: 14.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2261307, upper bound: 1.2278089
time: 4.80 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2261308, upper bound: 1.2320811
time: 4.80 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.1739669, -4.4076018, -7.1260443, -4.3953686, -2.4432309, 2.4172101
1: -7.3089809, -5.0584173, -7.2993312, -4.9826541, -2.2788558, 2.2409139
2: -6.1151533, -4.0233793, -6.0901976, -3.9856801, -1.8000698, 1.7492671
3: -6.1834917, -3.5623150, -6.1717005, -3.4978857, -2.5215435, 2.4543092
4: -6.5303001, -4.0690274, -6.4969859, -4.0540257, -2.4762745, 2.4279585
5: -6.5301943, -4.2948475, -6.5582914, -4.2969856, -2.0993810, 2.1457441
6: -11.4851284, -8.7016344, -11.4923830, -8.6975422, -2.7230043, 2.7265072
7: 2.7370973, 4.7971916, 2.7415943, 4.8004966, -1.9896703, 1.9735317
8: -4.4010682, -2.0445027, -4.4275427, -2.0446568, -2.0804100, 2.0911143
9: -2.7744975, -1.0535977, -2.7938061, -1.0569943, -1.7175032, 1.7402084

Time for backsubstitution: 14.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2342017, upper bound: 1.2299320
time: 4.40 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2342016, upper bound: 1.2341856
time: 4.28 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.1739669, -4.4076018, -7.1559687, -4.3696289, -2.4588914, 2.4336345
1: -7.3089809, -5.0584173, -7.3301811, -4.9731483, -2.2875834, 2.2717028
2: -6.1151533, -4.0233793, -6.1234560, -3.9539576, -1.8130605, 1.7833122
3: -6.1834917, -3.5623150, -6.2141099, -3.4869003, -2.5356035, 2.5012305
4: -6.5303001, -4.0690274, -6.5191507, -4.0193763, -2.5109239, 2.4501233
5: -6.5301943, -4.2948475, -6.5737586, -4.2846513, -2.1191087, 2.1650066
6: -11.4851284, -8.7016344, -11.5341482, -8.6715832, -2.7507148, 2.7672529
7: 2.7370973, 4.7971916, 2.7091260, 4.8535309, -2.0349517, 2.0004454
8: -4.4010682, -2.0445027, -4.4589634, -2.0227432, -2.0945897, 2.1120036
9: -2.7744975, -1.0535977, -2.8312187, -1.0376308, -1.7368667, 1.7776210

Time for backsubstitution: 14.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2342017, upper bound: 1.2299320
time: 4.90 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2342016, upper bound: 1.2341855
time: 4.66 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -7.1127729, -4.4453974, -7.0903182, -4.4444299, -2.3648901, 2.2899764
1: -7.2703643, -5.0934491, -7.2173038, -5.0911636, -2.1792006, 2.1040354
2: -6.0934734, -4.0111094, -6.0504031, -4.0409298, -1.7303300, 1.7205319
3: -6.1291828, -3.6008005, -6.0674067, -3.5969045, -2.4554777, 2.3244591
4: -6.4828911, -4.1036711, -6.4580936, -4.1150370, -2.3678541, 2.3544226
5: -6.5016899, -4.3251276, -6.4960408, -4.3338861, -1.9807744, 2.0379620
6: -11.4657984, -8.7088623, -11.4387341, -8.7330160, -2.6452541, 2.6392546
7: 2.7670627, 4.8153224, 2.8027439, 4.7648878, -1.9192531, 1.9310443
8: -4.4054003, -2.0568190, -4.3745737, -2.0844908, -2.0214820, 2.0220478
9: -2.7822101, -1.0680823, -2.7473683, -1.0901634, -1.6920468, 1.6792860

Time for backsubstitution: 14.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2260688, upper bound: 1.2286104
time: 4.62 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2260686, upper bound: 1.2328201
time: 4.28 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -7.1127729, -4.4453974, -7.1199770, -4.4187851, -2.3889332, 2.3192332
1: -7.2703643, -5.0934491, -7.2483482, -5.0818624, -2.1885018, 2.1351526
2: -6.0934734, -4.0111094, -6.0837107, -4.0092487, -1.7569919, 1.7475507
3: -6.1291828, -3.6008005, -6.1103373, -3.5857980, -2.4731436, 2.3721766
4: -6.4828911, -4.1036711, -6.4802999, -4.0804915, -2.4023995, 2.3766289
5: -6.5016899, -4.3251276, -6.5112634, -4.3218541, -2.0080171, 2.0703132
6: -11.4657984, -8.7088623, -11.4803991, -8.7071657, -2.6677837, 2.6755686
7: 2.7670627, 4.8153224, 2.7703466, 4.8173051, -1.9619513, 1.9518983
8: -4.4054003, -2.0568190, -4.4055319, -2.0625610, -2.0514174, 2.0618210
9: -2.7822101, -1.0680823, -2.7841997, -1.0707964, -1.7114137, 1.7161174

Time for backsubstitution: 14.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2260688, upper bound: 1.2286111
time: 4.63 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2260687, upper bound: 1.2328206
time: 4.56 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.1992226, -4.3831062, -7.1004934, -4.4165487, -2.4254999, 2.3787441
1: -7.3125544, -5.0529480, -7.2239132, -5.0750589, -2.2374954, 2.1709652
2: -6.1347046, -3.9946680, -6.0544538, -4.0355463, -1.7833834, 1.7473507
3: -6.1972280, -3.5547252, -6.0800228, -3.5790098, -2.4736614, 2.3847342
4: -6.5424185, -4.0372119, -6.4652872, -4.0890179, -2.4534006, 2.4280753
5: -6.5425410, -4.2902260, -6.5098391, -4.3232937, -2.0823700, 2.1109619
6: -11.5216141, -8.6772089, -11.4581842, -8.7297812, -2.7233829, 2.7131634
7: 2.7212520, 4.8463936, 2.7883921, 4.7697535, -1.9742148, 1.9750896
8: -4.4294457, -2.0344634, -4.3772392, -2.0776043, -2.0809689, 2.0557141
9: -2.8081429, -1.0438921, -2.7541101, -1.0822996, -1.7258433, 1.7102180

Time for backsubstitution: 14.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2341827, upper bound: 1.2307312
time: 4.45 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2341826, upper bound: 1.2349175
time: 4.94 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.1992226, -4.3831062, -7.1302667, -4.3909240, -2.4502134, 2.4059734
1: -7.3125544, -5.0529480, -7.2549381, -5.0657640, -2.2467904, 2.2019901
2: -6.1347046, -3.9946680, -6.0877571, -4.0038705, -1.8100359, 1.7752211
3: -6.1972280, -3.5547252, -6.1228428, -3.5679901, -2.4903235, 2.4323964
4: -6.5424185, -4.0372119, -6.4875002, -4.0545506, -2.4878678, 2.4502883
5: -6.5425410, -4.2902260, -6.5252781, -4.3112626, -2.1134195, 2.1432767
6: -11.5216141, -8.6772089, -11.4998035, -8.7039375, -2.7465210, 2.7491016
7: 2.7212520, 4.8463936, 2.7559752, 4.8221083, -2.0157266, 2.0014832
8: -4.4294457, -2.0344634, -4.4081435, -2.0556488, -2.1066027, 2.0956655
9: -2.8081429, -1.0438921, -2.7911742, -1.0629299, -1.7452130, 1.7472820

Time for backsubstitution: 14.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2341827, upper bound: 1.2307342
time: 4.53 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2341826, upper bound: 1.2349190
time: 4.85 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -7.1172104, -4.4442263, -7.1156540, -4.4234133, -2.4012604, 2.3355391
1: -7.2980118, -5.0895944, -7.2928839, -4.9988594, -2.2625289, 2.1612959
2: -6.1071739, -4.0081673, -6.0861077, -3.9910243, -1.7647605, 1.7525694
3: -6.1579237, -3.5972176, -6.1590648, -3.5157871, -2.5446634, 2.4101028
4: -6.4931498, -4.1009483, -6.4898849, -4.0801220, -2.4130278, 2.3889365
5: -6.5044794, -4.3176560, -6.5443707, -4.3075657, -2.0096703, 2.0997095
6: -11.4710169, -8.7074261, -11.4729671, -8.7008228, -2.6861763, 2.6771035
7: 2.7505250, 4.8187771, 2.7560139, 4.7956209, -1.9585309, 1.9779360
8: -4.4083505, -2.0449162, -4.4248476, -2.0515509, -2.0517335, 2.0670540
9: -2.7855067, -1.0584114, -2.7866340, -1.0648301, -1.7169971, 1.7282226

Time for backsubstitution: 14.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2261149, upper bound: 1.2335363
time: 4.34 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2261150, upper bound: 1.2377752
time: 5.05 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.1172104, -4.4442263, -7.1455021, -4.3976502, -2.4262252, 2.3624167
1: -7.2980118, -5.0895944, -7.3238068, -4.9893208, -2.2793334, 2.1923273
2: -6.1071739, -4.0081673, -6.1193714, -3.9593134, -1.7915335, 1.7794828
3: -6.1579237, -3.5972176, -6.2015581, -3.5047150, -2.5664573, 2.4578986
4: -6.4931498, -4.1009483, -6.5120440, -4.0453935, -2.4477563, 2.4110956
5: -6.5044794, -4.3176560, -6.5596237, -4.2952232, -2.0372753, 2.1288197
6: -11.4710169, -8.7074261, -11.5146770, -8.6748562, -2.7088184, 2.7135210
7: 2.7505250, 4.8187771, 2.7235646, 4.8487196, -2.0077386, 1.9994335
8: -4.4083505, -2.0449162, -4.4563227, -2.0296621, -2.0816689, 2.1034813
9: -2.7855067, -1.0584114, -2.8240850, -1.0454684, -1.7400383, 1.7656736

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2261149, upper bound: 1.2335367
time: 4.62 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2261150, upper bound: 1.2377756
time: 4.41 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.2037191, -4.3817501, -7.1260443, -4.3953686, -2.4595940, 2.4325571
1: -7.3401527, -5.0489402, -7.2993312, -4.9826541, -2.3026581, 2.2503910
2: -6.1484289, -3.9915967, -6.0901976, -3.9856801, -1.8178425, 1.7785846
3: -6.2259407, -3.5511441, -6.1717005, -3.4978857, -2.5635529, 2.4703770
4: -6.5526705, -4.0343275, -6.4969859, -4.0540257, -2.4986448, 2.4626584
5: -6.5455341, -4.2827616, -6.5582914, -4.2969856, -2.1200271, 2.1635222
6: -11.5267048, -8.6757164, -11.4923830, -8.6975422, -2.7636614, 2.7541785
7: 2.7045584, 4.8498669, 2.7415943, 4.8004966, -2.0131700, 2.0220950
8: -4.4324207, -2.0225363, -4.4275427, -2.0446568, -2.1052876, 2.1010878
9: -2.8115802, -1.0342265, -2.7938061, -1.0569943, -1.7545859, 1.7595795

Time for backsubstitution: 14.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2341860, upper bound: 1.2356580
time: 4.51 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2341859, upper bound: 1.2398864
time: 4.75 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.2037191, -4.3817501, -7.1559687, -4.3696289, -2.4845710, 2.4583840
1: -7.3401527, -5.0489402, -7.3301811, -4.9731483, -2.3193722, 2.2812409
2: -6.1484289, -3.9915967, -6.1234560, -3.9539576, -1.8445928, 1.8063879
3: -6.2259407, -3.5511441, -6.2141099, -3.4869003, -2.5846128, 2.5181146
4: -6.5526705, -4.0343275, -6.5191507, -4.0193763, -2.5332942, 2.4848232
5: -6.5455341, -4.2827616, -6.5737586, -4.2846513, -2.1513939, 2.1925817
6: -11.5267048, -8.6757164, -11.5341482, -8.6715832, -2.7869101, 2.7904596
7: 2.7045584, 4.8498669, 2.7091260, 4.8535309, -2.0627718, 2.0489163
8: -4.4324207, -2.0225363, -4.4589634, -2.0227432, -2.1309133, 2.1377144
9: -2.8115802, -1.0342265, -2.8312187, -1.0376308, -1.7739494, 1.7969922

Time for backsubstitution: 14.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2341860, upper bound: 1.2356607
time: 4.23 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2341859, upper bound: 1.2398870
time: 4.63 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 23.20 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.20
Output dim: 7, lower bound: -1.2260845, upper bound: 1.2228837
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.20
Output dim: 7, lower bound: -1.2260843, upper bound: 1.2271385
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.20
Output dim: 7, lower bound: -1.2260845, upper bound: 1.2228862
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.20
Output dim: 7, lower bound: -1.2260843, upper bound: 1.2271382
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.20
Output dim: 7, lower bound: -1.2341984, upper bound: 1.2250056
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.20
Output dim: 7, lower bound: -1.2341983, upper bound: 1.2292343
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.20
Output dim: 7, lower bound: -1.2341984, upper bound: 1.2250060
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.20
Output dim: 7, lower bound: -1.2341983, upper bound: 1.2292345
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.20
Output dim: 7, lower bound: -1.2261307, upper bound: 1.2278092
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.20
Output dim: 7, lower bound: -1.2261308, upper bound: 1.2320810
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.20
Output dim: 7, lower bound: -1.2261307, upper bound: 1.2278089
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.20
Output dim: 7, lower bound: -1.2261308, upper bound: 1.2320811
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.20
Output dim: 7, lower bound: -1.2342017, upper bound: 1.2299320
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.20
Output dim: 7, lower bound: -1.2342016, upper bound: 1.2341856
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.20
Output dim: 7, lower bound: -1.2342017, upper bound: 1.2299320
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.20
Output dim: 7, lower bound: -1.2342016, upper bound: 1.2341855
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.20
Output dim: 7, lower bound: -1.2260688, upper bound: 1.2286104
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.20
Output dim: 7, lower bound: -1.2260686, upper bound: 1.2328201
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.20
Output dim: 7, lower bound: -1.2260688, upper bound: 1.2286111
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.20
Output dim: 7, lower bound: -1.2260687, upper bound: 1.2328206
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.20
Output dim: 7, lower bound: -1.2341827, upper bound: 1.2307312
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.20
Output dim: 7, lower bound: -1.2341826, upper bound: 1.2349175
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.20
Output dim: 7, lower bound: -1.2341827, upper bound: 1.2307342
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.20
Output dim: 7, lower bound: -1.2341826, upper bound: 1.2349190
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.20
Output dim: 7, lower bound: -1.2261149, upper bound: 1.2335363
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.20
Output dim: 7, lower bound: -1.2261150, upper bound: 1.2377752
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.20
Output dim: 7, lower bound: -1.2261149, upper bound: 1.2335367
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.20
Output dim: 7, lower bound: -1.2261150, upper bound: 1.2377756
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.20
Output dim: 7, lower bound: -1.2341860, upper bound: 1.2356580
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.20
Output dim: 7, lower bound: -1.2341859, upper bound: 1.2398864
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.20
Output dim: 7, lower bound: -1.2341860, upper bound: 1.2356607
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.20
Output dim: 7, lower bound: -1.2341859, upper bound: 1.2398870

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7.0708084, -4.4742651, -7.0865021, -4.4457192, -2.3162479, 2.2180495
1: -7.1847548, -5.1363068, -7.1921668, -5.0961914, -1.8983302, 2.0371795
2: -6.0348167, -4.0701313, -6.0461845, -4.0531988, -1.5689096, 1.6581709
3: -6.0520754, -3.6222405, -6.0597496, -3.5990071, -2.3591805, 2.2773249
4: -6.4083099, -4.1736069, -6.4345241, -4.1201367, -2.2881732, 2.2609172
5: -6.4634876, -4.3503733, -6.4854345, -4.3370328, -1.9345226, 1.9912050
6: -11.3903131, -8.7517900, -11.4244156, -8.7357473, -2.5665989, 2.4819374
7: 2.8364253, 4.7448115, 2.8088346, 4.7601361, -1.8458133, 1.8488216
8: -4.3444562, -2.1200719, -4.3681417, -2.1034164, -1.9402556, 1.9514763
9: -2.7001271, -1.1439168, -2.7407680, -1.1167769, -1.5833502, 1.5968511

Time for backsubstitution: 14.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5805

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2161227, upper bound: 1.2186250
time: 4.45 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2260815, upper bound: 1.2228970
time: 4.95 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.0830350, -4.4706311, -7.0903168, -4.4444313, -2.3431935, 2.2271945
1: -7.2391591, -5.1023788, -7.2172966, -5.0911655, -1.9116714, 2.0955994
2: -6.0601759, -4.0427284, -6.0504022, -4.0409327, -1.6072950, 1.6771362
3: -6.0865469, -3.6119924, -6.0674038, -3.5969052, -2.4002776, 2.3054352
4: -6.4606414, -4.1378632, -6.4580865, -4.1150379, -2.3342514, 2.3202233
5: -6.4872351, -4.3371201, -6.4960375, -4.3338876, -1.9549072, 2.0187232
6: -11.4243231, -8.7346964, -11.4387302, -8.7330170, -2.5944443, 2.5182204
7: 2.7992330, 4.7624354, 2.8027449, 4.7648869, -1.8877575, 1.8761733
8: -4.3739514, -2.0786581, -4.3745723, -2.0844936, -1.9889765, 1.9861784
9: -2.7458310, -1.0874054, -2.7473664, -1.0901667, -1.6556643, 1.6599610

Time for backsubstitution: 14.18 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.0076169967651367
rel_dist={7: [-1.2399111761101596, 1.2399105900004943]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 484
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 466

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1636913, upper bound: 1.1594552
time: 4.87 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1637590, upper bound: 1.1637583
time: 4.39 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 9.43 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 9.43
Output dim: 7, lower bound: -1.1636913, upper bound: 1.1594552
IS_A2, status: Status.UNKNOWN, split count: 1, time: 9.43
Output dim: 7, lower bound: -1.1637590, upper bound: 1.1637583

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -7.1095839, -4.4137192, -7.1133862, -4.4058709, -2.3436456, 2.3389220
1: -7.2805634, -5.0668259, -7.2903066, -5.0643177, -2.1938162, 2.1994538
2: -6.0828629, -4.0292816, -6.0950289, -4.0274425, -1.7119725, 1.7227933
3: -6.1397429, -3.5712752, -6.1505303, -3.5684052, -2.3675084, 2.3781581
4: -6.4862461, -4.0831118, -6.4899611, -4.0727897, -2.4134564, 2.4068494
5: -6.5159121, -4.3084583, -6.5187373, -4.3056450, -2.0365863, 2.0345383
6: -11.4689255, -8.7266827, -11.4750614, -8.7169609, -2.6382465, 2.6336136
7: 2.7540634, 4.7761955, 2.7514668, 4.7935300, -1.9374166, 1.9228113
8: -4.3826246, -2.0528831, -4.3924427, -2.0505838, -1.9979105, 2.0065403
9: -2.7611513, -1.0621337, -2.7738707, -1.0593510, -1.7018003, 1.7117370

Time for backsubstitution: 14.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6178
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6178

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1636875, upper bound: 1.1557523
time: 8.03 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1636875, upper bound: 1.1594498
time: 4.79 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -7.1395717, -4.3879652, -7.1188097, -4.3942356, -2.3859792, 2.3688498
1: -7.3117166, -5.0573659, -7.3048520, -5.0606909, -2.2256181, 2.2230248
2: -6.1161394, -3.9974980, -6.1131401, -4.0248551, -1.7462776, 1.7709501
3: -6.1823025, -3.5602818, -6.1664538, -3.5639729, -2.4201045, 2.4140289
4: -6.5086150, -4.0486188, -6.4951210, -4.0573630, -2.4512520, 2.4465022
5: -6.5314770, -4.2962823, -6.5228033, -4.3014240, -2.0741398, 2.0628045
6: -11.5104237, -8.7007751, -11.4839706, -8.7025070, -2.6943789, 2.6677046
7: 2.7215409, 4.8289042, 2.7477782, 4.8194480, -1.9988239, 1.9769466
8: -4.4139233, -2.0308938, -4.4071012, -2.0474062, -2.0341668, 2.0501778
9: -2.7983041, -1.0427648, -2.7929277, -1.0555949, -1.7427092, 1.7501630

Time for backsubstitution: 14.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6178
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6178

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1637551, upper bound: 1.1600580
time: 4.24 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1637551, upper bound: 1.1637545
time: 4.38 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 23.04 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 23.04
Output dim: 7, lower bound: -1.1636875, upper bound: 1.1557523
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 23.04
Output dim: 7, lower bound: -1.1636875, upper bound: 1.1594498
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 23.04
Output dim: 7, lower bound: -1.1637551, upper bound: 1.1600580
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 23.04
Output dim: 7, lower bound: -1.1637551, upper bound: 1.1637545

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -7.1040964, -4.4152937, -7.1042619, -4.4087067, -2.3008304, 2.2921219
1: -7.2467299, -5.0716066, -7.2336836, -5.0725932, -2.1526029, 2.1391938
2: -6.0661144, -4.0329361, -6.0666251, -4.0337448, -1.6883280, 1.6904194
3: -6.1044874, -3.5756960, -6.0909691, -3.5761161, -2.3233356, 2.3134494
4: -6.4737959, -4.0866175, -6.4690175, -4.0786834, -2.3951125, 2.3824000
5: -6.5123363, -4.3175888, -6.5126534, -4.3204508, -2.0118761, 2.0162115
6: -11.4624453, -8.7284765, -11.4644833, -8.7200603, -2.6259470, 2.6161723
7: 2.7743392, 4.7718763, 2.7857542, 4.7870650, -1.9110222, 1.8856571
8: -4.3789201, -2.0674548, -4.3870449, -2.0752959, -1.9637895, 1.9829788
9: -2.7568791, -1.0739689, -2.7668154, -1.0795103, -1.6773688, 1.6928465

Time for backsubstitution: 14.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 484
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 484

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1565716, upper bound: 1.1531499
time: 4.27 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1636824, upper bound: 1.1557479
time: 4.60 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -7.1095810, -4.4137201, -7.1298990, -4.3875351, -2.3399925, 2.3551002
1: -7.2805552, -5.0668278, -7.3090177, -4.9801116, -2.2172236, 2.1917014
2: -6.0828428, -4.0292835, -6.1023698, -3.9838116, -1.7385566, 1.7210588
3: -6.1397114, -3.5712776, -6.1823106, -3.4950290, -2.4415832, 2.3978379
4: -6.4862409, -4.0831146, -6.5006218, -4.0436115, -2.4426293, 2.4175072
5: -6.5159111, -4.3084641, -6.5611129, -4.2941661, -2.0492647, 2.0794301
6: -11.4689188, -8.7266836, -11.4988785, -8.6878071, -2.6675749, 2.6573648
7: 2.7540765, 4.7761931, 2.7389860, 4.8179188, -1.9637992, 1.9336157
8: -4.3826227, -2.0529089, -4.4373722, -2.0423832, -1.9924922, 2.0384696
9: -2.7611489, -1.0621420, -2.8066020, -1.0542094, -1.7069396, 1.7444600

Time for backsubstitution: 14.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 484
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 484

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1565716, upper bound: 1.1568446
time: 4.78 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1636824, upper bound: 1.1594454
time: 4.72 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -7.1340399, -4.3896017, -7.1096115, -4.3970923, -2.3430433, 2.3218923
1: -7.2780037, -5.0622315, -7.2482643, -5.0689435, -2.1844294, 2.1627884
2: -6.0994124, -4.0012083, -6.0847406, -4.0312204, -1.7226229, 1.7384527
3: -6.1472564, -3.5646811, -6.1070766, -3.5716801, -2.3762827, 2.3500476
4: -6.4960999, -4.0521035, -6.4741902, -4.0632777, -2.4328222, 2.4220867
5: -6.5278516, -4.3054333, -6.5166922, -4.3161230, -2.0494361, 2.0445275
6: -11.5041103, -8.7026014, -11.4735146, -8.7056170, -2.6818700, 2.6503649
7: 2.7418609, 4.8246393, 2.7820582, 4.8129425, -1.9715204, 1.9397883
8: -4.4102669, -2.0454741, -4.4016509, -2.0721126, -2.0000982, 2.0265951
9: -2.7940798, -1.0545971, -2.7858467, -1.0757507, -1.7183291, 1.7312496

Time for backsubstitution: 14.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 484
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 484

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1565715, upper bound: 1.1574124
time: 4.60 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1637505, upper bound: 1.1600542
time: 4.49 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -7.1395693, -4.3879662, -7.1353769, -4.3759069, -2.3823690, 2.3849461
1: -7.3117094, -5.0573688, -7.3234768, -4.9762740, -2.2498741, 2.2152219
2: -6.1161184, -3.9974992, -6.1204863, -3.9811802, -1.7731106, 1.7691438
3: -6.1822710, -3.5602822, -6.1979203, -3.4905944, -2.4941034, 2.4337244
4: -6.5086083, -4.0486202, -6.5056682, -4.0281005, -2.4805079, 2.4570479
5: -6.5314736, -4.2962890, -6.5651588, -4.2898626, -2.0868983, 2.1077504
6: -11.5104198, -8.7007771, -11.5080585, -8.6733456, -2.7237015, 2.6917524
7: 2.7215519, 4.8289018, 2.7353425, 4.8439984, -2.0137262, 1.9876578
8: -4.4139223, -2.0309200, -4.4522724, -2.0392427, -2.0287499, 2.0700607
9: -2.7983003, -1.0427731, -2.8257928, -1.0504529, -1.7478473, 1.7830197

Time for backsubstitution: 14.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 484
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 484

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1565716, upper bound: 1.1611107
time: 4.44 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1637506, upper bound: 1.1637504
time: 4.25 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 23.09 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 23.09
Output dim: 7, lower bound: -1.1565716, upper bound: 1.1531499
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 23.09
Output dim: 7, lower bound: -1.1636824, upper bound: 1.1557479
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 23.09
Output dim: 7, lower bound: -1.1565716, upper bound: 1.1568446
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 23.09
Output dim: 7, lower bound: -1.1636824, upper bound: 1.1594454
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 23.09
Output dim: 7, lower bound: -1.1565715, upper bound: 1.1574124
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 23.09
Output dim: 7, lower bound: -1.1637505, upper bound: 1.1600542
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 23.09
Output dim: 7, lower bound: -1.1565716, upper bound: 1.1611107
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 23.09
Output dim: 7, lower bound: -1.1637506, upper bound: 1.1637504

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.1685119, -4.4092188, -7.1042542, -4.4087238, -2.3423438, 2.2842169
1: -7.2751875, -5.0632458, -7.2336774, -5.0726085, -2.1733451, 2.1460891
2: -6.0983939, -4.0270824, -6.0666237, -4.0337505, -1.7205544, 1.6874797
3: -6.1482501, -3.5667315, -6.0909567, -3.5761418, -2.3659678, 2.3241487
4: -6.5178690, -4.0725822, -6.4690123, -4.0787086, -2.4391603, 2.3964300
5: -6.5265765, -4.3039970, -6.5126390, -4.3204594, -2.0085766, 2.0331495
6: -11.4787741, -8.7034397, -11.4644670, -8.7200632, -2.6377540, 2.6374693
7: 2.7574072, 4.7928572, 2.7857683, 4.7870622, -1.9203434, 1.8957784
8: -4.3973293, -2.0590429, -4.3870430, -2.0753016, -1.9926214, 1.9840856
9: -2.7702169, -1.0653988, -2.7668092, -1.0795166, -1.6907003, 1.7014104

Time for backsubstitution: 14.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5746

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1602467, upper bound: 1.1557423
time: 4.73 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1636766, upper bound: 1.1557425
time: 4.35 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.1739659, -4.4076014, -7.1298943, -4.3875523, -2.3795133, 2.3470085
1: -7.3089800, -5.0584183, -7.3090115, -4.9801283, -2.2311087, 2.1986516
2: -6.1151505, -4.0233793, -6.1023679, -3.9838171, -1.7577453, 1.7181604
3: -6.1834865, -3.5623145, -6.1823001, -3.4950526, -2.4652491, 2.4085064
4: -6.5302992, -4.0690289, -6.5006166, -4.0436378, -2.4866614, 2.4315877
5: -6.5301938, -4.2948484, -6.5610986, -4.2941766, -2.0460324, 2.0881119
6: -11.4851274, -8.7016335, -11.4988594, -8.6878119, -2.6795702, 2.6786604
7: 2.7370996, 4.7971926, 2.7389998, 4.8179140, -1.9730282, 1.9433472
8: -4.4010677, -2.0445070, -4.4373693, -2.0423865, -2.0213642, 2.0396042
9: -2.7744966, -1.0535984, -2.8065960, -1.0542160, -1.7202805, 1.7529976

Time for backsubstitution: 14.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5746

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1602459, upper bound: 1.1594396
time: 6.69 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1636767, upper bound: 1.1594398
time: 4.47 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.1982069, -4.3834286, -7.1096029, -4.3971090, -2.3733592, 2.3139510
1: -7.3064785, -5.0538516, -7.2482581, -5.0689592, -2.2060280, 2.1697984
2: -6.1316881, -3.9953568, -6.0847383, -4.0312262, -1.7552483, 1.7355950
3: -6.1909065, -3.5555410, -6.1070662, -3.5717034, -2.4188404, 2.3609090
4: -6.5401754, -4.0378609, -6.4741850, -4.0633035, -2.4768720, 2.4363241
5: -6.5418692, -4.2918320, -6.5166779, -4.3161330, -2.0460849, 2.0615237
6: -11.5204659, -8.6775494, -11.4734983, -8.7056198, -2.6936173, 2.6716542
7: 2.7249143, 4.8455853, 2.7820709, 4.8129396, -1.9808419, 1.9497550
8: -4.4287391, -2.0370874, -4.4016485, -2.0721169, -2.0286393, 2.0276868
9: -2.8073516, -1.0460253, -2.7858410, -1.0757573, -1.7315943, 1.7398157

Time for backsubstitution: 14.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5746

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1603317, upper bound: 1.1600498
time: 4.36 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1637449, upper bound: 1.1600489
time: 4.49 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -7.1172099, -4.4442263, -7.1224747, -4.4100780, -2.3449199, 2.2785664
1: -7.2980108, -5.0895948, -7.3155675, -4.9960017, -2.2198505, 2.1324770
2: -6.1071701, -4.0081677, -6.1154389, -3.9877036, -1.7379203, 1.7370776
3: -6.1579189, -3.5972173, -6.1828251, -3.5123665, -2.4973826, 2.3816769
4: -6.4931488, -4.1009493, -6.4968934, -4.0598297, -2.4145231, 2.3959441
5: -6.5044765, -4.3176551, -6.5481486, -4.3027220, -1.9665349, 2.0492959
6: -11.4710159, -8.7074261, -11.4843254, -8.6773663, -2.6554456, 2.6301718
7: 2.7505269, 4.8187757, 2.7529531, 4.8380585, -1.9681230, 1.9442241
8: -4.4083490, -2.0449190, -4.4490809, -2.0476489, -1.9951987, 2.0352135
9: -2.7855060, -1.0584118, -2.8170576, -1.0599644, -1.6861074, 1.7586458

Time for backsubstitution: 14.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5746

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1531682, upper bound: 1.1611049
time: 4.59 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1565660, upper bound: 1.1611056
time: 4.63 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.2037191, -4.3817487, -7.1353703, -4.3759222, -2.4105229, 2.3767977
1: -7.3401504, -5.0489416, -7.3234720, -4.9762902, -2.2637758, 2.2222445
2: -6.1484261, -3.9915962, -6.1204853, -3.9811864, -1.7923582, 1.7663102
3: -6.2259359, -3.5511444, -6.1979094, -3.4906170, -2.5180073, 2.4445744
4: -6.5526695, -4.0343275, -6.5056629, -4.0281267, -2.5245428, 2.4713354
5: -6.5455341, -4.2827621, -6.5651441, -4.2898731, -2.0836306, 2.1173368
6: -11.5267048, -8.6757164, -11.5080404, -8.6733475, -2.7355990, 2.7130404
7: 2.7045596, 4.8498659, 2.7353570, 4.8439941, -2.0230665, 1.9972558
8: -4.4324198, -2.0225391, -4.4522715, -2.0392461, -2.0573149, 2.0712276
9: -2.8115807, -1.0342282, -2.8257871, -1.0504587, -1.7611220, 1.7915589

Time for backsubstitution: 14.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5746

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1603318, upper bound: 1.1637444
time: 6.00 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1637448, upper bound: 1.1637446
time: 4.69 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 25.15 seconds
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.15
Output dim: 7, lower bound: -1.1602467, upper bound: 1.1557423
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.15
Output dim: 7, lower bound: -1.1636766, upper bound: 1.1557425
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.15
Output dim: 7, lower bound: -1.1602459, upper bound: 1.1594396
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.15
Output dim: 7, lower bound: -1.1636767, upper bound: 1.1594398
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.15
Output dim: 7, lower bound: -1.1603317, upper bound: 1.1600498
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.15
Output dim: 7, lower bound: -1.1637449, upper bound: 1.1600489
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.15
Output dim: 7, lower bound: -1.1531682, upper bound: 1.1611049
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.15
Output dim: 7, lower bound: -1.1565660, upper bound: 1.1611056
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.15
Output dim: 7, lower bound: -1.1603318, upper bound: 1.1637444
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.15
Output dim: 7, lower bound: -1.1637448, upper bound: 1.1637446

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.1638432, -4.4107428, -7.0916014, -4.4134798, -2.3279381, 2.2638617
1: -7.2453613, -5.0692949, -7.1776972, -5.1068134, -2.1001306, 2.0862288
2: -6.0932975, -4.0416479, -6.0407782, -4.0614424, -1.6870813, 1.6460221
3: -6.1390448, -3.5692129, -6.0549541, -3.5865145, -2.3328490, 2.2782078
4: -6.4898796, -4.0787106, -6.4162979, -4.1148605, -2.3750191, 2.3375874
5: -6.5140095, -4.3077536, -6.4884629, -4.3342557, -1.9786944, 2.0042291
6: -11.4615688, -8.7067575, -11.4302654, -8.7377148, -2.5998940, 2.5992594
7: 2.7646279, 4.7871504, 2.8232598, 4.7692876, -1.8907938, 1.8512700
8: -4.3895550, -2.0814910, -4.3574281, -2.1167936, -1.9417629, 1.9315679
9: -2.7620745, -1.0970013, -2.7204034, -1.1364119, -1.6256626, 1.6234021

Time for backsubstitution: 14.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5805

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1554985, upper bound: 1.1466997
time: 5.27 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1602436, upper bound: 1.1557396
time: 4.59 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.1685123, -4.4092197, -7.1042509, -4.4087238, -2.3365002, 2.2885342
1: -7.2751780, -5.0632472, -7.2336550, -5.0726118, -2.1569796, 2.1137538
2: -6.0983934, -4.0270867, -6.0666208, -4.0337591, -1.7031720, 1.6874735
3: -6.1482487, -3.5667322, -6.0909534, -3.5761421, -2.3618875, 2.3197680
4: -6.5178585, -4.0725832, -6.4689913, -4.0787115, -2.4391470, 2.3845468
5: -6.5265722, -4.3039961, -6.5126314, -4.3204627, -2.0085716, 2.0237756
6: -11.4787693, -8.7034397, -11.4644547, -8.7200661, -2.6375456, 2.6271138
7: 2.7574086, 4.7928557, 2.7857716, 4.7870584, -1.9186187, 1.8943050
8: -4.3973279, -2.0590477, -4.3870387, -2.0753093, -1.9771810, 1.9840798
9: -2.7702153, -1.0654044, -2.7668066, -1.0795287, -1.6774473, 1.7014022

Time for backsubstitution: 14.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5805

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1589163, upper bound: 1.1466995
time: 4.44 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1636744, upper bound: 1.1557397
time: 4.30 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.1692553, -4.4091282, -7.1170273, -4.3922215, -2.3649755, 2.3267975
1: -7.2791538, -5.0645537, -7.2533207, -5.0144424, -2.1579063, 2.1388493
2: -6.1100574, -4.0379539, -6.0763087, -4.0116634, -1.7243538, 1.6764460
3: -6.1742697, -3.5647893, -6.1464443, -3.5053840, -2.4323516, 2.3628373
4: -6.5023174, -4.0752668, -6.4478760, -4.0799880, -2.4223294, 2.3726091
5: -6.5176134, -4.2986097, -6.5368118, -4.3080354, -2.0161166, 2.0595455
6: -11.4679375, -8.7050133, -11.4640779, -8.7057133, -2.6415501, 2.6398401
7: 2.7443209, 4.7914619, 2.7764401, 4.8001041, -1.9434905, 1.8989236
8: -4.3932738, -2.0669432, -4.4077134, -2.0838404, -1.9704876, 1.9782276
9: -2.7662597, -1.0851960, -2.7598665, -1.1110688, -1.6551908, 1.6743196

Time for backsubstitution: 14.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5805

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1554984, upper bound: 1.1504208
time: 4.49 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1602436, upper bound: 1.1594369
time: 4.62 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.1739645, -4.4076014, -7.1298900, -4.3875542, -2.3736625, 2.3491101
1: -7.3089695, -5.0584197, -7.3089895, -4.9801311, -2.2147589, 2.1663170
2: -6.1151490, -4.0233836, -6.1023655, -3.9838266, -1.7403672, 1.7181535
3: -6.1834836, -3.5623174, -6.1822958, -3.4950552, -2.4564481, 2.4042072
4: -6.5302896, -4.0690289, -6.5005960, -4.0436401, -2.4866495, 2.4162436
5: -6.5301914, -4.2948484, -6.5610905, -4.2941780, -2.0460281, 2.0781200
6: -11.4851246, -8.7016363, -11.4988508, -8.6878128, -2.6793971, 2.6680570
7: 2.7371011, 4.7971907, 2.7390029, 4.8179121, -1.9672852, 1.9418745
8: -4.4010658, -2.0445099, -4.4373651, -2.0423942, -2.0059237, 2.0312548
9: -2.7744946, -1.0536047, -2.8065929, -1.0542278, -1.7095547, 1.7496161

Time for backsubstitution: 14.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5805

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1589165, upper bound: 1.1504225
time: 4.33 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1636744, upper bound: 1.1594367
time: 4.19 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.1935625, -4.3849125, -7.0969133, -4.4018412, -2.3589461, 2.2935767
1: -7.2769556, -5.0598927, -7.1924934, -5.1032763, -2.1329324, 2.1100338
2: -6.1266794, -4.0099244, -6.0588169, -4.0590200, -1.7217600, 1.6887531
3: -6.1816697, -3.5580630, -6.0712152, -3.5822153, -2.3855128, 2.3150647
4: -6.5120606, -4.0440578, -6.4215102, -4.0994673, -2.4125934, 2.3774524
5: -6.5293179, -4.2956448, -6.4925022, -4.3300228, -2.0161929, 2.0325978
6: -11.5033932, -8.6809063, -11.4393959, -8.7232580, -2.6561360, 2.6338630
7: 2.7320311, 4.8399720, 2.8194745, 4.7951393, -1.9506721, 1.9053953
8: -4.4208918, -2.0595484, -4.3720722, -2.1136084, -1.9776554, 1.9725552
9: -2.7991915, -1.0775777, -2.7392535, -1.1327015, -1.6664900, 1.6616758

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5805

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1555729, upper bound: 1.1509901
time: 4.73 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1603292, upper bound: 1.1600454
time: 6.70 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.1982040, -4.3834295, -7.1096015, -4.3971100, -2.3675094, 2.3181067
1: -7.3064690, -5.0538535, -7.2482352, -5.0689626, -2.1896572, 2.1374636
2: -6.1316872, -3.9953611, -6.0847373, -4.0312366, -1.7378335, 1.7256031
3: -6.1909032, -3.5555415, -6.1070595, -3.5717046, -2.4148030, 2.3565269
4: -6.5401659, -4.0378623, -6.4741635, -4.0633063, -2.4768596, 2.4261494
5: -6.5418644, -4.2918315, -6.5166702, -4.3161340, -2.0460808, 2.0521502
6: -11.5204639, -8.6775513, -11.4734898, -8.7056208, -2.6934242, 2.6617374
7: 2.7249172, 4.8455830, 2.7820749, 4.8129358, -1.9727514, 1.9482832
8: -4.4287376, -2.0370898, -4.4016438, -2.0721250, -2.0131979, 2.0257373
9: -2.8073499, -1.0460314, -2.7858374, -1.0757699, -1.7206452, 1.7398061

Time for backsubstitution: 14.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5805

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1589788, upper bound: 1.1509899
time: 5.42 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1637422, upper bound: 1.1600459
time: 4.31 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -7.1127024, -4.4456167, -7.1097918, -4.4147711, -2.3292806, 2.2578390
1: -7.2685208, -5.0956764, -7.2600436, -5.0303893, -2.1463346, 2.0722599
2: -6.1022167, -4.0227127, -6.0893397, -4.0156717, -1.7044888, 1.6893916
3: -6.1489091, -3.5997510, -6.1473227, -3.5228329, -2.4655466, 2.3364635
4: -6.4651327, -4.1072168, -6.4443007, -4.0961475, -2.3370647, 2.3370838
5: -6.4920073, -4.3213577, -6.5238037, -4.3165569, -1.9364114, 2.0203846
6: -11.4542103, -8.7107916, -11.4496479, -8.6952314, -2.6176400, 2.5919394
7: 2.7575872, 4.8131638, 2.7902360, 4.8202505, -1.9380441, 1.8999805
8: -4.4005518, -2.0673738, -4.4194651, -2.0891218, -1.9441433, 1.9741287
9: -2.7775631, -1.0899417, -2.7704241, -1.1168902, -1.6211610, 1.6804824

Time for backsubstitution: 14.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5805

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1485276, upper bound: 1.1521923
time: 5.68 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1531656, upper bound: 1.1611026
time: 4.80 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.1172075, -4.4442263, -7.1224709, -4.4100790, -2.3377872, 2.2808433
1: -7.2980003, -5.0895963, -7.3155437, -4.9960055, -2.2036505, 2.0992863
2: -6.1071701, -4.0081711, -6.1154370, -3.9877141, -1.7205329, 1.7271320
3: -6.1579170, -3.5972176, -6.1828203, -3.5123682, -2.4885941, 2.3773029
4: -6.4931397, -4.1009507, -6.4968719, -4.0598330, -2.3990579, 2.3780899
5: -6.5044742, -4.3176570, -6.5481415, -4.3027244, -1.9665291, 2.0413458
6: -11.4710121, -8.7074261, -11.4843159, -8.6773672, -2.6513991, 2.6199851
7: 2.7505288, 4.8187761, 2.7529569, 4.8380551, -1.9601486, 1.9427509
8: -4.4083486, -2.0449228, -4.4490747, -2.0476580, -1.9797578, 2.0268865
9: -2.7855048, -1.0584170, -2.8170538, -1.0599765, -1.6692991, 1.7586368

Time for backsubstitution: 14.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5805

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1519133, upper bound: 1.1521929
time: 4.53 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1565633, upper bound: 1.1611021
time: 4.81 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.1990314, -4.3832369, -7.1224976, -4.3805695, -2.3960037, 2.3565488
1: -7.3106289, -5.0550661, -7.2679920, -5.0107155, -2.1906967, 2.1626256
2: -6.1434216, -4.0061722, -6.0943489, -4.0091381, -1.7588477, 1.7192082
3: -6.2166877, -3.5536606, -6.1621952, -3.5010867, -2.4848356, 2.3991532
4: -6.5245595, -4.0406332, -6.4529605, -4.0644817, -2.4600778, 2.4123273
5: -6.5329771, -4.2865410, -6.5408583, -4.3037920, -2.0537663, 2.0887551
6: -11.5096369, -8.6791325, -11.4733601, -8.6912346, -2.6978006, 2.6746459
7: 2.7116742, 4.8442311, 2.7727182, 4.8261609, -1.9933276, 1.9529159
8: -4.4245520, -2.0449891, -4.4225764, -2.0806999, -2.0063267, 2.0098562
9: -2.8033328, -1.0657766, -2.7789714, -1.1073611, -1.6959717, 1.7101728

Time for backsubstitution: 14.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5805

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1555729, upper bound: 1.1547122
time: 4.65 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1603292, upper bound: 1.1637412
time: 4.91 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.2037177, -4.3817501, -7.1353674, -4.3759236, -2.4046650, 2.3787546
1: -7.3401413, -5.0489416, -7.3234482, -4.9762940, -2.2474227, 2.1899107
2: -6.1484261, -3.9916022, -6.1204829, -3.9811969, -1.7749486, 1.7562692
3: -6.2259336, -3.5511441, -6.1979041, -3.4906187, -2.5088749, 2.4402735
4: -6.5526595, -4.0343299, -6.5056415, -4.0281296, -2.5245299, 2.4577198
5: -6.5455313, -4.2827635, -6.5651369, -4.2898750, -2.0836253, 2.1073246
6: -11.5266991, -8.6757164, -11.5080309, -8.6733503, -2.7326627, 2.7028785
7: 2.7045619, 4.8498645, 2.7353611, 4.8439913, -2.0149055, 1.9957829
8: -4.4324198, -2.0225449, -4.4522676, -2.0392561, -2.0418744, 2.0628881
9: -2.8115788, -1.0342335, -2.8257833, -1.0504715, -1.7526407, 1.7855757

Time for backsubstitution: 14.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5805

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1589791, upper bound: 1.1547121
time: 8.14 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1637422, upper bound: 1.1637416
time: 4.54 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 27.04 seconds
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 27.04
Output dim: 7, lower bound: -1.1554985, upper bound: 1.1466997
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 27.04
Output dim: 7, lower bound: -1.1602436, upper bound: 1.1557396
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 27.04
Output dim: 7, lower bound: -1.1589163, upper bound: 1.1466995
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 27.04
Output dim: 7, lower bound: -1.1636744, upper bound: 1.1557397
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 27.04
Output dim: 7, lower bound: -1.1554984, upper bound: 1.1504208
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 27.04
Output dim: 7, lower bound: -1.1602436, upper bound: 1.1594369
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 27.04
Output dim: 7, lower bound: -1.1589165, upper bound: 1.1504225
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 27.04
Output dim: 7, lower bound: -1.1636744, upper bound: 1.1594367
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 27.04
Output dim: 7, lower bound: -1.1555729, upper bound: 1.1509901
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 27.04
Output dim: 7, lower bound: -1.1603292, upper bound: 1.1600454
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 27.04
Output dim: 7, lower bound: -1.1589788, upper bound: 1.1509899
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 27.04
Output dim: 7, lower bound: -1.1637422, upper bound: 1.1600459
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 27.04
Output dim: 7, lower bound: -1.1485276, upper bound: 1.1521923
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 27.04
Output dim: 7, lower bound: -1.1531656, upper bound: 1.1611026
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 27.04
Output dim: 7, lower bound: -1.1519133, upper bound: 1.1521929
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 27.04
Output dim: 7, lower bound: -1.1565633, upper bound: 1.1611021
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 27.04
Output dim: 7, lower bound: -1.1555729, upper bound: 1.1547122
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 27.04
Output dim: 7, lower bound: -1.1603292, upper bound: 1.1637412
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 27.04
Output dim: 7, lower bound: -1.1589791, upper bound: 1.1547121
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 27.04
Output dim: 7, lower bound: -1.1637422, upper bound: 1.1637416

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.1638317, -4.4107561, -7.0915985, -4.4134836, -2.3146157, 2.2316179
1: -7.2453566, -5.0692997, -7.1776948, -5.1068139, -2.0926504, 2.0862212
2: -6.0932937, -4.0416603, -6.0407777, -4.0614467, -1.6790991, 1.6165102
3: -6.1390324, -3.5692163, -6.0549498, -3.5865149, -2.3177161, 2.2761366
4: -6.4898787, -4.0787253, -6.4162984, -4.1148648, -2.3750138, 2.3375731
5: -6.5139904, -4.3077555, -6.4884586, -4.3342562, -1.9652791, 2.0042264
6: -11.4615622, -8.7067728, -11.4302635, -8.7377205, -2.5998802, 2.5839448
7: 2.7646351, 4.7871399, 2.8232622, 4.7692852, -1.8907819, 1.8351042
8: -4.3895426, -2.0814929, -4.3574247, -2.1167936, -1.9168925, 1.9315624
9: -2.7620649, -1.0970031, -2.7204010, -1.1364126, -1.6256523, 1.6233979

Time for backsubstitution: 14.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1560213, upper bound: 1.1557393
time: 4.19 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1560213, upper bound: 1.1557394
time: 4.83 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.1684999, -4.4092321, -7.1042504, -4.4087276, -2.3231790, 2.2563019
1: -7.2751741, -5.0632515, -7.2336531, -5.0726132, -2.1495075, 2.1137452
2: -6.0983891, -4.0270987, -6.0666208, -4.0337644, -1.6951902, 1.6579611
3: -6.1482344, -3.5667350, -6.0909481, -3.5761418, -2.3467450, 2.3177021
4: -6.5178571, -4.0725965, -6.4689889, -4.0787163, -2.4391408, 2.3744822
5: -6.5265551, -4.3039999, -6.5126286, -4.3204646, -1.9951558, 2.0237715
6: -11.4787607, -8.7034550, -11.4644556, -8.7200699, -2.6375332, 2.6117964
7: 2.7574165, 4.7928448, 2.7857749, 4.7870560, -1.9186068, 1.8781395
8: -4.3973150, -2.0590506, -4.3870344, -2.0753098, -1.9523106, 1.9840736
9: -2.7702060, -1.0654070, -2.7668042, -1.0795296, -1.6774344, 1.7013972

Time for backsubstitution: 14.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1594532, upper bound: 1.1557396
time: 4.10 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1594532, upper bound: 1.1557402
time: 9.63 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.1692433, -4.4091406, -7.1170235, -4.3922272, -2.3516545, 2.3087578
1: -7.2791481, -5.0645599, -7.2533174, -5.0144434, -2.1504259, 2.1388388
2: -6.1100550, -4.0379667, -6.0763083, -4.0116673, -1.7163696, 1.6469336
3: -6.1742558, -3.5647945, -6.1464381, -3.5053849, -2.4198785, 2.3607509
4: -6.5023160, -4.0752821, -6.4478736, -4.0799909, -2.4223251, 2.3725915
5: -6.5175962, -4.2986112, -6.5368085, -4.3080373, -2.0027018, 2.0551450
6: -11.4679327, -8.7050276, -11.4640770, -8.7057171, -2.6415372, 2.6245270
7: 2.7443283, 4.7914505, 2.7764418, 4.8001013, -1.9417880, 1.8827584
8: -4.3932619, -2.0669451, -4.4077086, -2.0838413, -1.9456177, 1.9701192
9: -2.7662497, -1.0851984, -2.7598634, -1.1110698, -1.6551799, 1.6746650

Time for backsubstitution: 14.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1560213, upper bound: 1.1594366
time: 4.32 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1560213, upper bound: 1.1594367
time: 4.68 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.1739540, -4.4076147, -7.1298871, -4.3875580, -2.3603399, 2.3315058
1: -7.3089638, -5.0584240, -7.3089867, -4.9801321, -2.2072871, 2.1663094
2: -6.1151462, -4.0233960, -6.1023655, -3.9838312, -1.7323852, 1.6886418
3: -6.1834717, -3.5623195, -6.1822910, -3.4950554, -2.4439602, 2.4021251
4: -6.5302887, -4.0690441, -6.5005941, -4.0436440, -2.4864850, 2.4061780
5: -6.5301733, -4.2948523, -6.5610857, -4.2941809, -2.0326126, 2.0737169
6: -11.4851131, -8.7016506, -11.4988489, -8.6878166, -2.6793842, 2.6527443
7: 2.7371085, 4.7971797, 2.7390070, 4.8179078, -1.9631009, 1.9257090
8: -4.4010534, -2.0445137, -4.4373598, -2.0423956, -1.9806027, 2.0231428
9: -2.7744853, -1.0536067, -2.8065901, -1.0542290, -1.7095428, 1.7503638

Time for backsubstitution: 14.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1594534, upper bound: 1.1594367
time: 4.28 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1594534, upper bound: 1.1594372
time: 7.41 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.1935501, -4.3849249, -7.0969095, -4.4018469, -2.3456235, 2.2613392
1: -7.2769518, -5.0598960, -7.1924915, -5.1032758, -2.1254523, 2.1100268
2: -6.1266770, -4.0099359, -6.0588169, -4.0590253, -1.7137952, 1.6588289
3: -6.1816568, -3.5580668, -6.0712094, -3.5822170, -2.3701077, 2.3129928
4: -6.5120578, -4.0440745, -6.4215078, -4.0994720, -2.4125857, 2.3774333
5: -6.5293031, -4.2956462, -6.4924984, -4.3300228, -2.0027781, 2.0325937
6: -11.5033855, -8.6809196, -11.4393959, -8.7232666, -2.6561227, 2.6185513
7: 2.7320383, 4.8399611, 2.8194776, 4.7951369, -1.9464951, 1.8892279
8: -4.4208775, -2.0595531, -4.3720689, -2.1136093, -1.9527855, 1.9644396
9: -2.7991815, -1.0775797, -2.7392507, -1.1327026, -1.6664789, 1.6616709

Time for backsubstitution: 14.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1560050, upper bound: 1.1599700
time: 4.56 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1560050, upper bound: 1.1599704
time: 4.59 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.1981945, -4.3834414, -7.1095982, -4.3971138, -2.3541875, 2.2858772
1: -7.3064642, -5.0538568, -7.2482347, -5.0689650, -2.1821849, 2.1374562
2: -6.1316843, -3.9953723, -6.0847349, -4.0312400, -1.7298698, 1.6956848
3: -6.1908903, -3.5555444, -6.1070552, -3.5717053, -2.3993917, 2.3544617
4: -6.5401630, -4.0378766, -6.4741616, -4.0633101, -2.4768529, 2.4160824
5: -6.5418472, -4.2918348, -6.5166664, -4.3161354, -2.0326662, 2.0521462
6: -11.5204563, -8.6775637, -11.4734869, -8.7056274, -2.6934118, 2.6464252
7: 2.7249236, 4.8455739, 2.7820771, 4.8129334, -1.9685695, 1.9321175
8: -4.4287252, -2.0370932, -4.4016395, -2.0721254, -1.9883280, 2.0176177
9: -2.8073404, -1.0460329, -2.7858343, -1.0757704, -1.7206335, 1.7398014

Time for backsubstitution: 14.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1594369, upper bound: 1.1599703
time: 4.40 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1594369, upper bound: 1.1599715
time: 8.14 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.1126866, -4.4456306, -7.1097879, -4.4147754, -2.3157048, 2.2392223
1: -7.2685146, -5.0956821, -7.2600422, -5.0303912, -2.1396403, 2.0722501
2: -6.1022134, -4.0227246, -6.0893383, -4.0156755, -1.6961470, 1.6597308
3: -6.1488957, -3.5997553, -6.1473179, -3.5228338, -2.4523177, 2.3343728
4: -6.4651308, -4.1072345, -6.4443007, -4.0961533, -2.3316259, 2.3370662
5: -6.4919825, -4.3213606, -6.5237961, -4.3165579, -1.9230609, 2.0174961
6: -11.4541969, -8.7108049, -11.4496441, -8.6952343, -2.6147614, 2.5770421
7: 2.7575943, 4.8131523, 2.7902384, 4.8202462, -1.9337857, 1.8838139
8: -4.4005413, -2.0673780, -4.4194603, -2.0891228, -1.9192734, 1.9660208
9: -2.7775524, -1.0899434, -2.7704215, -1.1168907, -1.6211467, 1.6804781

Time for backsubstitution: 14.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1488836, upper bound: 1.1610948
time: 4.10 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1488836, upper bound: 1.1610940
time: 4.23 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.1171961, -4.4442387, -7.1224680, -4.4100838, -2.3242106, 2.2635994
1: -7.2979951, -5.0896025, -7.3155432, -4.9960060, -2.1969635, 2.0992765
2: -6.1071668, -4.0081840, -6.1154361, -3.9877186, -1.7121940, 1.6974770
3: -6.1579037, -3.5972219, -6.1828146, -3.5123692, -2.4753518, 2.3752179
4: -6.4931383, -4.1009698, -6.4968719, -4.0598388, -2.3936152, 2.3680720
5: -6.5044494, -4.3176594, -6.5481353, -4.3027253, -1.9531784, 2.0371404
6: -11.4710007, -8.7074423, -11.4843121, -8.6773720, -2.6443977, 2.6050830
7: 2.7505355, 4.8187628, 2.7529602, 4.8380489, -1.9558845, 1.9265840
8: -4.4083347, -2.0449281, -4.4490700, -2.0476589, -1.9548869, 2.0187747
9: -2.7854929, -1.0584197, -2.8170502, -1.0599779, -1.6692853, 1.7586305

Time for backsubstitution: 14.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1522888, upper bound: 1.1610936
time: 4.34 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1522888, upper bound: 1.1611032
time: 5.76 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.1990209, -4.3832474, -7.1224942, -4.3805733, -2.3826809, 2.3386173
1: -7.3106256, -5.0550709, -7.2679901, -5.0107174, -2.1832166, 2.1626182
2: -6.1434197, -4.0061846, -6.0943480, -4.0091419, -1.7508812, 1.6893234
3: -6.2166743, -3.5536644, -6.1621871, -3.5010874, -2.4721990, 2.3970661
4: -6.5245566, -4.0406480, -6.4529581, -4.0644865, -2.4600701, 2.4123101
5: -6.5329580, -4.2865419, -6.5408549, -4.3037925, -2.0403495, 2.0843654
6: -11.5096302, -8.6791458, -11.4733601, -8.6912403, -2.6966820, 2.6593337
7: 2.7116823, 4.8442202, 2.7727203, 4.8261566, -1.9891496, 1.9367495
8: -4.4245415, -2.0449934, -4.4225726, -2.0806999, -1.9814572, 2.0017498
9: -2.8033226, -1.0657785, -2.7789683, -1.1073635, -1.6959591, 1.7109127

Time for backsubstitution: 14.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1560050, upper bound: 1.1636734
time: 4.40 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1560050, upper bound: 1.1636741
time: 4.90 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.2037067, -4.3817625, -7.1353641, -4.3759289, -2.3913436, 2.3611736
1: -7.3401365, -5.0489454, -7.3234463, -4.9762959, -2.2399497, 2.1899023
2: -6.1484227, -3.9916148, -6.1204815, -3.9812000, -1.7669814, 1.7263911
3: -6.2259207, -3.5511494, -6.1978993, -3.4906187, -2.4962287, 2.4381933
4: -6.5526571, -4.0343437, -6.5056400, -4.0281334, -2.5245237, 2.4476552
5: -6.5455136, -4.2827640, -6.5651326, -4.2898760, -2.0702095, 2.1029325
6: -11.5266914, -8.6757317, -11.5080280, -8.6733551, -2.7258077, 2.6875653
7: 2.7045679, 4.8498540, 2.7353625, 4.8439884, -2.0107222, 1.9796174
8: -4.4324055, -2.0225472, -4.4522629, -2.0392580, -2.0170040, 2.0547771
9: -2.8115697, -1.0342362, -2.8257804, -1.0504725, -1.7526283, 1.7863009

Time for backsubstitution: 14.18 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=1.974029779434204
rel_dist={7: [-1.163764129436149, 1.1637638135586528]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2418.61 seconds
