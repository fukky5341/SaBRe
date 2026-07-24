## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.13914137999999998


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-12.1458006, -11.1196728, -12.1458006, -11.1196728, -0.4895549, 0.4895554)
1: (-10.2953033, -9.5193138, -10.2953033, -9.5193138, -0.3319845, 0.3319844)
2: (-2.5454104, -1.7512214, -2.5454104, -1.7512214, -0.4164302, 0.4164302)
3: (5.9724727, 6.7451792, 5.9724727, 6.7451792, -0.3241732, 0.3241735)
4: (-11.1797190, -10.2502203, -11.1797190, -10.2502203, -0.3510623, 0.3510623)
5: (-6.6089749, -5.8434906, -6.6089749, -5.8434906, -0.3565919, 0.3565919)
6: (-12.3693848, -11.4272785, -12.3693848, -11.4272785, -0.4059968, 0.4059967)
7: (-6.4395571, -5.4970260, -6.4395571, -5.4970260, -0.3246870, 0.3246870)
8: (2.1057334, 3.0144646, 2.1057334, 3.0144646, -0.6173553, 0.6173553)
9: (-6.2699022, -5.3168850, -6.2699022, -5.3168850, -0.5370440, 0.5370440)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.34 + 32.57 = 54.92 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.1419808, upper bound: 0.1419806

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 455
type: B, layer: 1, pos: 455
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 455

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1419795, upper bound: 0.1402312
time: 3.09 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1419795, upper bound: 0.1419791
time: 2.90 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 6.22 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 6.22
Output dim: 3, lower bound: -0.1419795, upper bound: 0.1402312
NS_A2, status: Status.UNKNOWN, split count: 1, time: 6.22
Output dim: 3, lower bound: -0.1419795, upper bound: 0.1419791

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -12.1455307, -11.1228237, -12.1456642, -11.1213055, -0.4878201, 0.4864905
1: -10.2951880, -9.5249643, -10.2952452, -9.5222445, -0.3291395, 0.3266317
2: -2.5453322, -1.7536416, -2.5453687, -1.7524879, -0.4130533, 0.4123292
3: 5.9724970, 6.7407489, 5.9724846, 6.7428799, -0.3204937, 0.3185999
4: -11.1790676, -10.2524586, -11.1793919, -10.2513876, -0.3495097, 0.3486295
5: -6.6033220, -5.8435459, -6.6060305, -5.8435192, -0.3482401, 0.3497801
6: -12.3642693, -11.4273510, -12.3667326, -11.4273167, -0.4001825, 0.4024963
7: -6.4374523, -5.4970889, -6.4384656, -5.4970589, -0.3225658, 0.3235453
8: 2.1065252, 3.0140798, 2.1061451, 3.0142732, -0.6164961, 0.6166115
9: -6.2696619, -5.3183041, -6.2697802, -5.3176188, -0.5361500, 0.5355573

Time for backsubstitution: 20.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 455
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 455

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1402310, upper bound: 0.1402312
time: 3.28 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1402310, upper bound: 0.1402307
time: 3.47 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -12.1527719, -11.1195717, -12.1458015, -11.1196728, -0.4968183, 0.4900060
1: -10.3068037, -9.5189619, -10.2953024, -9.5193224, -0.3367500, 0.3306006
2: -2.5511589, -1.7496777, -2.5454106, -1.7512240, -0.4201944, 0.4194849
3: 5.9629707, 6.7457466, 5.9724703, 6.7451763, -0.3282285, 0.3237269
4: -11.1846523, -10.2469454, -11.1797180, -10.2502232, -0.3565414, 0.3540008
5: -6.6111660, -5.8348808, -6.6089711, -5.8434892, -0.3619273, 0.3603940
6: -12.3698730, -11.4154654, -12.3693838, -11.4272823, -0.4051352, 0.4129977
7: -6.4403133, -5.4925485, -6.4395556, -5.4970255, -0.3248622, 0.3283583
8: 2.1045647, 3.0154772, 2.1057343, 3.0144651, -0.6185875, 0.6184340
9: -6.2751656, -5.3168564, -6.2699022, -5.3168859, -0.5430818, 0.5364709

Time for backsubstitution: 20.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 455

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 525

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1419761, upper bound: 0.1406740
time: 2.90 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1419761, upper bound: 0.1419749
time: 2.90 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 26.02 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 26.02
Output dim: 3, lower bound: -0.1402310, upper bound: 0.1402312
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 26.02
Output dim: 3, lower bound: -0.1402310, upper bound: 0.1402307
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 26.02
Output dim: 3, lower bound: -0.1419761, upper bound: 0.1406740
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 26.02
Output dim: 3, lower bound: -0.1419761, upper bound: 0.1419749

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -12.1455307, -11.1228237, -12.1455307, -11.1228237, -0.4863958, 0.4863958
1: -10.2951880, -9.5249643, -10.2951880, -9.5249643, -0.3265820, 0.3265820
2: -2.5453322, -1.7536416, -2.5453322, -1.7536416, -0.4115045, 0.4115045
3: 5.9724970, 6.7407489, 5.9724970, 6.7407489, -0.3180773, 0.3180771
4: -11.1790676, -10.2524586, -11.1790676, -10.2524586, -0.3484380, 0.3484383
5: -6.6033220, -5.8435459, -6.6033220, -5.8435459, -0.3466015, 0.3466015
6: -12.3642693, -11.4273510, -12.3642693, -11.4273510, -0.3998635, 0.3998635
7: -6.4374523, -5.4970889, -6.4374523, -5.4970889, -0.3225377, 0.3225377
8: 2.1065252, 3.0140798, 2.1065252, 3.0140798, -0.6163077, 0.6163082
9: -6.2696619, -5.3183041, -6.2696619, -5.3183041, -0.5354748, 0.5354748

Time for backsubstitution: 21.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 525

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1389259, upper bound: 0.1402276
time: 3.01 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1402269, upper bound: 0.1402278
time: 3.19 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -12.1455307, -11.1228237, -12.1527719, -11.1195717, -0.4910603, 0.4938531
1: -10.2951880, -9.5249643, -10.3068037, -9.5189619, -0.3333876, 0.3314443
2: -2.5453322, -1.7536416, -2.5511589, -1.7496777, -0.4158165, 0.4170015
3: 5.9724970, 6.7407489, 5.9629707, 6.7457466, -0.3231745, 0.3232442
4: -11.1790676, -10.2524586, -11.1846523, -10.2469454, -0.3542275, 0.3543050
5: -6.6033220, -5.8435459, -6.6111660, -5.8348808, -0.3538082, 0.3554964
6: -12.3642693, -11.4273510, -12.3698730, -11.4154654, -0.4075487, 0.4061408
7: -6.4374523, -5.4970889, -6.4403133, -5.4925485, -0.3262677, 0.3257121
8: 2.1065252, 3.0140798, 2.1045647, 3.0154772, -0.6177974, 0.6183171
9: -6.2696619, -5.3183041, -6.2751656, -5.3168564, -0.5369501, 0.5416837

Time for backsubstitution: 21.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 525

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1389259, upper bound: 0.1402276
time: 2.91 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1402269, upper bound: 0.1402278
time: 3.13 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -12.1527719, -11.1195717, -12.1456394, -11.1197720, -0.4967246, 0.4898412
1: -10.3068037, -9.5189619, -10.2952919, -9.5213976, -0.3346543, 0.3305664
2: -2.5511589, -1.7496777, -2.5453744, -1.7529340, -0.4184747, 0.4194582
3: 5.9629707, 6.7457466, 5.9736242, 6.7451105, -0.3281970, 0.3225729
4: -11.1846523, -10.2469454, -11.1795073, -10.2502851, -0.3565097, 0.3537295
5: -6.6111660, -5.8348808, -6.6086912, -5.8435163, -0.3619051, 0.3601155
6: -12.3698730, -11.4154654, -12.3686504, -11.4273949, -0.4049716, 0.4122555
7: -6.4403133, -5.4925485, -6.4395499, -5.4982653, -0.3235707, 0.3283451
8: 2.1045647, 3.0154772, 2.1059973, 3.0130749, -0.6171803, 0.6182132
9: -6.2751656, -5.3168564, -6.2698269, -5.3179994, -0.5419579, 0.5364208

Time for backsubstitution: 21.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 455
type: A, layer: 1, pos: 525

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 455

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1402276, upper bound: 0.1406735
time: 2.92 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1402276, upper bound: 0.1406742
time: 3.72 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -12.1527729, -11.1195765, -12.1458397, -11.1177568, -0.4975867, 0.4901459
1: -10.3068027, -9.5189667, -10.3192701, -9.5190496, -0.3370434, 0.3350253
2: -2.5511599, -1.7496841, -2.5673425, -1.7511785, -0.4217427, 0.4225452
3: 5.9629745, 6.7457433, 5.9723873, 6.7614660, -0.3300853, 0.3245964
4: -11.1846504, -10.2469454, -11.1799278, -10.2444334, -0.3603420, 0.3555884
5: -6.6111650, -5.8348823, -6.6091976, -5.8416576, -0.3634632, 0.3607253
6: -12.3698711, -11.4154654, -12.3701363, -11.4183903, -0.4101200, 0.4144932
7: -6.4403133, -5.4925542, -6.4535236, -5.4964528, -0.3260636, 0.3286395
8: 2.1045661, 3.0154719, 2.0814955, 3.0145314, -0.6197515, 0.6298923
9: -6.2751651, -5.3168573, -6.2826071, -5.3146086, -0.5464430, 0.5430534

Time for backsubstitution: 21.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 455
type: A, layer: 1, pos: 525

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 455

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1402276, upper bound: 0.1419744
time: 3.07 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1402276, upper bound: 0.1419752
time: 3.45 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 28.57 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 28.57
Output dim: 3, lower bound: -0.1389259, upper bound: 0.1402276
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.57
Output dim: 3, lower bound: -0.1402269, upper bound: 0.1402278
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.57
Output dim: 3, lower bound: -0.1389259, upper bound: 0.1402276
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.57
Output dim: 3, lower bound: -0.1402269, upper bound: 0.1402278
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 28.57
Output dim: 3, lower bound: -0.1402276, upper bound: 0.1406735
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 28.57
Output dim: 3, lower bound: -0.1402276, upper bound: 0.1406742
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 28.57
Output dim: 3, lower bound: -0.1402276, upper bound: 0.1419744
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 28.57
Output dim: 3, lower bound: -0.1402276, upper bound: 0.1419752

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -12.1453676, -11.1229219, -12.1455307, -11.1228237, -0.4862318, 0.4863014
1: -10.2951784, -9.5270414, -10.2951880, -9.5249643, -0.3265476, 0.3244889
2: -2.5452919, -1.7553501, -2.5453322, -1.7536416, -0.4114778, 0.4097850
3: 5.9736490, 6.7406821, 5.9724970, 6.7407489, -0.3169234, 0.3180454
4: -11.1788597, -10.2525206, -11.1790676, -10.2524586, -0.3481667, 0.3484066
5: -6.6030397, -5.8435707, -6.6033220, -5.8435459, -0.3463347, 0.3465788
6: -12.3635387, -11.4274654, -12.3642693, -11.4273510, -0.3991203, 0.3996999
7: -6.4374456, -5.4983320, -6.4374523, -5.4970889, -0.3225241, 0.3212465
8: 2.1067882, 3.0126905, 2.1065252, 3.0140798, -0.6160874, 0.6149020
9: -6.2695875, -5.3194156, -6.2696619, -5.3183041, -0.5354233, 0.5343494

Time for backsubstitution: 21.52 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2579
type: B, layer: 3, pos: 2579
type: B, layer: 3, pos: 2147
type: A, layer: 3, pos: 2147
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 2565
type: B, layer: 3, pos: 2565
type: A, layer: 3, pos: 2229
type: B, layer: 3, pos: 2229
type: B, layer: 3, pos: 2376
type: A, layer: 3, pos: 2376
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 746
type: A, layer: 3, pos: 746
type: A, layer: 3, pos: 2228
type: B, layer: 3, pos: 2228
type: B, layer: 3, pos: 674
type: A, layer: 3, pos: 674
type: B, layer: 3, pos: 915
type: A, layer: 3, pos: 915
type: B, layer: 3, pos: 2458
type: A, layer: 3, pos: 2458
type: A, layer: 3, pos: 205
type: B, layer: 3, pos: 205
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 717
type: A, layer: 3, pos: 717
type: A, layer: 3, pos: 1382
type: B, layer: 3, pos: 1382
type: A, layer: 3, pos: 2620
type: B, layer: 3, pos: 2620
type: B, layer: 3, pos: 1998
type: A, layer: 3, pos: 1998
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 628
type: B, layer: 3, pos: 628
type: B, layer: 3, pos: 1729
type: A, layer: 3, pos: 1729

Time for candidate selection: 0.41 seconds

### Candidate
type: A, layer: 3, pos: 2536

## Relational analysis of NS_A1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1381216, upper bound: 0.1349124
time: 3.01 seconds

## Relational analysis of NS_A1_B1_A1_A2

### Relational analysis result of NS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1381278, upper bound: 0.1394292
time: 2.96 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -12.1455708, -11.1209059, -12.1455307, -11.1228247, -0.4865355, 0.4883332
1: -10.3191547, -9.5246925, -10.2951870, -9.5249701, -0.3312248, 0.3280346
2: -2.5672626, -1.7535970, -2.5453305, -1.7536483, -0.4170310, 0.4130523
3: 5.9724112, 6.7570376, 5.9725018, 6.7407494, -0.3189468, 0.3235623
4: -11.1792793, -10.2466688, -11.1790686, -10.2524586, -0.3500268, 0.3551106
5: -6.6035490, -5.8417149, -6.6033201, -5.8435454, -0.3469658, 0.3486354
6: -12.3650217, -11.4184599, -12.3642664, -11.4273529, -0.4018009, 0.4059052
7: -6.4514203, -5.4965162, -6.4374542, -5.4970942, -0.3262124, 0.3237396
8: 2.0822849, 3.0141463, 2.1065254, 3.0140738, -0.6280928, 0.6174726
9: -6.2823653, -5.3160243, -6.2696624, -5.3183050, -0.5420456, 0.5388350

Time for backsubstitution: 21.74 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 2579
type: B, layer: 3, pos: 2579
type: A, layer: 3, pos: 2147
type: B, layer: 3, pos: 2147
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 2565
type: B, layer: 3, pos: 2565
type: A, layer: 3, pos: 2229
type: B, layer: 3, pos: 2229
type: B, layer: 3, pos: 2376
type: A, layer: 3, pos: 2376
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 746
type: A, layer: 3, pos: 746
type: A, layer: 3, pos: 2228
type: B, layer: 3, pos: 2228
type: B, layer: 3, pos: 674
type: A, layer: 3, pos: 674
type: A, layer: 3, pos: 915
type: B, layer: 3, pos: 915
type: B, layer: 3, pos: 2458
type: A, layer: 3, pos: 2458
type: A, layer: 3, pos: 717
type: A, layer: 3, pos: 205
type: B, layer: 3, pos: 205
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 717
type: A, layer: 3, pos: 1382
type: B, layer: 3, pos: 1382
type: A, layer: 3, pos: 2620
type: B, layer: 3, pos: 2620
type: B, layer: 3, pos: 1998
type: A, layer: 3, pos: 1998
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 628
type: B, layer: 3, pos: 628
type: B, layer: 3, pos: 1729
type: A, layer: 3, pos: 1729

Time for candidate selection: 0.33 seconds

### Candidate
type: B, layer: 3, pos: 2536

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1349116, upper bound: 0.1394233
time: 3.04 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1394283, upper bound: 0.1394296
time: 3.10 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -12.1453676, -11.1229219, -12.1527719, -11.1195717, -0.4908957, 0.4937587
1: -10.2951784, -9.5270414, -10.3068037, -9.5189619, -0.3333533, 0.3293481
2: -2.5452919, -1.7553501, -2.5511589, -1.7496777, -0.4157901, 0.4152820
3: 5.9736490, 6.7406821, 5.9629707, 6.7457466, -0.3220208, 0.3232121
4: -11.1788597, -10.2525206, -11.1846523, -10.2469454, -0.3539560, 0.3542736
5: -6.6030397, -5.8435707, -6.6111660, -5.8348808, -0.3535297, 0.3554738
6: -12.3635387, -11.4274654, -12.3698730, -11.4154654, -0.4068065, 0.4059772
7: -6.4374456, -5.4983320, -6.4403133, -5.4925485, -0.3262544, 0.3244210
8: 2.1067882, 3.0126905, 2.1045647, 3.0154772, -0.6175766, 0.6169109
9: -6.2695875, -5.3194156, -6.2751656, -5.3168564, -0.5368986, 0.5405583

Time for backsubstitution: 21.92 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2579
type: B, layer: 3, pos: 2579
type: B, layer: 3, pos: 2147
type: A, layer: 3, pos: 2147
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 2565
type: A, layer: 3, pos: 2565
type: B, layer: 3, pos: 2229
type: A, layer: 3, pos: 2229
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 2376
type: B, layer: 3, pos: 2376
type: B, layer: 3, pos: 746
type: A, layer: 3, pos: 746
type: A, layer: 3, pos: 2228
type: B, layer: 3, pos: 2228
type: A, layer: 3, pos: 674
type: B, layer: 3, pos: 674
type: B, layer: 3, pos: 915
type: A, layer: 3, pos: 915
type: B, layer: 3, pos: 2458
type: A, layer: 3, pos: 2458
type: B, layer: 3, pos: 205
type: A, layer: 3, pos: 205
type: B, layer: 3, pos: 1382
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 557
type: B, layer: 3, pos: 717
type: A, layer: 3, pos: 717
type: A, layer: 3, pos: 1382
type: B, layer: 3, pos: 2620
type: A, layer: 3, pos: 2620
type: A, layer: 3, pos: 1998
type: B, layer: 3, pos: 1998
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 628
type: A, layer: 3, pos: 628
type: B, layer: 3, pos: 1729
type: A, layer: 3, pos: 1729

Time for candidate selection: 0.46 seconds

### Candidate
type: A, layer: 3, pos: 2536

## Relational analysis of NS_A1_B2_A1_A1

### Relational analysis result of NS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1398692, upper bound: 0.1349124
time: 3.03 seconds

## Relational analysis of NS_A1_B2_A1_A2

### Relational analysis result of NS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1398754, upper bound: 0.1394290
time: 3.06 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -12.1455708, -11.1209059, -12.1527729, -11.1195765, -0.4912000, 0.4946108
1: -10.3191547, -9.5246925, -10.3068027, -9.5189667, -0.3348898, 0.3316472
2: -2.5672626, -1.7535970, -2.5511599, -1.7496841, -0.4203641, 0.4185493
3: 5.9724112, 6.7570376, 5.9629745, 6.7457433, -0.3240438, 0.3251003
4: -11.1792793, -10.2466688, -11.1846504, -10.2469454, -0.3558161, 0.3581156
5: -6.6035490, -5.8417149, -6.6111650, -5.8348823, -0.3541389, 0.3575301
6: -12.3650217, -11.4184599, -12.3698711, -11.4154654, -0.4087899, 0.4092416
7: -6.4514203, -5.4965162, -6.4403133, -5.4925542, -0.3265489, 0.3269141
8: 2.0822849, 3.0141463, 2.1045661, 3.0154719, -0.6292567, 0.6194811
9: -6.2823653, -5.3160243, -6.2751651, -5.3168573, -0.5427217, 0.5450454

Time for backsubstitution: 22.09 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2579
type: B, layer: 3, pos: 2579
type: B, layer: 3, pos: 2147
type: A, layer: 3, pos: 2147
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 2565
type: B, layer: 3, pos: 2565
type: B, layer: 3, pos: 2229
type: A, layer: 3, pos: 2229
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 2376
type: A, layer: 3, pos: 2376
type: B, layer: 3, pos: 746
type: A, layer: 3, pos: 746
type: A, layer: 3, pos: 2228
type: B, layer: 3, pos: 2228
type: B, layer: 3, pos: 674
type: A, layer: 3, pos: 674
type: B, layer: 3, pos: 915
type: A, layer: 3, pos: 915
type: A, layer: 3, pos: 2458
type: B, layer: 3, pos: 2458
type: A, layer: 3, pos: 717
type: A, layer: 3, pos: 205
type: B, layer: 3, pos: 205
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 1382
type: B, layer: 3, pos: 717
type: B, layer: 3, pos: 557
type: A, layer: 3, pos: 1382
type: A, layer: 3, pos: 2620
type: B, layer: 3, pos: 2620
type: B, layer: 3, pos: 1998
type: A, layer: 3, pos: 1998
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 628
type: A, layer: 3, pos: 628
type: B, layer: 3, pos: 1729
type: A, layer: 3, pos: 1729

Time for candidate selection: 0.44 seconds

### Candidate
type: A, layer: 3, pos: 2536

## Relational analysis of NS_A1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1411701, upper bound: 0.1349122
time: 3.19 seconds

## Relational analysis of NS_A1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1411763, upper bound: 0.1394289
time: 3.14 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -12.1527719, -11.1195717, -12.1453676, -11.1229219, -0.4937587, 0.4908957
1: -10.3068037, -9.5189619, -10.2951784, -9.5270414, -0.3293480, 0.3333533
2: -2.5511589, -1.7496777, -2.5452919, -1.7553501, -0.4152820, 0.4157901
3: 5.9629707, 6.7457466, 5.9736490, 6.7406821, -0.3232121, 0.3220208
4: -11.1846523, -10.2469454, -11.1788597, -10.2525206, -0.3542736, 0.3539560
5: -6.6111660, -5.8348808, -6.6030397, -5.8435707, -0.3554738, 0.3535297
6: -12.3698730, -11.4154654, -12.3635387, -11.4274654, -0.4059772, 0.4068065
7: -6.4403133, -5.4925485, -6.4374456, -5.4983320, -0.3244209, 0.3262544
8: 2.1045647, 3.0154772, 2.1067882, 3.0126905, -0.6169109, 0.6175766
9: -6.2751656, -5.3168564, -6.2695875, -5.3194156, -0.5405583, 0.5368986

Time for backsubstitution: 21.79 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2579
type: A, layer: 3, pos: 2579
type: A, layer: 3, pos: 2147
type: B, layer: 3, pos: 2147
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 2565
type: B, layer: 3, pos: 2565
type: A, layer: 3, pos: 2229
type: B, layer: 3, pos: 2229
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 2376
type: A, layer: 3, pos: 2376
type: A, layer: 3, pos: 746
type: B, layer: 3, pos: 746
type: B, layer: 3, pos: 2228
type: A, layer: 3, pos: 2228
type: B, layer: 3, pos: 674
type: A, layer: 3, pos: 674
type: A, layer: 3, pos: 915
type: B, layer: 3, pos: 915
type: A, layer: 3, pos: 2458
type: B, layer: 3, pos: 2458
type: A, layer: 3, pos: 205
type: B, layer: 3, pos: 205
type: A, layer: 3, pos: 1382
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 557
type: A, layer: 3, pos: 717
type: B, layer: 3, pos: 717
type: B, layer: 3, pos: 1382
type: A, layer: 3, pos: 2620
type: B, layer: 3, pos: 2620
type: B, layer: 3, pos: 1998
type: A, layer: 3, pos: 1998
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 628
type: B, layer: 3, pos: 628
type: A, layer: 3, pos: 1729
type: B, layer: 3, pos: 1729

Time for candidate selection: 0.39 seconds

### Candidate
type: B, layer: 3, pos: 2536

## Relational analysis of NS_A2_B1_B1_B1

### Relational analysis result of NS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1349123, upper bound: 0.1398690
time: 3.17 seconds

## Relational analysis of NS_A2_B1_B1_B2

### Relational analysis result of NS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1394290, upper bound: 0.1398754
time: 2.91 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -12.1527719, -11.1195717, -12.1526108, -11.1196709, -0.4929259, 0.4928546
1: -10.3068037, -9.5189619, -10.3067951, -9.5210352, -0.3291118, 0.3311708
2: -2.5511589, -1.7496777, -2.5511215, -1.7513871, -0.4202108, 0.4219043
3: 5.9629707, 6.7457466, 5.9641218, 6.7456779, -0.3250515, 0.3239300
4: -11.1846523, -10.2469454, -11.1844425, -10.2470083, -0.3573654, 0.3571262
5: -6.6111660, -5.8348808, -6.6108856, -5.8349075, -0.3637490, 0.3635046
6: -12.3698730, -11.4154654, -12.3691435, -11.4155798, -0.4066343, 0.4060552
7: -6.4403133, -5.4925485, -6.4403090, -5.4937897, -0.3238535, 0.3251314
8: 2.1045647, 3.0154772, 2.1048281, 3.0140877, -0.6179886, 0.6191754
9: -6.2751656, -5.3168564, -6.2750893, -5.3179684, -0.5391765, 0.5402513

Time for backsubstitution: 21.86 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2579
type: A, layer: 3, pos: 2579
type: A, layer: 3, pos: 2147
type: B, layer: 3, pos: 2147
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 2565
type: A, layer: 3, pos: 2565
type: B, layer: 3, pos: 2229
type: A, layer: 3, pos: 2229
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 2376
type: B, layer: 3, pos: 2376
type: A, layer: 3, pos: 746
type: B, layer: 3, pos: 746
type: B, layer: 3, pos: 2228
type: A, layer: 3, pos: 2228
type: A, layer: 3, pos: 674
type: B, layer: 3, pos: 674
type: A, layer: 3, pos: 915
type: B, layer: 3, pos: 915
type: A, layer: 3, pos: 2458
type: B, layer: 3, pos: 2458
type: B, layer: 3, pos: 1382
type: A, layer: 3, pos: 1382
type: B, layer: 3, pos: 557
type: A, layer: 3, pos: 557
type: B, layer: 3, pos: 205
type: A, layer: 3, pos: 205
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 717
type: B, layer: 3, pos: 717
type: B, layer: 3, pos: 2620
type: A, layer: 3, pos: 2620
type: A, layer: 3, pos: 1998
type: B, layer: 3, pos: 1998
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 628
type: A, layer: 3, pos: 628
type: A, layer: 3, pos: 1729
type: B, layer: 3, pos: 1729

Time for candidate selection: 0.41 seconds

### Candidate
type: B, layer: 3, pos: 2536

## Relational analysis of NS_A2_B1_B2_B1

### Relational analysis result of NS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1349123, upper bound: 0.1398698
time: 3.63 seconds

## Relational analysis of NS_A2_B1_B2_B2

### Relational analysis result of NS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1394290, upper bound: 0.1398762
time: 3.38 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: -12.1527729, -11.1195765, -12.1455708, -11.1209059, -0.4946108, 0.4912000
1: -10.3068027, -9.5189667, -10.3191547, -9.5246925, -0.3316472, 0.3348898
2: -2.5511599, -1.7496841, -2.5672626, -1.7535970, -0.4185493, 0.4203641
3: 5.9629745, 6.7457433, 5.9724112, 6.7570376, -0.3251003, 0.3240438
4: -11.1846504, -10.2469454, -11.1792793, -10.2466688, -0.3581157, 0.3558161
5: -6.6111650, -5.8348823, -6.6035490, -5.8417149, -0.3575296, 0.3541390
6: -12.3698711, -11.4154654, -12.3650217, -11.4184599, -0.4092417, 0.4087899
7: -6.4403133, -5.4925542, -6.4514203, -5.4965162, -0.3269141, 0.3265491
8: 2.1045661, 3.0154719, 2.0822849, 3.0141463, -0.6194811, 0.6292562
9: -6.2751651, -5.3168573, -6.2823653, -5.3160243, -0.5450459, 0.5427222

Time for backsubstitution: 21.77 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2579
type: A, layer: 3, pos: 2579
type: A, layer: 3, pos: 2147
type: B, layer: 3, pos: 2147
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 2565
type: A, layer: 3, pos: 2565
type: A, layer: 3, pos: 2229
type: B, layer: 3, pos: 2229
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 2376
type: B, layer: 3, pos: 2376
type: A, layer: 3, pos: 746
type: B, layer: 3, pos: 746
type: B, layer: 3, pos: 2228
type: A, layer: 3, pos: 2228
type: A, layer: 3, pos: 674
type: B, layer: 3, pos: 674
type: A, layer: 3, pos: 915
type: B, layer: 3, pos: 915
type: A, layer: 3, pos: 2458
type: B, layer: 3, pos: 2458
type: B, layer: 3, pos: 717
type: B, layer: 3, pos: 205
type: A, layer: 3, pos: 205
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 1382
type: A, layer: 3, pos: 717
type: A, layer: 3, pos: 557
type: B, layer: 3, pos: 1382
type: B, layer: 3, pos: 2620
type: A, layer: 3, pos: 2620
type: A, layer: 3, pos: 1998
type: B, layer: 3, pos: 1998
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 628
type: B, layer: 3, pos: 628
type: A, layer: 3, pos: 1729
type: B, layer: 3, pos: 1729

Time for candidate selection: 0.35 seconds

### Candidate
type: B, layer: 3, pos: 2536

## Relational analysis of NS_A2_B2_B1_B1

### Relational analysis result of NS_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1349123, upper bound: 0.1411700
time: 3.07 seconds

## Relational analysis of NS_A2_B2_B1_B2

### Relational analysis result of NS_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1394290, upper bound: 0.1411764
time: 2.94 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: -12.1527729, -11.1195765, -12.1528101, -11.1176567, -0.4949572, 0.4931595
1: -10.3068027, -9.5189667, -10.3307714, -9.5186872, -0.3326581, 0.3354518
2: -2.5511599, -1.7496841, -2.5730920, -1.7496326, -0.4234784, 0.4249964
3: 5.9629745, 6.7457433, 5.9628849, 6.7620339, -0.3307763, 0.3259530
4: -11.1846504, -10.2469454, -11.1848621, -10.2411547, -0.3639660, 0.3589852
5: -6.6111650, -5.8348823, -6.6113939, -5.8330517, -0.3653564, 0.3641357
6: -12.3698711, -11.4154654, -12.3706264, -11.4065723, -0.4117777, 0.4087356
7: -6.4403133, -5.4925542, -6.4542794, -5.4919758, -0.3263464, 0.3288515
8: 2.1045661, 3.0154719, 2.0803268, 3.0155435, -0.6205597, 0.6313448
9: -6.2751651, -5.3168573, -6.2878695, -5.3145771, -0.5436611, 0.5455470

Time for backsubstitution: 21.91 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 2579
type: A, layer: 3, pos: 2579
type: B, layer: 3, pos: 2147
type: A, layer: 3, pos: 2147
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 2565
type: A, layer: 3, pos: 2565
type: B, layer: 3, pos: 2229
type: A, layer: 3, pos: 2229
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 2376
type: B, layer: 3, pos: 2376
type: A, layer: 3, pos: 746
type: B, layer: 3, pos: 746
type: B, layer: 3, pos: 2228
type: A, layer: 3, pos: 2228
type: A, layer: 3, pos: 674
type: B, layer: 3, pos: 674
type: B, layer: 3, pos: 915
type: A, layer: 3, pos: 915
type: A, layer: 3, pos: 2458
type: B, layer: 3, pos: 2458
type: B, layer: 3, pos: 1382
type: A, layer: 3, pos: 1382
type: B, layer: 3, pos: 717
type: A, layer: 3, pos: 557
type: B, layer: 3, pos: 557
type: B, layer: 3, pos: 205
type: A, layer: 3, pos: 205
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 717
type: B, layer: 3, pos: 2620
type: A, layer: 3, pos: 2620
type: A, layer: 3, pos: 1998
type: B, layer: 3, pos: 1998
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 628
type: A, layer: 3, pos: 628
type: B, layer: 3, pos: 1729
type: A, layer: 3, pos: 1729

Time for candidate selection: 0.39 seconds

### Candidate
type: A, layer: 3, pos: 2536

## Relational analysis of NS_A2_B2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1394232, upper bound: 0.1366599
time: 3.38 seconds

## Relational analysis of NS_A2_B2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1394295, upper bound: 0.1411766
time: 3.86 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 29.54 seconds
NS_A1_B1_A1_A1, status: Status.VERIFIED, split count: 4, time: 29.54
Output dim: 3, lower bound: -0.1381216, upper bound: 0.1349124
NS_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 29.54
Output dim: 3, lower bound: -0.1381278, upper bound: 0.1394292
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.54
Output dim: 3, lower bound: -0.1349116, upper bound: 0.1394233
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.54
Output dim: 3, lower bound: -0.1394283, upper bound: 0.1394296
NS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 29.54
Output dim: 3, lower bound: -0.1398692, upper bound: 0.1349124
NS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 29.54
Output dim: 3, lower bound: -0.1398754, upper bound: 0.1394290
NS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 29.54
Output dim: 3, lower bound: -0.1411701, upper bound: 0.1349122
NS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 29.54
Output dim: 3, lower bound: -0.1411763, upper bound: 0.1394289
NS_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 29.54
Output dim: 3, lower bound: -0.1349123, upper bound: 0.1398690
NS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 29.54
Output dim: 3, lower bound: -0.1394290, upper bound: 0.1398754
NS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 29.54
Output dim: 3, lower bound: -0.1349123, upper bound: 0.1398698
NS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 29.54
Output dim: 3, lower bound: -0.1394290, upper bound: 0.1398762
NS_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 29.54
Output dim: 3, lower bound: -0.1349123, upper bound: 0.1411700
NS_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 29.54
Output dim: 3, lower bound: -0.1394290, upper bound: 0.1411764
NS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 29.54
Output dim: 3, lower bound: -0.1394232, upper bound: 0.1366599
NS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 29.54
Output dim: 3, lower bound: -0.1394295, upper bound: 0.1411766

## BFS NS instance: NS_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -12.1386089, -11.1058388, -12.1412392, -11.1228237, -0.4716144, 0.4966192
1: -10.2892170, -9.5304298, -10.2916269, -9.5249672, -0.3202906, 0.3315072
2: -2.5436387, -1.7467685, -2.5441778, -1.7536414, -0.4047873, 0.4134374
3: 5.9590960, 6.7373734, 5.9724979, 6.7387772, -0.3263030, 0.3097649
4: -11.1775341, -10.2535534, -11.1782827, -10.2524586, -0.3477674, 0.3468993
5: -6.6164174, -5.8465052, -6.6033216, -5.8453560, -0.3608675, 0.3345270
6: -12.3560638, -11.4026461, -12.3597355, -11.4273510, -0.3737257, 0.4295770
7: -6.4403324, -5.4987464, -6.4374523, -5.4973636, -0.3238943, 0.3198462
8: 2.1126711, 3.0304120, 2.1101797, 3.0140798, -0.5989251, 0.6264243
9: -6.2746572, -5.3200154, -6.2696629, -5.3187609, -0.5396452, 0.5303903

Time for backsubstitution: 21.48 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2579
type: B, layer: 3, pos: 2579
type: A, layer: 3, pos: 2147
type: B, layer: 3, pos: 2147
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 2565
type: B, layer: 3, pos: 2565
type: A, layer: 3, pos: 2229
type: B, layer: 3, pos: 2229
type: B, layer: 3, pos: 2376
type: A, layer: 3, pos: 2376
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 746
type: A, layer: 3, pos: 746
type: B, layer: 3, pos: 2228
type: A, layer: 3, pos: 2228
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 674
type: A, layer: 3, pos: 674
type: A, layer: 3, pos: 915
type: B, layer: 3, pos: 915
type: A, layer: 3, pos: 2458
type: B, layer: 3, pos: 2458
type: A, layer: 3, pos: 205
type: B, layer: 3, pos: 205
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 717
type: A, layer: 3, pos: 717
type: A, layer: 3, pos: 1382
type: B, layer: 3, pos: 1382
type: A, layer: 3, pos: 2620
type: B, layer: 3, pos: 2620
type: B, layer: 3, pos: 1998
type: A, layer: 3, pos: 1998
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 628
type: B, layer: 3, pos: 628
type: A, layer: 3, pos: 1729
type: B, layer: 3, pos: 1729

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 2579

## Relational analysis of NS_A1_B1_A1_A2_A1

### Relational analysis result of NS_A1_B1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1354985, upper bound: 0.1368501
time: 3.51 seconds

## Relational analysis of NS_A1_B1_A1_A2_A2

### Relational analysis result of NS_A1_B1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1354985, upper bound: 0.1369618
time: 4.89 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -12.1332760, -11.1209059, -12.1247082, -11.1228247, -0.4747977, 0.4684520
1: -10.3186531, -9.5247126, -10.2943363, -9.5250044, -0.3271863, 0.3228415
2: -2.5616143, -1.7535970, -2.5372348, -1.7536483, -0.4121549, 0.4048052
3: 5.9724193, 6.7474432, 5.9725094, 6.7244949, -0.3069966, 0.3147860
4: -11.1791773, -10.2466755, -11.1788969, -10.2524691, -0.3498936, 0.3549263
5: -6.6035485, -5.8505497, -6.6033192, -5.8585100, -0.3313248, 0.3391979
6: -12.3470383, -11.4184589, -12.3338089, -11.4273510, -0.3829014, 0.3742366
7: -6.4514203, -5.4983149, -6.4374542, -5.5001402, -0.3242640, 0.3222581
8: 2.0967858, 3.0141463, 2.1285446, 3.0140738, -0.6151743, 0.5956140
9: -6.2823668, -5.3188810, -6.2696629, -5.3231449, -0.5371847, 0.5359678

Time for backsubstitution: 21.98 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2579
type: B, layer: 3, pos: 2579
type: B, layer: 3, pos: 2147
type: A, layer: 3, pos: 2147
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 2565
type: B, layer: 3, pos: 2565
type: A, layer: 3, pos: 2229
type: B, layer: 3, pos: 2229
type: B, layer: 3, pos: 2376
type: A, layer: 3, pos: 2376
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 746
type: A, layer: 3, pos: 746
type: A, layer: 3, pos: 2228
type: B, layer: 3, pos: 2228
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 674
type: A, layer: 3, pos: 674
type: B, layer: 3, pos: 915
type: A, layer: 3, pos: 915
type: B, layer: 3, pos: 2458
type: A, layer: 3, pos: 2458
type: A, layer: 3, pos: 717
type: A, layer: 3, pos: 205
type: B, layer: 3, pos: 205
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 717
type: A, layer: 3, pos: 1382
type: B, layer: 3, pos: 1382
type: A, layer: 3, pos: 2620
type: B, layer: 3, pos: 2620
type: B, layer: 3, pos: 1998
type: A, layer: 3, pos: 1998
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 628
type: B, layer: 3, pos: 628
type: B, layer: 3, pos: 1729
type: A, layer: 3, pos: 1729

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 2579

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1324897, upper bound: 0.1368444
time: 3.27 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1324897, upper bound: 0.1369602
time: 3.53 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -12.1412773, -11.1209087, -12.1387691, -11.1057396, -0.4968567, 0.4737172
1: -10.3155966, -9.5246954, -10.2892227, -9.5283470, -0.3382593, 0.3217866
2: -2.5661647, -1.7535970, -2.5436749, -1.7450664, -0.4207079, 0.4063623
3: 5.9724135, 6.7550673, 5.9579487, 6.7374420, -0.3106647, 0.3327559
4: -11.1784935, -10.2466688, -11.1777439, -10.2534933, -0.3485172, 0.3547106
5: -6.6035490, -5.8435249, -6.6166997, -5.8464813, -0.3349137, 0.3631680
6: -12.3604870, -11.4184589, -12.3567915, -11.4025240, -0.4316716, 0.3800448
7: -6.4514203, -5.4967875, -6.4403386, -5.4975109, -0.3248296, 0.3251094
8: 2.0859387, 3.0141463, 2.1124246, 3.0317960, -0.6397185, 0.6003094
9: -6.2823658, -5.3163953, -6.2747326, -5.3189154, -0.5380878, 0.5430584

Time for backsubstitution: 21.90 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2579
type: B, layer: 3, pos: 2579
type: B, layer: 3, pos: 2147
type: A, layer: 3, pos: 2147
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 2565
type: B, layer: 3, pos: 2565
type: B, layer: 3, pos: 2229
type: A, layer: 3, pos: 2229
type: B, layer: 3, pos: 2376
type: A, layer: 3, pos: 2376
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 746
type: B, layer: 3, pos: 746
type: A, layer: 3, pos: 2228
type: B, layer: 3, pos: 2228
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 674
type: A, layer: 3, pos: 674
type: B, layer: 3, pos: 915
type: A, layer: 3, pos: 915
type: A, layer: 3, pos: 2458
type: B, layer: 3, pos: 2458
type: A, layer: 3, pos: 717
type: A, layer: 3, pos: 205
type: B, layer: 3, pos: 205
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 717
type: A, layer: 3, pos: 1382
type: B, layer: 3, pos: 1382
type: A, layer: 3, pos: 2620
type: B, layer: 3, pos: 2620
type: B, layer: 3, pos: 1998
type: A, layer: 3, pos: 1998
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 628
type: A, layer: 3, pos: 628
type: B, layer: 3, pos: 1729
type: A, layer: 3, pos: 1729

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 2579

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1369610, upper bound: 0.1368502
time: 3.22 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1369609, upper bound: 0.1369622
time: 3.65 seconds

## BFS NS instance: NS_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -12.1245499, -11.1229210, -12.1404819, -11.1195717, -0.4710135, 0.4820201
1: -10.2943296, -9.5270720, -10.3063021, -9.5189800, -0.3281598, 0.3253100
2: -2.5371964, -1.7553501, -2.5455096, -1.7496777, -0.4075429, 0.4104125
3: 5.9736567, 6.7244282, 5.9629745, 6.7361498, -0.3131747, 0.3113327
4: -11.1786871, -10.2525291, -11.1845531, -10.2469511, -0.3537726, 0.3541398
5: -6.6030397, -5.8585353, -6.6111660, -5.8437166, -0.3442554, 0.3398328
6: -12.3330822, -11.4274654, -12.3518934, -11.4154654, -0.3751407, 0.3870780
7: -6.4374456, -5.5013790, -6.4403133, -5.4943466, -0.3247721, 0.3224573
8: 2.1288280, 3.0126905, 2.1190677, 3.0154772, -0.5957174, 0.6040049
9: -6.2695880, -5.3242545, -6.2751665, -5.3197131, -0.5340314, 0.5357027

Time for backsubstitution: 21.87 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2579
type: B, layer: 3, pos: 2579
type: B, layer: 3, pos: 2147
type: A, layer: 3, pos: 2147
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 2565
type: A, layer: 3, pos: 2565
type: B, layer: 3, pos: 2229
type: A, layer: 3, pos: 2229
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 2376
type: B, layer: 3, pos: 2376
type: B, layer: 3, pos: 746
type: A, layer: 3, pos: 746
type: B, layer: 3, pos: 2228
type: A, layer: 3, pos: 2228
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 674
type: B, layer: 3, pos: 674
type: A, layer: 3, pos: 915
type: B, layer: 3, pos: 915
type: B, layer: 3, pos: 2458
type: A, layer: 3, pos: 2458
type: B, layer: 3, pos: 205
type: A, layer: 3, pos: 205
type: B, layer: 3, pos: 1382
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 557
type: B, layer: 3, pos: 717
type: A, layer: 3, pos: 717
type: A, layer: 3, pos: 1382
type: B, layer: 3, pos: 2620
type: A, layer: 3, pos: 2620
type: A, layer: 3, pos: 1998
type: B, layer: 3, pos: 1998
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 628
type: A, layer: 3, pos: 628
type: B, layer: 3, pos: 1729
type: A, layer: 3, pos: 1729

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 2579

## Relational analysis of NS_A1_B2_A1_A1_A1

### Relational analysis result of NS_A1_B2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1372436, upper bound: 0.1322930
time: 3.77 seconds

## Relational analysis of NS_A1_B2_A1_A1_A2

### Relational analysis result of NS_A1_B2_A1_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1372436, upper bound: 0.1324906
time: 10.02 seconds

## BFS NS instance: NS_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -12.1386089, -11.1058388, -12.1484814, -11.1195736, -0.4762785, 0.5040758
1: -10.2892170, -9.5304298, -10.3032455, -9.5189638, -0.3270965, 0.3363827
2: -2.5436387, -1.7467685, -2.5500112, -1.7496777, -0.4090993, 0.4189339
3: 5.9590960, 6.7373734, 5.9629703, 6.7437716, -0.3313999, 0.3150089
4: -11.1775341, -10.2535534, -11.1838665, -10.2469463, -0.3535566, 0.3527665
5: -6.6164174, -5.8465052, -6.6111650, -5.8366923, -0.3681149, 0.3434215
6: -12.3560638, -11.4026461, -12.3653412, -11.4154654, -0.3809490, 0.4358542
7: -6.4403324, -5.4987464, -6.4403133, -5.4928222, -0.3276306, 0.3230207
8: 2.1126711, 3.0304120, 2.1082187, 3.0154772, -0.6004143, 0.6284337
9: -6.2746572, -5.3200154, -6.2751656, -5.3173094, -0.5411201, 0.5366006

Time for backsubstitution: 21.77 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 54.92 + 547.66 = 602.58 seconds
