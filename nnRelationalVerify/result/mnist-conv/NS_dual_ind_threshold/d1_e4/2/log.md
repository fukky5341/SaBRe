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
execution time: IAR + RelationalAnalysis = 22.94 + 33.06 = 56.00 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.1419808, upper bound: 0.1419806

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 455
type: A, layer: 1, pos: 525

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 455

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1419795, upper bound: 0.1402312
time: 3.16 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1419795, upper bound: 0.1419791
time: 2.99 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 6.38 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 6.38
Output dim: 3, lower bound: -0.1419795, upper bound: 0.1402312
NS_A2, status: Status.UNKNOWN, split count: 1, time: 6.38
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

Time for backsubstitution: 21.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 455
type: B, layer: 1, pos: 525

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 455

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1402310, upper bound: 0.1402312
time: 3.29 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1402310, upper bound: 0.1402307
time: 3.46 seconds

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

Time for backsubstitution: 20.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 455
type: B, layer: 1, pos: 525

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 455

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1402310, upper bound: 0.1419798
time: 3.15 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1402310, upper bound: 0.1419798
time: 3.24 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 27.42 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 27.42
Output dim: 3, lower bound: -0.1402310, upper bound: 0.1402312
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 27.42
Output dim: 3, lower bound: -0.1402310, upper bound: 0.1402307
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 27.42
Output dim: 3, lower bound: -0.1402310, upper bound: 0.1419798
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 27.42
Output dim: 3, lower bound: -0.1402310, upper bound: 0.1419798

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

Time for backsubstitution: 21.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 525

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 525

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1389259, upper bound: 0.1402276
time: 3.00 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1402269, upper bound: 0.1402278
time: 3.14 seconds

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

Time for backsubstitution: 21.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 525

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 525

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1389259, upper bound: 0.1402276
time: 3.00 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1402269, upper bound: 0.1402278
time: 3.21 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -12.1527719, -11.1195717, -12.1455307, -11.1228237, -0.4938531, 0.4910600
1: -10.3068037, -9.5189619, -10.2951880, -9.5249643, -0.3314445, 0.3333875
2: -2.5511589, -1.7496777, -2.5453322, -1.7536416, -0.4170015, 0.4158165
3: 5.9629707, 6.7457466, 5.9724970, 6.7407489, -0.3232442, 0.3231745
4: -11.1846523, -10.2469454, -11.1790676, -10.2524586, -0.3543050, 0.3542275
5: -6.6111660, -5.8348808, -6.6033220, -5.8435459, -0.3554962, 0.3538082
6: -12.3698730, -11.4154654, -12.3642693, -11.4273510, -0.4061408, 0.4075487
7: -6.4403133, -5.4925485, -6.4374523, -5.4970889, -0.3257120, 0.3262676
8: 2.1045647, 3.0154772, 2.1065252, 3.0140798, -0.6183171, 0.6177974
9: -6.2751656, -5.3168564, -6.2696619, -5.3183041, -0.5416837, 0.5369501

Time for backsubstitution: 21.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 525

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 525

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1389257, upper bound: 0.1419753
time: 2.89 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1402267, upper bound: 0.1419756
time: 3.19 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -12.1527719, -11.1195717, -12.1527719, -11.1195717, -0.4930198, 0.4930201
1: -10.3068037, -9.5189619, -10.3068037, -9.5189619, -0.3312047, 0.3312048
2: -2.5511589, -1.7496777, -2.5511589, -1.7496777, -0.4219306, 0.4219306
3: 5.9629707, 6.7457466, 5.9629707, 6.7457466, -0.3250833, 0.3250835
4: -11.1846523, -10.2469454, -11.1846523, -10.2469454, -0.3573973, 0.3573973
5: -6.6111660, -5.8348808, -6.6111660, -5.8348808, -0.3637714, 0.3637714
6: -12.3698730, -11.4154654, -12.3698730, -11.4154654, -0.4067981, 0.4067980
7: -6.4403133, -5.4925485, -6.4403133, -5.4925485, -0.3251448, 0.3251445
8: 2.1045647, 3.0154772, 2.1045647, 3.0154772, -0.6193967, 0.6193962
9: -6.2751656, -5.3168564, -6.2751656, -5.3168564, -0.5403013, 0.5403013

Time for backsubstitution: 21.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 525

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 525

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1389257, upper bound: 0.1419761
time: 3.36 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1402267, upper bound: 0.1419758
time: 3.79 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 29.13 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 29.13
Output dim: 3, lower bound: -0.1389259, upper bound: 0.1402276
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.13
Output dim: 3, lower bound: -0.1402269, upper bound: 0.1402278
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.13
Output dim: 3, lower bound: -0.1389259, upper bound: 0.1402276
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.13
Output dim: 3, lower bound: -0.1402269, upper bound: 0.1402278
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 29.13
Output dim: 3, lower bound: -0.1389257, upper bound: 0.1419753
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.13
Output dim: 3, lower bound: -0.1402267, upper bound: 0.1419756
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.13
Output dim: 3, lower bound: -0.1389257, upper bound: 0.1419761
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.13
Output dim: 3, lower bound: -0.1402267, upper bound: 0.1419758

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

Time for backsubstitution: 21.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 525

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 525

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1389259, upper bound: 0.1389259
time: 3.01 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1389259, upper bound: 0.1402278
time: 2.92 seconds

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

Time for backsubstitution: 21.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 525

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 525

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1402278, upper bound: 0.1389259
time: 2.91 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1402278, upper bound: 0.1402278
time: 2.90 seconds

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

Time for backsubstitution: 21.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 525

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 525

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1406735, upper bound: 0.1389257
time: 3.17 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1406735, upper bound: 0.1402276
time: 3.15 seconds

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

Time for backsubstitution: 21.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 525

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 525

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1419754, upper bound: 0.1389257
time: 3.04 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1419754, upper bound: 0.1402276
time: 2.91 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -12.1526108, -11.1196709, -12.1455307, -11.1228237, -0.4936886, 0.4909658
1: -10.3067951, -9.5210352, -10.2951880, -9.5249643, -0.3314102, 0.3312941
2: -2.5511215, -1.7513871, -2.5453322, -1.7536416, -0.4169750, 0.4140973
3: 5.9641218, 6.7456779, 5.9724970, 6.7407489, -0.3220878, 0.3231421
4: -11.1844425, -10.2470083, -11.1790676, -10.2524586, -0.3540339, 0.3541956
5: -6.6108856, -5.8349075, -6.6033220, -5.8435459, -0.3552291, 0.3537844
6: -12.3691435, -11.4155798, -12.3642693, -11.4273510, -0.4053979, 0.4073780
7: -6.4403090, -5.4937897, -6.4374523, -5.4970889, -0.3256986, 0.3249737
8: 2.1048281, 3.0140877, 2.1065252, 3.0140798, -0.6180959, 0.6163902
9: -6.2750893, -5.3179684, -6.2696619, -5.3183041, -0.5416341, 0.5358243

Time for backsubstitution: 21.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 525

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 525

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1389257, upper bound: 0.1406735
time: 2.91 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1389257, upper bound: 0.1419753
time: 2.91 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -12.1528101, -11.1176567, -12.1455307, -11.1228247, -0.4939923, 0.4929979
1: -10.3307714, -9.5186872, -10.2951870, -9.5249701, -0.3317453, 0.3348334
2: -2.5730920, -1.7496326, -2.5453305, -1.7536483, -0.4197726, 0.4173648
3: 5.9628849, 6.7620339, 5.9725018, 6.7407494, -0.3237598, 0.3262303
4: -11.1848621, -10.2411547, -11.1790686, -10.2524586, -0.3555081, 0.3608999
5: -6.6113939, -5.8330517, -6.6033201, -5.8435454, -0.3558605, 0.3548677
6: -12.3706264, -11.4065723, -12.3642664, -11.4273529, -0.4080782, 0.4078128
7: -6.4542794, -5.4919758, -6.4374542, -5.4970942, -0.3282306, 0.3267647
8: 2.0803268, 3.0155435, 2.1065254, 3.0140738, -0.6298618, 0.6189618
9: -6.2878695, -5.3145771, -6.2696624, -5.3183050, -0.5447111, 0.5403090

Time for backsubstitution: 21.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 525

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 525

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1402276, upper bound: 0.1406734
time: 3.12 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1402276, upper bound: 0.1419754
time: 3.06 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -12.1526108, -11.1196709, -12.1527719, -11.1195717, -0.4928544, 0.4929256
1: -10.3067951, -9.5210352, -10.3068037, -9.5189619, -0.3311708, 0.3291117
2: -2.5511215, -1.7513871, -2.5511589, -1.7496777, -0.4219043, 0.4202106
3: 5.9641218, 6.7456779, 5.9629707, 6.7457466, -0.3239300, 0.3250518
4: -11.1844425, -10.2470083, -11.1846523, -10.2469454, -0.3571262, 0.3573651
5: -6.6108856, -5.8349075, -6.6111660, -5.8348808, -0.3635044, 0.3637490
6: -12.3691435, -11.4155798, -12.3698730, -11.4154654, -0.4060552, 0.4066342
7: -6.4403090, -5.4937897, -6.4403133, -5.4925485, -0.3251314, 0.3238534
8: 2.1048281, 3.0140877, 2.1045647, 3.0154772, -0.6191754, 0.6179886
9: -6.2750893, -5.3179684, -6.2751656, -5.3168564, -0.5402517, 0.5391765

Time for backsubstitution: 21.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 525

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 525

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1389257, upper bound: 0.1406745
time: 3.17 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1389257, upper bound: 0.1419763
time: 3.13 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -12.1528101, -11.1176567, -12.1527729, -11.1195765, -0.4931598, 0.4949572
1: -10.3307714, -9.5186872, -10.3068027, -9.5189667, -0.3358828, 0.3326581
2: -2.5730920, -1.7496326, -2.5511599, -1.7496841, -0.4266875, 0.4234784
3: 5.9628849, 6.7620339, 5.9629745, 6.7457433, -0.3259530, 0.3290044
4: -11.1848621, -10.2411547, -11.1846504, -10.2469454, -0.3589852, 0.3633530
5: -6.6113939, -5.8330517, -6.6111650, -5.8348823, -0.3641357, 0.3658051
6: -12.3706264, -11.4065723, -12.3698711, -11.4154654, -0.4087355, 0.4128561
7: -6.4542794, -5.4919758, -6.4403133, -5.4925542, -0.3286163, 0.3263462
8: 2.0803268, 3.0155435, 2.1045661, 3.0154719, -0.6312060, 0.6205597
9: -6.2878695, -5.3145771, -6.2751651, -5.3168573, -0.5461965, 0.5436611

Time for backsubstitution: 21.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 525

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 525

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1402276, upper bound: 0.1406742
time: 3.09 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1402276, upper bound: 0.1419760
time: 3.33 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 28.47 seconds
NS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 28.47
Output dim: 3, lower bound: -0.1389259, upper bound: 0.1389259
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.47
Output dim: 3, lower bound: -0.1389259, upper bound: 0.1402278
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.47
Output dim: 3, lower bound: -0.1402278, upper bound: 0.1389259
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.47
Output dim: 3, lower bound: -0.1402278, upper bound: 0.1402278
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.47
Output dim: 3, lower bound: -0.1406735, upper bound: 0.1389257
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.47
Output dim: 3, lower bound: -0.1406735, upper bound: 0.1402276
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.47
Output dim: 3, lower bound: -0.1419754, upper bound: 0.1389257
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.47
Output dim: 3, lower bound: -0.1419754, upper bound: 0.1402276
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.47
Output dim: 3, lower bound: -0.1389257, upper bound: 0.1406735
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.47
Output dim: 3, lower bound: -0.1389257, upper bound: 0.1419753
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.47
Output dim: 3, lower bound: -0.1402276, upper bound: 0.1406734
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.47
Output dim: 3, lower bound: -0.1402276, upper bound: 0.1419754
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.47
Output dim: 3, lower bound: -0.1389257, upper bound: 0.1406745
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.47
Output dim: 3, lower bound: -0.1389257, upper bound: 0.1419763
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.47
Output dim: 3, lower bound: -0.1402276, upper bound: 0.1406742
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.47
Output dim: 3, lower bound: -0.1402276, upper bound: 0.1419760

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -12.1453676, -11.1229219, -12.1455708, -11.1209059, -0.4881699, 0.4863753
1: -10.2951784, -9.5270414, -10.3191547, -9.5246925, -0.3267415, 0.3291292
2: -2.5452919, -1.7553501, -2.5672626, -1.7535970, -0.4117236, 0.4153070
3: 5.9736490, 6.7406821, 5.9724112, 6.7570376, -0.3224063, 0.3182027
4: -11.1788597, -10.2525206, -11.1792793, -10.2466688, -0.3548403, 0.3485920
5: -6.6030397, -5.8435707, -6.6035490, -5.8417149, -0.3483696, 0.3469172
6: -12.3635387, -11.4274654, -12.3650217, -11.4184599, -0.4051630, 0.4006597
7: -6.4374456, -5.4983320, -6.4514203, -5.4965162, -0.3230116, 0.3249191
8: 2.1067882, 3.0126905, 2.0822849, 3.0141463, -0.6164274, 0.6266842
9: -6.2695875, -5.3194156, -6.2823653, -5.3160243, -0.5382237, 0.5409129

Time for backsubstitution: 21.56 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 2579
type: A, layer: 3, pos: 2376
type: A, layer: 3, pos: 2147
type: A, layer: 3, pos: 2229
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 746
type: A, layer: 3, pos: 2565
type: A, layer: 3, pos: 2228
type: A, layer: 3, pos: 915
type: A, layer: 3, pos: 674
type: A, layer: 3, pos: 2458
type: A, layer: 3, pos: 717
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 205
type: A, layer: 3, pos: 2620
type: A, layer: 3, pos: 1382
type: A, layer: 3, pos: 1998
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 628
type: A, layer: 3, pos: 1729

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 3, pos: 2536

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1381216, upper bound: 0.1349124
time: 2.99 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1381278, upper bound: 0.1394292
time: 2.97 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -12.1455708, -11.1209059, -12.1453676, -11.1229219, -0.4863753, 0.4881697
1: -10.3191547, -9.5246925, -10.2951784, -9.5270414, -0.3291291, 0.3267415
2: -2.5672626, -1.7535970, -2.5452919, -1.7553501, -0.4153073, 0.4117236
3: 5.9724112, 6.7570376, 5.9736490, 6.7406821, -0.3182025, 0.3224063
4: -11.1792793, -10.2466688, -11.1788597, -10.2525206, -0.3485920, 0.3548405
5: -6.6035490, -5.8417149, -6.6030397, -5.8435707, -0.3469172, 0.3483696
6: -12.3650217, -11.4184599, -12.3635387, -11.4274654, -0.4006596, 0.4051630
7: -6.4514203, -5.4965162, -6.4374456, -5.4983320, -0.3249191, 0.3230118
8: 2.0822849, 3.0141463, 2.1067882, 3.0126905, -0.6266837, 0.6164269
9: -6.2823653, -5.3160243, -6.2695875, -5.3194156, -0.5409126, 0.5382237

Time for backsubstitution: 21.66 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 2579
type: A, layer: 3, pos: 2376
type: A, layer: 3, pos: 2147
type: A, layer: 3, pos: 2229
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 746
type: A, layer: 3, pos: 2565
type: A, layer: 3, pos: 2228
type: A, layer: 3, pos: 915
type: A, layer: 3, pos: 674
type: A, layer: 3, pos: 717
type: A, layer: 3, pos: 2458
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 205
type: A, layer: 3, pos: 2620
type: A, layer: 3, pos: 1382
type: A, layer: 3, pos: 1998
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 628
type: A, layer: 3, pos: 1729

Time for candidate selection: 0.40 seconds

### Candidate
type: A, layer: 3, pos: 2536

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1394225, upper bound: 0.1336106
time: 2.99 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1394288, upper bound: 0.1381271
time: 3.13 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -12.1455708, -11.1209059, -12.1455708, -11.1209059, -0.4871995, 0.4871993
1: -10.3191547, -9.5246925, -10.3191547, -9.5246925, -0.3283486, 0.3283485
2: -2.5672626, -1.7535970, -2.5672626, -1.7535970, -0.4147308, 0.4147308
3: 5.9724112, 6.7570376, 5.9724112, 6.7570376, -0.3227358, 0.3227358
4: -11.1792793, -10.2466688, -11.1792793, -10.2466688, -0.3537302, 0.3537302
5: -6.6035490, -5.8417149, -6.6035490, -5.8417149, -0.3485131, 0.3485131
6: -12.3650217, -11.4184599, -12.3650217, -11.4184599, -0.4020844, 0.4020842
7: -6.4514203, -5.4965162, -6.4514203, -5.4965162, -0.3240137, 0.3240138
8: 2.0822849, 3.0141463, 2.0822849, 3.0141463, -0.6268363, 0.6268363
9: -6.2823653, -5.3160243, -6.2823653, -5.3160243, -0.5412722, 0.5412722

Time for backsubstitution: 21.71 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 2579
type: A, layer: 3, pos: 2376
type: A, layer: 3, pos: 2147
type: A, layer: 3, pos: 2229
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 746
type: A, layer: 3, pos: 2565
type: A, layer: 3, pos: 2228
type: A, layer: 3, pos: 915
type: A, layer: 3, pos: 674
type: A, layer: 3, pos: 717
type: A, layer: 3, pos: 2458
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 205
type: A, layer: 3, pos: 2620
type: A, layer: 3, pos: 1382
type: A, layer: 3, pos: 1998
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 628
type: A, layer: 3, pos: 1729

Time for candidate selection: 0.43 seconds

### Candidate
type: A, layer: 3, pos: 2536

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1394234, upper bound: 0.1336114
time: 3.20 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1394297, upper bound: 0.1381279
time: 3.30 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -12.1453676, -11.1229219, -12.1526108, -11.1196709, -0.4908018, 0.4935939
1: -10.2951784, -9.5270414, -10.3067951, -9.5210352, -0.3312600, 0.3293139
2: -2.5452919, -1.7553501, -2.5511215, -1.7513871, -0.4140706, 0.4152558
3: 5.9736490, 6.7406821, 5.9641218, 6.7456779, -0.3219883, 0.3220557
4: -11.1788597, -10.2525206, -11.1844425, -10.2470083, -0.3539243, 0.3540022
5: -6.6030397, -5.8435707, -6.6108856, -5.8349075, -0.3535061, 0.3552065
6: -12.3635387, -11.4274654, -12.3691435, -11.4155798, -0.4066358, 0.4052342
7: -6.4374456, -5.4983320, -6.4403090, -5.4937897, -0.3249604, 0.3244076
8: 2.1067882, 3.0126905, 2.1048281, 3.0140877, -0.6161695, 0.6166902
9: -6.2695875, -5.3194156, -6.2750893, -5.3179684, -0.5357728, 0.5405087

Time for backsubstitution: 21.75 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 2579
type: A, layer: 3, pos: 2376
type: A, layer: 3, pos: 2147
type: A, layer: 3, pos: 2229
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 746
type: A, layer: 3, pos: 2565
type: A, layer: 3, pos: 2228
type: A, layer: 3, pos: 915
type: A, layer: 3, pos: 674
type: A, layer: 3, pos: 2458
type: A, layer: 3, pos: 717
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 205
type: A, layer: 3, pos: 2620
type: A, layer: 3, pos: 1382
type: A, layer: 3, pos: 1998
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 628
type: A, layer: 3, pos: 1729

Time for candidate selection: 0.41 seconds

### Candidate
type: A, layer: 3, pos: 2536

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1398691, upper bound: 0.1336105
time: 3.22 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1398754, upper bound: 0.1381271
time: 2.92 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -12.1453676, -11.1229219, -12.1528101, -11.1176567, -0.4928346, 0.4938326
1: -10.2951784, -9.5270414, -10.3307714, -9.5186872, -0.3335474, 0.3296498
2: -2.5452919, -1.7553501, -2.5730920, -1.7496326, -0.4160352, 0.4180489
3: 5.9736490, 6.7406821, 5.9628849, 6.7620339, -0.3250744, 0.3233651
4: -11.1788597, -10.2525206, -11.1848621, -10.2411547, -0.3606296, 0.3544602
5: -6.6030397, -5.8435707, -6.6113939, -5.8330517, -0.3545899, 0.3558121
6: -12.3635387, -11.4274654, -12.3706264, -11.4065723, -0.4070706, 0.4069371
7: -6.4374456, -5.4983320, -6.4542794, -5.4919758, -0.3267487, 0.3269373
8: 2.1067882, 3.0126905, 2.0803268, 3.0155435, -0.6179156, 0.6284523
9: -6.2695875, -5.3194156, -6.2878695, -5.3145771, -0.5396967, 0.5435784

Time for backsubstitution: 21.81 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 2579
type: A, layer: 3, pos: 2376
type: A, layer: 3, pos: 2147
type: A, layer: 3, pos: 2229
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 746
type: A, layer: 3, pos: 2565
type: A, layer: 3, pos: 2228
type: A, layer: 3, pos: 915
type: A, layer: 3, pos: 674
type: A, layer: 3, pos: 2458
type: A, layer: 3, pos: 717
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 205
type: A, layer: 3, pos: 2620
type: A, layer: 3, pos: 1382
type: A, layer: 3, pos: 1998
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 628
type: A, layer: 3, pos: 1729

Time for candidate selection: 0.37 seconds

### Candidate
type: A, layer: 3, pos: 2536

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1398691, upper bound: 0.1349123
time: 3.18 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1398754, upper bound: 0.1394290
time: 3.07 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -12.1455708, -11.1209059, -12.1526108, -11.1196709, -0.4910398, 0.4944458
1: -10.3191547, -9.5246925, -10.3067951, -9.5210352, -0.3327941, 0.3316128
2: -2.5672626, -1.7535970, -2.5511215, -1.7513871, -0.4186409, 0.4172211
3: 5.9724112, 6.7570376, 5.9641218, 6.7456779, -0.3232992, 0.3239439
4: -11.1792793, -10.2466688, -11.1844425, -10.2470083, -0.3543816, 0.3578441
5: -6.6035490, -5.8417149, -6.6108856, -5.8349075, -0.3540893, 0.3572638
6: -12.3650217, -11.4184599, -12.3691435, -11.4155798, -0.4083753, 0.4084996
7: -6.4514203, -5.4965162, -6.4403090, -5.4937897, -0.3252554, 0.3261863
8: 2.0822849, 3.0141463, 2.1048281, 3.0140877, -0.6278467, 0.6184359
9: -6.2823653, -5.3160243, -6.2750893, -5.3179684, -0.5415893, 0.5444345

Time for backsubstitution: 21.73 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.00 + 562.61 = 618.61 seconds
