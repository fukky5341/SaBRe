## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.5754619315


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.6113167, -7.0396461, -8.6113167, -7.0396461, -1.1536942, 1.1536946)
1: (1.2912707, 2.5321927, 1.2912707, 2.5321927, -1.1601706, 1.1601706)
2: (-6.3139620, -5.1013927, -6.3139620, -5.1013927, -0.9770384, 0.9770386)
3: (-10.5567799, -8.9935284, -10.5567799, -8.9935284, -1.1204410, 1.1204410)
4: (-4.5573525, -3.2772324, -4.5573525, -3.2772324, -0.9474130, 0.9474130)
5: (-8.6873465, -7.4558282, -8.6873465, -7.4558282, -1.0102134, 1.0102134)
6: (-8.2650509, -6.7415175, -8.2650509, -6.7415175, -1.0024719, 1.0024717)
7: (-7.3879914, -6.3009577, -7.3879914, -6.3009577, -0.8303411, 0.8303411)
8: (-0.2403114, 1.0873740, -0.2403114, 1.0873740, -1.3009887, 1.3009882)
9: (-5.1309075, -4.0237150, -5.1309075, -4.0237150, -0.7976398, 0.7976398)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.01 + 34.38 = 57.39 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.5783445, upper bound: 0.5783522

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4642
type: B, layer: 1, pos: 4642
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 5760
type: B, layer: 1, pos: 5760
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 5816
type: A, layer: 1, pos: 5816
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 4642

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5728968, upper bound: 0.5783455
time: 3.79 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5783446, upper bound: 0.5783444
time: 4.14 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 8.20 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 8.20
Output dim: 1, lower bound: -0.5728968, upper bound: 0.5783455
NS_A2, status: Status.UNKNOWN, split count: 1, time: 8.20
Output dim: 1, lower bound: -0.5783446, upper bound: 0.5783444

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -8.6106024, -7.0397201, -8.6112680, -7.0396519, -1.1523714, 1.1530552
1: 1.2983065, 2.5320759, 1.2917676, 2.5321827, -1.1531439, 1.1595483
2: -6.3137093, -5.1015682, -6.3139443, -5.1014051, -0.9759316, 0.9758646
3: -10.5567017, -9.0012035, -10.5567741, -8.9940691, -1.1198301, 1.1127493
4: -4.5509148, -3.2773638, -4.5568981, -3.2772417, -0.9400978, 0.9463391
5: -8.6856604, -7.4559546, -8.6872292, -7.4558382, -1.0078497, 1.0098858
6: -8.2648315, -6.7505183, -8.2650356, -6.7421527, -1.0015984, 0.9934707
7: -7.3810759, -6.3011847, -7.3875031, -6.3009734, -0.8227181, 0.8294833
8: -0.2398896, 1.0796328, -0.2402811, 1.0868275, -1.2977719, 1.2900782
9: -5.1227365, -4.0238576, -5.1303296, -4.0237255, -0.7893972, 0.7969041

Time for backsubstitution: 21.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 4642
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 5760
type: A, layer: 1, pos: 5760
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 5816

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 832

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5723134, upper bound: 0.5685361
time: 3.69 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5723134, upper bound: 0.5777461
time: 3.66 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -8.6138000, -7.0364122, -8.6113176, -7.0396481, -1.1549129, 1.1580765
1: 1.2892151, 2.5511198, 1.2912803, 2.5321903, -1.1615839, 1.1673455
2: -6.3156705, -5.0988693, -6.3139615, -5.1013947, -0.9779494, 0.9817681
3: -10.5776348, -8.9911413, -10.5567808, -8.9935379, -1.1290307, 1.1221800
4: -4.5583458, -3.2592928, -4.5573444, -3.2772319, -0.9504676, 0.9578099
5: -8.6907940, -7.4491272, -8.6873446, -7.4558287, -1.0162973, 1.0167947
6: -8.2924948, -6.7410131, -8.2650490, -6.7415280, -1.0147429, 1.0023808
7: -7.3888068, -6.2798338, -7.3879828, -6.3009567, -0.8323729, 0.8381741
8: -0.2659402, 1.0883899, -0.2403114, 1.0873702, -1.3238578, 1.3086848
9: -5.1315789, -4.0012808, -5.1308961, -4.0237141, -0.7972763, 0.8080195

Time for backsubstitution: 21.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 4642
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5760
type: B, layer: 1, pos: 5760
type: B, layer: 1, pos: 5816
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 5816

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 832

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5777447, upper bound: 0.5685348
time: 3.80 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5777447, upper bound: 0.5777462
time: 3.65 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 28.79 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 28.79
Output dim: 1, lower bound: -0.5723134, upper bound: 0.5685361
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 28.79
Output dim: 1, lower bound: -0.5723134, upper bound: 0.5777461
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 28.79
Output dim: 1, lower bound: -0.5777447, upper bound: 0.5685348
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 28.79
Output dim: 1, lower bound: -0.5777447, upper bound: 0.5777462

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -8.6106033, -7.0397215, -8.6112671, -7.0396595, -1.1490993, 1.1528349
1: 1.2983098, 2.5320768, 1.2917781, 2.5321827, -1.1519217, 1.1500340
2: -6.3137093, -5.1015697, -6.3139429, -5.1014075, -0.9704151, 0.9749484
3: -10.5566998, -9.0012140, -10.5567741, -8.9940910, -1.1048279, 1.1125937
4: -4.5509124, -3.2773652, -4.5568943, -3.2772481, -0.9400887, 0.9401436
5: -8.6856546, -7.4559536, -8.6872063, -7.4558377, -1.0078449, 1.0038009
6: -8.2648268, -6.7505202, -8.2650261, -6.7421532, -1.0011611, 0.9776933
7: -7.3810763, -6.3011889, -7.3875022, -6.3009949, -0.8062329, 0.8259944
8: -0.2398860, 1.0796309, -0.2402706, 1.0868244, -1.2925568, 1.2900515
9: -5.1227384, -4.0238605, -5.1303301, -4.0237360, -0.7812994, 0.7969012

Time for backsubstitution: 21.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4642
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 5760
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 5816
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5760
type: A, layer: 1, pos: 5746

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 4642

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5723134, upper bound: 0.5723237
time: 3.99 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5723134, upper bound: 0.5777461
time: 3.96 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -8.6135187, -7.0370536, -8.6075315, -7.0414810, -1.1528192, 1.1535735
1: 1.2910290, 2.5510125, 1.2966886, 2.5262127, -1.1518154, 1.1618223
2: -6.3154392, -5.0998359, -6.3099995, -5.1043425, -0.9747803, 0.9755583
3: -10.5776281, -8.9951973, -10.5481157, -9.0055065, -1.1171598, 1.1092844
4: -4.5562425, -3.2593038, -4.5510664, -3.2825346, -0.9430008, 0.9515371
5: -8.6889591, -7.4491549, -8.6809940, -7.4602375, -1.0100727, 1.0103135
6: -8.2878857, -6.7410750, -8.2512741, -6.7524228, -0.9964950, 0.9885480
7: -7.3887520, -6.2845669, -7.3780184, -6.3150997, -0.8183150, 0.8207973
8: -0.2657671, 1.0869505, -0.2354617, 1.0829179, -1.3192387, 1.3023639
9: -5.1314573, -4.0038462, -5.1229644, -4.0312791, -0.7896564, 0.7961693

Time for backsubstitution: 21.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 4642
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 5760
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5816
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 5760
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 5816

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 846

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5708320, upper bound: 0.5681192
time: 3.70 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5773101, upper bound: 0.5681179
time: 3.91 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -8.6138010, -7.0364141, -8.6113157, -7.0396504, -1.1516399, 1.1578579
1: 1.2892179, 2.5511184, 1.2912889, 2.5321927, -1.1603627, 1.1619711
2: -6.3156705, -5.0988712, -6.3139601, -5.1013966, -0.9724336, 0.9808519
3: -10.5776348, -8.9911470, -10.5567799, -8.9935589, -1.1144531, 1.1221297
4: -4.5583429, -3.2592924, -4.5573411, -3.2772381, -0.9504590, 0.9511502
5: -8.6907854, -7.4491248, -8.6873255, -7.4558287, -1.0162921, 1.0107112
6: -8.2924938, -6.7410145, -8.2650414, -6.7415304, -1.0045614, 0.9866033
7: -7.3888092, -6.2798409, -7.3879819, -6.3009791, -0.8158872, 0.8284056
8: -0.2659380, 1.0883889, -0.2402997, 1.0873625, -1.3186412, 1.3086596
9: -5.1315775, -4.0012856, -5.1308961, -4.0237265, -0.7891788, 0.8023355

Time for backsubstitution: 20.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 4642
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 5760
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 5816
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5760
type: A, layer: 1, pos: 5746

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 846

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5773102, upper bound: 0.5708335
time: 3.64 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5773102, upper bound: 0.5773105
time: 3.79 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 28.28 seconds
NS_A1_B2_B1, status: Status.VERIFIED, split count: 3, time: 28.28
Output dim: 1, lower bound: -0.5723134, upper bound: 0.5723237
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 28.28
Output dim: 1, lower bound: -0.5723134, upper bound: 0.5777461
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 28.28
Output dim: 1, lower bound: -0.5708320, upper bound: 0.5681192
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.28
Output dim: 1, lower bound: -0.5773101, upper bound: 0.5681179
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 28.28
Output dim: 1, lower bound: -0.5773102, upper bound: 0.5708335
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 28.28
Output dim: 1, lower bound: -0.5773102, upper bound: 0.5773105

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -8.6106033, -7.0397215, -8.6137981, -7.0364189, -1.1518879, 1.1541433
1: 1.2983098, 2.5320768, 1.2892256, 2.5511189, -1.1550088, 1.1523700
2: -6.3137093, -5.1015697, -6.3156695, -5.0988727, -0.9720192, 0.9759338
3: -10.5566998, -9.0012140, -10.5776348, -8.9911652, -1.1076770, 1.1126890
4: -4.5509124, -3.2773652, -4.5583401, -3.2592986, -0.9458477, 0.9410753
5: -8.6856546, -7.4559536, -8.6907711, -7.4491258, -1.0144410, 1.0073180
6: -8.2648268, -6.7505202, -8.2924862, -6.7410164, -1.0019035, 0.9894905
7: -7.3810763, -6.3011889, -7.3888073, -6.2798543, -0.8139834, 0.8263223
8: -0.2398860, 1.0796309, -0.2659304, 1.0883863, -1.2912598, 1.3130999
9: -5.1227384, -4.0238605, -5.1315794, -4.0012932, -0.7912476, 0.7978840

Time for backsubstitution: 21.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 5760
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 5816
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5760

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 846

## Relational analysis of NS_A1_B2_B2_B1

### Relational analysis result of NS_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5718881, upper bound: 0.5708325
time: 3.86 seconds

## Relational analysis of NS_A1_B2_B2_B2

### Relational analysis result of NS_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5718881, upper bound: 0.5773115
time: 3.71 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -8.6130962, -7.0382633, -8.6074066, -7.0418358, -1.1521392, 1.1526165
1: 1.2912703, 2.5510106, 1.2967563, 2.5262122, -1.1439195, 1.1576047
2: -6.3154364, -5.0999818, -6.3099990, -5.1043854, -0.9714317, 0.9660239
3: -10.5773458, -8.9952717, -10.5480299, -9.0055275, -1.1168699, 1.1091087
4: -4.5560594, -3.2597013, -4.5510139, -3.2826514, -0.9417982, 0.9512603
5: -8.6879253, -7.4493341, -8.6800413, -7.4602890, -1.0145860, 1.0086770
6: -8.2878857, -6.7412033, -8.2512741, -6.7524595, -0.9935136, 0.9832995
7: -7.3887038, -6.2847338, -7.3780050, -6.3151484, -0.8171911, 0.8179352
8: -0.2606912, 1.0869458, -0.2339749, 1.0829160, -1.3161292, 1.3007588
9: -5.1306849, -4.0041089, -5.1227355, -4.0313549, -0.7915008, 0.7948713

Time for backsubstitution: 22.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4642
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 5760
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 5816
type: B, layer: 1, pos: 5760

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 1, pos: 4642

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5773100, upper bound: 0.5626936
time: 3.91 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5773109, upper bound: 0.5626934
time: 3.76 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: -8.6136236, -7.0364561, -8.6100960, -7.0401134, -1.1509361, 1.1565032
1: 1.2903781, 2.5509262, 1.2944455, 2.5282278, -1.1523566, 1.1585121
2: -6.3152809, -5.0990472, -6.3094721, -5.1019320, -0.9710672, 0.9707592
3: -10.5776310, -8.9917898, -10.5564880, -8.9956741, -1.1121871, 1.1207812
4: -4.5581045, -3.2593389, -4.5567226, -3.2777214, -0.9490542, 0.9497736
5: -8.6906309, -7.4491806, -8.6861200, -7.4539986, -1.0176907, 1.0088935
6: -8.2923555, -6.7422528, -8.2612667, -6.7456474, -1.0002325, 0.9812562
7: -7.3886509, -6.2810264, -7.3848710, -6.3048110, -0.8118329, 0.8225651
8: -0.2656837, 1.0883243, -0.2388823, 1.0936320, -1.3142161, 1.3066425
9: -5.1315103, -4.0013471, -5.1297150, -4.0231895, -0.7896805, 0.8002198

Time for backsubstitution: 22.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4642
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 5760
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 5816
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5760
type: A, layer: 1, pos: 5746

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 4642

## Relational analysis of NS_A2_B2_B1_B1

### Relational analysis result of NS_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5773102, upper bound: 0.5654041
time: 3.66 seconds

## Relational analysis of NS_A2_B2_B1_B2

### Relational analysis result of NS_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5773110, upper bound: 0.5654041
time: 3.85 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: -8.6136799, -7.0367670, -8.6108952, -7.0408602, -1.1506834, 1.1571779
1: 1.2892890, 2.5511193, 1.2915330, 2.5321908, -1.1585088, 1.1594219
2: -6.3156700, -5.0989132, -6.3139591, -5.1015420, -0.9628990, 0.9775035
3: -10.5775528, -8.9911699, -10.5564976, -8.9936371, -1.1141348, 1.1218412
4: -4.5582910, -3.2594094, -4.5571561, -3.2776325, -0.9495492, 0.9513588
5: -8.6897640, -7.4491777, -8.6863003, -7.4560099, -1.0146542, 1.0152211
6: -8.2924938, -6.7410522, -8.2650404, -6.7416582, -1.0001597, 0.9864788
7: -7.3887925, -6.2798867, -7.3879347, -6.3011446, -0.8095737, 0.8249569
8: -0.2644529, 1.0883884, -0.2352285, 1.0873599, -1.3170352, 1.3172088
9: -5.1313534, -4.0013614, -5.1301231, -4.0239887, -0.7886181, 0.8013561

Time for backsubstitution: 22.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5760
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4642
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 5816
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 5816
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5760
type: A, layer: 1, pos: 5746

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 5760

## Relational analysis of NS_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5746

## Relational analysis of NS_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4642

## Relational analysis of NS_A2_B2_B2_B1

### Relational analysis result of NS_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5773102, upper bound: 0.5718886
time: 3.65 seconds

## Relational analysis of NS_A2_B2_B2_B2

### Relational analysis result of NS_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5773110, upper bound: 0.5718887
time: 3.85 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 42.39 seconds
NS_A1_B2_B2_B1, status: Status.VERIFIED, split count: 4, time: 42.39
Output dim: 1, lower bound: -0.5718881, upper bound: 0.5708325
NS_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 42.39
Output dim: 1, lower bound: -0.5718881, upper bound: 0.5773115
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 42.39
Output dim: 1, lower bound: -0.5773100, upper bound: 0.5626936
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 42.39
Output dim: 1, lower bound: -0.5773109, upper bound: 0.5626934
NS_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 42.39
Output dim: 1, lower bound: -0.5773102, upper bound: 0.5654041
NS_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 42.39
Output dim: 1, lower bound: -0.5773110, upper bound: 0.5654041
NS_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 42.39
Output dim: 1, lower bound: -0.5773102, upper bound: 0.5718886
NS_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 42.39
Output dim: 1, lower bound: -0.5773110, upper bound: 0.5718887

## BFS NS instance: NS_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -8.6104774, -7.0400743, -8.6133823, -7.0376287, -1.1509304, 1.1534648
1: 1.2983799, 2.5320764, 1.2894673, 2.5511184, -1.1507902, 1.1444755
2: -6.3137093, -5.1016111, -6.3156681, -5.0990195, -0.9624848, 0.9725866
3: -10.5566158, -9.0012350, -10.5773544, -8.9912386, -1.1075013, 1.1123981
4: -4.5508604, -3.2774811, -4.5581570, -3.2596960, -0.9455719, 0.9398718
5: -8.6846409, -7.4560070, -8.6897430, -7.4493065, -1.0128040, 1.0118279
6: -8.2648287, -6.7505579, -8.2924843, -6.7411427, -0.9975083, 0.9865096
7: -7.3810630, -6.3012404, -7.3887582, -6.2800198, -0.8111184, 0.8228718
8: -0.2384014, 1.0796292, -0.2608533, 1.0883806, -1.2896557, 1.3065152
9: -5.1225109, -4.0239377, -5.1308069, -4.0015550, -0.7899494, 0.7997289

Time for backsubstitution: 22.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5760
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5760

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 5760

## Relational analysis of NS_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5746

## Relational analysis of NS_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 916

## Relational analysis of NS_A1_B2_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5611802, upper bound: 0.5762638
time: 4.40 seconds

## Relational analysis of NS_A1_B2_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5718802, upper bound: 0.5773024
time: 4.09 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -8.6130962, -7.0382633, -8.6066904, -7.0419087, -1.1515908, 1.1496992
1: 1.2912703, 2.5510106, 1.3037858, 2.5260987, -1.1442170, 1.1505666
2: -6.3154364, -5.0999818, -6.3097482, -5.1045589, -0.9703310, 0.9617920
3: -10.5773458, -8.9952717, -10.5479507, -9.0131969, -1.1091790, 1.1096075
4: -4.5560594, -3.2597013, -4.5445857, -3.2827830, -0.9385977, 0.9439800
5: -8.6879253, -7.4493341, -8.6783638, -7.4604115, -1.0116882, 1.0063348
6: -8.2878857, -6.7412033, -8.2510586, -6.7614508, -0.9845130, 0.9838414
7: -7.3887038, -6.2847338, -7.3710995, -6.3153734, -0.8150811, 0.8103261
8: -0.2606912, 1.0869458, -0.2335563, 1.0751786, -1.3053765, 1.2885480
9: -5.1306849, -4.0041089, -5.1145763, -4.0314980, -0.7921112, 0.7866294

Time for backsubstitution: 22.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 5760
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5760

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 916

## Relational analysis of NS_A2_B1_A2_B1_B1

### Relational analysis result of NS_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5762635, upper bound: 0.5521130
time: 3.74 seconds

## Relational analysis of NS_A2_B1_A2_B1_B2

### Relational analysis result of NS_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5773027, upper bound: 0.5626856
time: 4.32 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -8.6130962, -7.0382633, -8.6098986, -7.0386028, -1.1569872, 1.1542964
1: 1.2912703, 2.5510106, 1.2946982, 2.5451403, -1.1509953, 1.1594291
2: -6.3154364, -5.0999818, -6.3117065, -5.1018620, -0.9763427, 0.9671168
3: -10.5773458, -8.9952717, -10.5688877, -9.0031261, -1.1156118, 1.1141958
4: -4.5560594, -3.2597013, -4.5520201, -3.2647114, -0.9421225, 0.9436188
5: -8.6879253, -7.4493341, -8.6835117, -7.4535866, -1.0163441, 1.0099635
6: -8.2878857, -6.7412033, -8.2787399, -6.7519455, -0.9926810, 0.9893374
7: -7.3887038, -6.2847338, -7.3788295, -6.2940264, -0.8186188, 0.8127780
8: -0.2606912, 1.0869458, -0.2596059, 1.0839458, -1.3159022, 1.3025417
9: -5.1306849, -4.0041089, -5.1234255, -4.0089211, -0.7931566, 0.7879567

Time for backsubstitution: 22.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 5760
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 5816
type: A, layer: 1, pos: 5816
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5760

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 916

## Relational analysis of NS_A2_B1_A2_B2_B1

### Relational analysis result of NS_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5762643, upper bound: 0.5521131
time: 3.71 seconds

## Relational analysis of NS_A2_B1_A2_B2_B2

### Relational analysis result of NS_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5773035, upper bound: 0.5626869
time: 4.12 seconds

## BFS NS instance: NS_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -8.6136236, -7.0364561, -8.6093731, -7.0401859, -1.1503892, 1.1535783
1: 1.2903781, 2.5509262, 1.3014727, 2.5281138, -1.1526566, 1.1514764
2: -6.3152809, -5.0990472, -6.3092208, -5.1021061, -0.9699669, 0.9665258
3: -10.5776310, -8.9917898, -10.5564060, -9.0033417, -1.1044984, 1.1206150
4: -4.5581045, -3.2593389, -4.5502954, -3.2778547, -0.9458556, 0.9424894
5: -8.6906309, -7.4491806, -8.6844368, -7.4541211, -1.0147943, 1.0065455
6: -8.2923555, -6.7422528, -8.2610493, -6.7546377, -0.9912317, 0.9817946
7: -7.3886509, -6.2810264, -7.3779669, -6.3050370, -0.8097253, 0.8149567
8: -0.2656837, 1.0883243, -0.2384620, 1.0858963, -1.3034635, 1.2944336
9: -5.1315103, -4.0013471, -5.1215553, -4.0233321, -0.7902915, 0.7919779

Time for backsubstitution: 22.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 5760
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 5816
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 5816
type: A, layer: 1, pos: 5760
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5746

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 916

## Relational analysis of NS_A2_B2_B1_B1_A1

### Relational analysis result of NS_A2_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5666082, upper bound: 0.5643717
time: 3.77 seconds

## Relational analysis of NS_A2_B2_B1_B1_A2

### Relational analysis result of NS_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5773022, upper bound: 0.5653959
time: 3.94 seconds

## BFS NS instance: NS_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -8.6136236, -7.0364561, -8.6124992, -7.0368824, -1.1557860, 1.1581051
1: 1.2903781, 2.5509262, 1.2923789, 2.5471563, -1.1562343, 1.1556935
2: -6.3152809, -5.0990472, -6.3111753, -5.0994077, -0.9759822, 0.9718542
3: -10.5776310, -8.9917898, -10.5773430, -8.9932737, -1.1100180, 1.1208701
4: -4.5581045, -3.2593389, -4.5577283, -3.2597814, -0.9493780, 0.9439788
5: -8.6906309, -7.4491806, -8.6895733, -7.4472928, -1.0194468, 1.0101418
6: -8.2923555, -6.7422528, -8.2887077, -6.7451324, -1.0009551, 0.9872761
7: -7.3886509, -6.2810264, -7.3856945, -6.2836885, -0.8132613, 0.8235480
8: -0.2656837, 1.0883243, -0.2645109, 1.0946605, -1.3111520, 1.3084273
9: -5.1315103, -4.0013471, -5.1304002, -4.0007553, -0.7913363, 0.7971365

Time for backsubstitution: 22.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5760
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 5816
type: A, layer: 1, pos: 5760
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5746

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 916

## Relational analysis of NS_A2_B2_B1_B2_A1

### Relational analysis result of NS_A2_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5666091, upper bound: 0.5643717
time: 3.91 seconds

## Relational analysis of NS_A2_B2_B1_B2_A2

### Relational analysis result of NS_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5773031, upper bound: 0.5653968
time: 3.76 seconds

## BFS NS instance: NS_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -8.6136799, -7.0367670, -8.6101770, -7.0409327, -1.1501355, 1.1542628
1: 1.2892890, 2.5511193, 1.2985592, 2.5320764, -1.1588058, 1.1523871
2: -6.3156700, -5.0989132, -6.3137074, -5.1017170, -0.9617982, 0.9732721
3: -10.5775528, -8.9911699, -10.5564165, -9.0013027, -1.1064463, 1.1216753
4: -4.5582910, -3.2594094, -4.5507298, -3.2777662, -0.9463511, 0.9440753
5: -8.6897640, -7.4491777, -8.6846189, -7.4561357, -1.0117612, 1.0128732
6: -8.2924938, -6.7410522, -8.2648220, -6.7506475, -0.9911592, 0.9870183
7: -7.3887925, -6.2798867, -7.3810315, -6.3013692, -0.8074651, 0.8173475
8: -0.2644529, 1.0883884, -0.2348051, 1.0796211, -1.3063064, 1.3064976
9: -5.1313534, -4.0013614, -5.1219635, -4.0241313, -0.7892272, 0.7931135

Time for backsubstitution: 22.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5760
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 5816
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5760
type: A, layer: 1, pos: 5746

Time for candidate selection: 0.34 seconds

### Candidate
type: B, layer: 1, pos: 5760

### Candidate
type: B, layer: 1, pos: 5746

### Candidate
type: A, layer: 1, pos: 916

## Relational analysis of NS_A2_B2_B2_B1_A1

### Relational analysis result of NS_A2_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5666079, upper bound: 0.5708365
time: 3.92 seconds

## Relational analysis of NS_A2_B2_B2_B1_A2

### Relational analysis result of NS_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5773022, upper bound: 0.5718813
time: 3.80 seconds

## BFS NS instance: NS_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -8.6136799, -7.0367670, -8.6133823, -7.0376287, -1.1555300, 1.1588621
1: 1.2892890, 2.5511193, 1.2894673, 2.5511184, -1.1592259, 1.1512527
2: -6.3156700, -5.0989132, -6.3156681, -5.0990195, -0.9678111, 0.9785998
3: -10.5775528, -8.9911699, -10.5773544, -8.9912386, -1.1120908, 1.1219301
4: -4.5582910, -3.2594094, -4.5581570, -3.2596960, -0.9498730, 0.9433956
5: -8.6897640, -7.4491777, -8.6897430, -7.4493065, -1.0164108, 1.0164790
6: -8.2924938, -6.7410522, -8.2924843, -6.7411427, -1.0008802, 0.9925075
7: -7.3887925, -6.2798867, -7.3887582, -6.2800198, -0.8110054, 0.8259380
8: -0.2644529, 1.0883884, -0.2608533, 1.0883806, -1.3036442, 1.3190131
9: -5.1313534, -4.0013614, -5.1308069, -4.0015550, -0.7902734, 0.8007739

Time for backsubstitution: 22.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5760
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5760

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 5760

### Candidate
type: B, layer: 1, pos: 5746

### Candidate
type: A, layer: 1, pos: 916

## Relational analysis of NS_A2_B2_B2_B2_A1

### Relational analysis result of NS_A2_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5666088, upper bound: 0.5708365
time: 3.66 seconds

## Relational analysis of NS_A2_B2_B2_B2_A2

### Relational analysis result of NS_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5773031, upper bound: 0.5718813
time: 3.82 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 29.99 seconds
NS_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 29.99
Output dim: 1, lower bound: -0.5611802, upper bound: 0.5762638
NS_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 29.99
Output dim: 1, lower bound: -0.5718802, upper bound: 0.5773024
NS_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 29.99
Output dim: 1, lower bound: -0.5762635, upper bound: 0.5521130
NS_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 29.99
Output dim: 1, lower bound: -0.5773027, upper bound: 0.5626856
NS_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 29.99
Output dim: 1, lower bound: -0.5762643, upper bound: 0.5521131
NS_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 29.99
Output dim: 1, lower bound: -0.5773035, upper bound: 0.5626869
NS_A2_B2_B1_B1_A1, status: Status.VERIFIED, split count: 5, time: 29.99
Output dim: 1, lower bound: -0.5666082, upper bound: 0.5643717
NS_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 29.99
Output dim: 1, lower bound: -0.5773022, upper bound: 0.5653959
NS_A2_B2_B1_B2_A1, status: Status.VERIFIED, split count: 5, time: 29.99
Output dim: 1, lower bound: -0.5666091, upper bound: 0.5643717
NS_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 29.99
Output dim: 1, lower bound: -0.5773031, upper bound: 0.5653968
NS_A2_B2_B2_B1_A1, status: Status.VERIFIED, split count: 5, time: 29.99
Output dim: 1, lower bound: -0.5666079, upper bound: 0.5708365
NS_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 29.99
Output dim: 1, lower bound: -0.5773022, upper bound: 0.5718813
NS_A2_B2_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 29.99
Output dim: 1, lower bound: -0.5666088, upper bound: 0.5708365
NS_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 29.99
Output dim: 1, lower bound: -0.5773031, upper bound: 0.5718813

## BFS NS instance: NS_A1_B2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -8.6070585, -7.0432796, -8.6132603, -7.0386901, -1.1448827, 1.1498144
1: 1.3053951, 2.5270419, 1.2916870, 2.5507617, -1.1434193, 1.1372886
2: -6.3070698, -5.1092987, -6.3152347, -5.1016331, -0.9535332, 0.9645233
3: -10.5539579, -9.0040455, -10.5765038, -8.9914446, -1.1047068, 1.1083364
4: -4.5457015, -3.2809439, -4.5564904, -3.2597065, -0.9404459, 0.9346867
5: -8.6840992, -7.4574842, -8.6896000, -7.4495163, -1.0120645, 1.0100632
6: -8.2638865, -6.7508264, -8.2923183, -6.7412291, -0.9961843, 0.9855590
7: -7.3794765, -6.3023138, -7.3883286, -6.2802000, -0.8094277, 0.8212889
8: -0.2339051, 1.0742149, -0.2603977, 1.0866754, -1.2833681, 1.3007231
9: -5.1197677, -4.0269880, -5.1298184, -4.0017796, -0.7870369, 0.7956600

Time for backsubstitution: 22.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5760
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 5816
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 5816
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5760

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 5760

### Candidate
type: A, layer: 1, pos: 832

## Relational analysis of NS_A1_B2_B2_B2_A1_A1

### Relational analysis result of NS_A1_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5521128, upper bound: 0.5762631
time: 3.99 seconds

## Relational analysis of NS_A1_B2_B2_B2_A1_A2

### Relational analysis result of NS_A1_B2_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5521124, upper bound: 0.5671844
time: 4.21 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -8.6104784, -7.0400801, -8.6133814, -7.0376272, -1.1499052, 1.1469526
1: 1.2983923, 2.5320740, 1.2894697, 2.5511179, -1.1478481, 1.1444707
2: -6.3137054, -5.1016254, -6.3156662, -5.0990229, -0.9624782, 0.9691043
3: -10.5566149, -9.0012360, -10.5773525, -8.9912405, -1.1060963, 1.1106250
4: -4.5508504, -3.2774839, -4.5581541, -3.2596951, -0.9404595, 0.9398685
5: -8.6846409, -7.4560080, -8.6897430, -7.4493065, -1.0127721, 1.0120120
6: -8.2647591, -6.7505593, -8.2924519, -6.7411413, -0.9973652, 0.9863317
7: -7.3810611, -6.3012409, -7.3887563, -6.2800202, -0.8109455, 0.8225330
8: -0.2383978, 1.0796199, -0.2608519, 1.0883789, -1.2896514, 1.3040135
9: -5.1225028, -4.0239372, -5.1308055, -4.0015540, -0.7868354, 0.7982916

Time for backsubstitution: 22.38 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 57.39 + 563.16 = 620.55 seconds
