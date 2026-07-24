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
execution time: IAR + RelationalAnalysis = 23.75 + 35.10 = 58.84 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.5783445, upper bound: 0.5783522

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4642
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 5760
type: A, layer: 1, pos: 5844

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 4642

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5728968, upper bound: 0.5783455
time: 3.76 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5783446, upper bound: 0.5783444
time: 4.48 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 8.50 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 8.50
Output dim: 1, lower bound: -0.5728968, upper bound: 0.5783455
NS_A2, status: Status.UNKNOWN, split count: 1, time: 8.50
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

Time for backsubstitution: 21.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4642
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 5760
type: B, layer: 1, pos: 5844

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 1, pos: 4642

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5728968, upper bound: 0.5729045
time: 4.25 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5728968, upper bound: 0.5783455
time: 4.07 seconds

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

Time for backsubstitution: 22.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4642
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 5760
type: B, layer: 1, pos: 5844

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 4642

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5783364, upper bound: 0.5729060
time: 4.01 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5783364, upper bound: 0.5783455
time: 4.19 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 30.90 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 30.90
Output dim: 1, lower bound: -0.5728968, upper bound: 0.5729045
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 30.90
Output dim: 1, lower bound: -0.5728968, upper bound: 0.5783455
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 30.90
Output dim: 1, lower bound: -0.5783364, upper bound: 0.5729060
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 30.90
Output dim: 1, lower bound: -0.5783364, upper bound: 0.5783455

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -8.6106024, -7.0397201, -8.6138000, -7.0364122, -1.1551600, 1.1543646
1: 1.2983065, 2.5320759, 1.2892151, 2.5511198, -1.1603103, 1.1618829
2: -6.3137093, -5.1015682, -6.3156705, -5.0988693, -0.9775357, 0.9768493
3: -10.5567017, -9.0012035, -10.5776348, -8.9911413, -1.1226797, 1.1213412
4: -4.5509148, -3.2773638, -4.5583458, -3.2592928, -0.9505286, 0.9472704
5: -8.6856604, -7.4559546, -8.6907940, -7.4491272, -1.0144463, 1.0134025
6: -8.2648315, -6.7505183, -8.2924948, -6.7410131, -1.0029192, 1.0057416
7: -7.3810759, -6.3011847, -7.3888068, -6.2798338, -0.8305643, 0.8302653
8: -0.2398896, 1.0796328, -0.2659402, 1.0883899, -1.2964773, 1.3131270
9: -5.1227365, -4.0238576, -5.1315789, -4.0012808, -0.7997768, 0.7978864

Time for backsubstitution: 22.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 5760
type: A, layer: 1, pos: 5844

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 832

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5631024, upper bound: 0.5777463
time: 4.08 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5723223, upper bound: 0.5777454
time: 3.91 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -8.6138000, -7.0364122, -8.6106024, -7.0397201, -1.1543646, 1.1551597
1: 1.2892151, 2.5511198, 1.2983065, 2.5320759, -1.1618829, 1.1603103
2: -6.3156705, -5.0988693, -6.3137093, -5.1015682, -0.9768491, 0.9775357
3: -10.5776348, -8.9911413, -10.5567017, -9.0012035, -1.1213412, 1.1226797
4: -4.5583458, -3.2592928, -4.5509148, -3.2773638, -0.9472699, 0.9505286
5: -8.6907940, -7.4491272, -8.6856604, -7.4559546, -1.0134025, 1.0144458
6: -8.2924948, -6.7410131, -8.2648315, -6.7505183, -1.0057416, 1.0029192
7: -7.3888068, -6.2798338, -7.3810759, -6.3011847, -0.8302653, 0.8305643
8: -0.2659402, 1.0883899, -0.2398896, 1.0796328, -1.3131270, 1.2964773
9: -5.1315789, -4.0012808, -5.1227365, -4.0238576, -0.7978864, 0.7997767

Time for backsubstitution: 22.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 5760
type: A, layer: 1, pos: 5844

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 832

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5685343, upper bound: 0.5723232
time: 3.94 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5777444, upper bound: 0.5723235
time: 3.80 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -8.6138000, -7.0364122, -8.6138000, -7.0364122, -1.1597624, 1.1597624
1: 1.2892151, 2.5511198, 1.2892151, 2.5511198, -1.1686592, 1.1686592
2: -6.3156705, -5.0988693, -6.3156705, -5.0988693, -0.9828620, 0.9828620
3: -10.5776348, -8.9911413, -10.5776348, -8.9911413, -1.1272683, 1.1272683
4: -4.5583458, -3.2592928, -4.5583458, -3.2592928, -0.9507918, 0.9507918
5: -8.6907940, -7.4491272, -8.6907940, -7.4491272, -1.0180526, 1.0180521
6: -8.2924948, -6.7410131, -8.2924948, -6.7410131, -1.0084009, 1.0084009
7: -7.3888068, -6.2798338, -7.3888068, -6.2798338, -0.8338017, 0.8338017
8: -0.2659402, 1.0883899, -0.2659402, 1.0883899, -1.3104630, 1.3104630
9: -5.1315789, -4.0012808, -5.1315789, -4.0012808, -0.7989318, 0.7989318

Time for backsubstitution: 22.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 5760
type: A, layer: 1, pos: 5844

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 832

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5685352, upper bound: 0.5723232
time: 3.98 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5777452, upper bound: 0.5723235
time: 3.77 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 30.47 seconds
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 30.47
Output dim: 1, lower bound: -0.5631024, upper bound: 0.5777463
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 30.47
Output dim: 1, lower bound: -0.5723223, upper bound: 0.5777454
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 30.47
Output dim: 1, lower bound: -0.5685343, upper bound: 0.5723232
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 30.47
Output dim: 1, lower bound: -0.5777444, upper bound: 0.5723235
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 30.47
Output dim: 1, lower bound: -0.5685352, upper bound: 0.5723232
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 30.47
Output dim: 1, lower bound: -0.5777452, upper bound: 0.5723235

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -8.6068144, -7.0415540, -8.6135187, -7.0370536, -1.1506548, 1.1522701
1: 1.3037171, 2.5260983, 1.2910290, 2.5510125, -1.1547847, 1.1521139
2: -6.3097486, -5.1045160, -6.3154392, -5.0998359, -0.9713261, 0.9736807
3: -10.5480347, -9.0131760, -10.5776281, -8.9951973, -1.1097822, 1.1094699
4: -4.5446367, -3.2826662, -4.5562425, -3.2593038, -0.9442577, 0.9398036
5: -8.6793146, -7.4603605, -8.6889591, -7.4491549, -1.0079713, 1.0071731
6: -8.2510605, -6.7614141, -8.2878857, -6.7410750, -0.9890900, 0.9874938
7: -7.3711143, -6.3153248, -7.3887520, -6.2845669, -0.8131883, 0.8162069
8: -0.2350414, 1.0751808, -0.2657671, 1.0869505, -1.2901525, 1.3085108
9: -5.1148052, -4.0314202, -5.1314573, -4.0038462, -0.7879279, 0.7902665

Time for backsubstitution: 22.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 5760
type: B, layer: 1, pos: 5844

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 902

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5619040, upper bound: 0.5664850
time: 3.81 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5631037, upper bound: 0.5777375
time: 4.22 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -8.6106014, -7.0397239, -8.6138010, -7.0364141, -1.1549425, 1.1510911
1: 1.2983170, 2.5320773, 1.2892179, 2.5511184, -1.1549344, 1.1606622
2: -6.3137078, -5.1015711, -6.3156705, -5.0988712, -0.9766197, 0.9713318
3: -10.5566998, -9.0012274, -10.5776348, -8.9911470, -1.1219649, 1.1067643
4: -4.5509114, -3.2773700, -4.5583429, -3.2592924, -0.9438684, 0.9472609
5: -8.6856432, -7.4559531, -8.6907854, -7.4491248, -1.0083623, 1.0133986
6: -8.2648211, -6.7505226, -8.2924938, -6.7410145, -0.9871411, 0.9955602
7: -7.3810768, -6.3012037, -7.3888092, -6.2798409, -0.8207967, 0.8137801
8: -0.2398748, 1.0796266, -0.2659380, 1.0883889, -1.2964497, 1.3079104
9: -5.1227369, -4.0238676, -5.1315775, -4.0012856, -0.7940948, 0.7897890

Time for backsubstitution: 22.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 5760
type: B, layer: 1, pos: 5844

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 902

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5709827, upper bound: 0.5664851
time: 3.71 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5723154, upper bound: 0.5777375
time: 4.14 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -8.6137981, -7.0364189, -8.6106033, -7.0397215, -1.1541433, 1.1518881
1: 1.2892256, 2.5511189, 1.2983098, 2.5320768, -1.1523700, 1.1550088
2: -6.3156695, -5.0988727, -6.3137093, -5.1015697, -0.9759338, 0.9720192
3: -10.5776348, -8.9911652, -10.5566998, -9.0012140, -1.1126893, 1.1076770
4: -4.5583401, -3.2592986, -4.5509124, -3.2773652, -0.9410748, 0.9458482
5: -8.6907711, -7.4491258, -8.6856546, -7.4559536, -1.0073180, 1.0144410
6: -8.2924862, -6.7410164, -8.2648268, -6.7505202, -0.9894905, 1.0019035
7: -7.3888073, -6.2798543, -7.3810763, -6.3011889, -0.8263223, 0.8139833
8: -0.2659304, 1.0883863, -0.2398860, 1.0796309, -1.3130999, 1.2912598
9: -5.1315794, -4.0012932, -5.1227384, -4.0238605, -0.7978840, 0.7912477

Time for backsubstitution: 22.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 5760
type: B, layer: 1, pos: 5844

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 902

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5763994, upper bound: 0.5610682
time: 3.71 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5777375, upper bound: 0.5723154
time: 3.95 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8.6137981, -7.0364189, -8.6138010, -7.0364141, -1.1595407, 1.1564879
1: 1.2892256, 2.5511189, 1.2892179, 2.5511184, -1.1591458, 1.1638741
2: -6.3156695, -5.0988727, -6.3156705, -5.0988712, -0.9819465, 0.9773457
3: -10.5776348, -8.9911652, -10.5776348, -8.9911470, -1.1227951, 1.1122656
4: -4.5583401, -3.2592986, -4.5583429, -3.2592924, -0.9445982, 0.9507833
5: -8.6907711, -7.4491258, -8.6907854, -7.4491248, -1.0119677, 1.0180469
6: -8.2924862, -6.7410164, -8.2924938, -6.7410145, -0.9926233, 1.0045068
7: -7.3888073, -6.2798543, -7.3888092, -6.2798409, -0.8286755, 0.8173153
8: -0.2659304, 1.0883863, -0.2659380, 1.0883889, -1.3104372, 1.3052473
9: -5.1315794, -4.0012932, -5.1315775, -4.0012856, -0.7989287, 0.7908335

Time for backsubstitution: 22.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 5760
type: B, layer: 1, pos: 5844

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 1, pos: 902

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5763995, upper bound: 0.5610682
time: 4.08 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5777383, upper bound: 0.5723154
time: 4.08 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 31.11 seconds
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 31.11
Output dim: 1, lower bound: -0.5619040, upper bound: 0.5664850
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.11
Output dim: 1, lower bound: -0.5631037, upper bound: 0.5777375
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 31.11
Output dim: 1, lower bound: -0.5709827, upper bound: 0.5664851
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.11
Output dim: 1, lower bound: -0.5723154, upper bound: 0.5777375
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.11
Output dim: 1, lower bound: -0.5763994, upper bound: 0.5610682
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.11
Output dim: 1, lower bound: -0.5777375, upper bound: 0.5723154
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.11
Output dim: 1, lower bound: -0.5763995, upper bound: 0.5610682
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.11
Output dim: 1, lower bound: -0.5777383, upper bound: 0.5723154

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -8.6068020, -7.0415535, -8.6134682, -7.0370560, -1.1506419, 1.1502151
1: 1.3037395, 2.5260997, 1.2911100, 2.5510097, -1.1509714, 1.1518464
2: -6.3097477, -5.1045237, -6.3154359, -5.0998611, -0.9708700, 0.9764986
3: -10.5480347, -9.0131989, -10.5776291, -8.9952803, -1.1076803, 1.1078475
4: -4.5446239, -3.2826667, -4.5561857, -3.2593038, -0.9435825, 0.9397416
5: -8.6792812, -7.4603615, -8.6888390, -7.4491568, -1.0079503, 0.9962873
6: -8.2510481, -6.7614150, -8.2878351, -6.7410784, -0.9879208, 0.9864304
7: -7.3711133, -6.3153296, -7.3887520, -6.2845836, -0.8056769, 0.8162022
8: -0.2350419, 1.0751731, -0.2657628, 1.0869255, -1.2900882, 1.3087373
9: -5.1148067, -4.0314226, -5.1314549, -4.0038495, -0.7871597, 0.7901998

Time for backsubstitution: 22.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 5760
type: A, layer: 1, pos: 5844

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 916

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5525626, upper bound: 0.5767136
time: 3.73 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5630958, upper bound: 0.5777302
time: 4.09 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -8.6105881, -7.0397239, -8.6137533, -7.0364161, -1.1549306, 1.1490359
1: 1.2983398, 2.5320773, 1.2892990, 2.5511165, -1.1511202, 1.1603942
2: -6.3137088, -5.1015787, -6.3156672, -5.0988970, -0.9761639, 0.9741502
3: -10.5567017, -9.0012503, -10.5776348, -8.9912329, -1.1205440, 1.1051412
4: -4.5508966, -3.2773695, -4.5582871, -3.2592940, -0.9431932, 0.9471984
5: -8.6856079, -7.4559536, -8.6906691, -7.4491282, -1.0083413, 1.0025125
6: -8.2648077, -6.7505202, -8.2924442, -6.7410145, -0.9859710, 0.9944935
7: -7.3810768, -6.3012099, -7.3888063, -6.2798557, -0.8132873, 0.8137748
8: -0.2398775, 1.0796204, -0.2659361, 1.0883648, -1.2963848, 1.3081403
9: -5.1227355, -4.0238690, -5.1315770, -4.0012884, -0.7933266, 0.7897222

Time for backsubstitution: 22.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 5760
type: A, layer: 1, pos: 5844

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 1, pos: 916

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5616413, upper bound: 0.5767135
time: 3.74 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5723075, upper bound: 0.5777301
time: 4.13 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -8.6128778, -7.0364447, -8.6075287, -7.0446472, -1.1480858, 1.1493421
1: 1.2916017, 2.5505853, 1.3059492, 2.5251122, -1.1422029, 1.1468492
2: -6.3150563, -5.1008053, -6.3076601, -5.1074648, -0.9695837, 0.9623001
3: -10.5775709, -8.9913568, -10.5556278, -9.0018101, -1.1102579, 1.1051841
4: -4.5579963, -3.2593241, -4.5495234, -3.2786498, -0.9394360, 0.9446673
5: -8.6879349, -7.4492593, -8.6775951, -7.4632463, -0.9971337, 1.0064120
6: -8.2922869, -6.7411561, -8.2631512, -6.7509403, -0.9870987, 0.9985952
7: -7.3887405, -6.2822204, -7.3758526, -6.3088226, -0.8186619, 0.8051670
8: -0.2652953, 1.0869074, -0.2328770, 1.0754251, -1.3083048, 1.2829127
9: -5.1311083, -4.0015850, -5.1206026, -4.0264339, -0.7945740, 0.7889309

Time for backsubstitution: 22.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 5760
type: A, layer: 1, pos: 5844

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 1, pos: 916

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5655686, upper bound: 0.5598919
time: 4.02 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5763908, upper bound: 0.5610611
time: 3.93 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -8.6137857, -7.0364199, -8.6105576, -7.0397215, -1.1541305, 1.1498349
1: 1.2892489, 2.5511193, 1.2983885, 2.5320764, -1.1536427, 1.1528244
2: -6.3156681, -5.0988817, -6.3137069, -5.1015949, -0.9754777, 0.9748368
3: -10.5776348, -8.9911880, -10.5566998, -9.0012941, -1.1117988, 1.1069412
4: -4.5583248, -3.2592988, -4.5508599, -3.2773657, -0.9412069, 0.9454751
5: -8.6907377, -7.4491277, -8.6855421, -7.4559546, -1.0072942, 1.0035634
6: -8.2924747, -6.7410159, -8.2647800, -6.7505236, -0.9875188, 1.0008373
7: -7.3888073, -6.2798591, -7.3810768, -6.3012056, -0.8188128, 0.8094290
8: -0.2659278, 1.0883794, -0.2398844, 1.0796061, -1.3130379, 1.2914872
9: -5.1315784, -4.0012937, -5.1227341, -4.0238628, -0.7982883, 0.7907544

Time for backsubstitution: 22.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 5760
type: A, layer: 1, pos: 5844

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 1, pos: 916

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5670580, upper bound: 0.5712968
time: 3.74 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5777295, upper bound: 0.5723081
time: 4.13 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -8.6128778, -7.0364447, -8.6107759, -7.0413380, -1.1534832, 1.1539791
1: 1.2916017, 2.5505853, 1.2968535, 2.5441537, -1.1489782, 1.1557164
2: -6.3150563, -5.1008053, -6.3096180, -5.1047592, -0.9756057, 0.9676299
3: -10.5775709, -8.9913568, -10.5765638, -8.9917393, -1.1203477, 1.1097739
4: -4.5579963, -3.2593241, -4.5569520, -3.2605774, -0.9429564, 0.9495873
5: -8.6879349, -7.4492593, -8.6827259, -7.4564190, -1.0017819, 1.0100250
6: -8.2922869, -6.7411561, -8.2908106, -6.7414417, -0.9907961, 1.0011892
7: -7.3887405, -6.2822204, -7.3835807, -6.2874627, -0.8210299, 0.8097148
8: -0.2652953, 1.0869074, -0.2588911, 1.0841815, -1.3056426, 1.2968974
9: -5.1311083, -4.0015850, -5.1294570, -4.0038562, -0.7956197, 0.7885123

Time for backsubstitution: 22.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 5760
type: A, layer: 1, pos: 5844

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 1, pos: 916

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5655686, upper bound: 0.5598919
time: 3.97 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5763909, upper bound: 0.5610612
time: 3.91 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -8.6137857, -7.0364199, -8.6137533, -7.0364161, -1.1595278, 1.1544328
1: 1.2892489, 2.5511193, 1.2892990, 2.5511165, -1.1595564, 1.1616883
2: -6.3156681, -5.0988817, -6.3156672, -5.0988970, -0.9814897, 0.9801624
3: -10.5776348, -8.9911880, -10.5776348, -8.9912329, -1.1218920, 1.1115294
4: -4.5583248, -3.2592988, -4.5582871, -3.2592940, -0.9447293, 0.9507227
5: -8.6907377, -7.4491277, -8.6906691, -7.4491282, -1.0119448, 1.0071673
6: -8.2924747, -6.7410159, -8.2924442, -6.7410145, -0.9914532, 1.0034337
7: -7.3888073, -6.2798591, -7.3888063, -6.2798557, -0.8211663, 0.8173115
8: -0.2659278, 1.0883794, -0.2659361, 1.0883648, -1.3103733, 1.3054743
9: -5.1315784, -4.0012937, -5.1315770, -4.0012884, -0.7993331, 0.7907662

Time for backsubstitution: 22.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 5760
type: A, layer: 1, pos: 5844

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 1, pos: 916

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5670589, upper bound: 0.5712968
time: 3.93 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5777304, upper bound: 0.5723081
time: 4.07 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 31.15 seconds
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 31.15
Output dim: 1, lower bound: -0.5525626, upper bound: 0.5767136
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 31.15
Output dim: 1, lower bound: -0.5630958, upper bound: 0.5777302
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 31.15
Output dim: 1, lower bound: -0.5616413, upper bound: 0.5767135
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 31.15
Output dim: 1, lower bound: -0.5723075, upper bound: 0.5777301
NS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 31.15
Output dim: 1, lower bound: -0.5655686, upper bound: 0.5598919
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 31.15
Output dim: 1, lower bound: -0.5763908, upper bound: 0.5610611
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 31.15
Output dim: 1, lower bound: -0.5670580, upper bound: 0.5712968
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 31.15
Output dim: 1, lower bound: -0.5777295, upper bound: 0.5723081
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 31.15
Output dim: 1, lower bound: -0.5655686, upper bound: 0.5598919
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 31.15
Output dim: 1, lower bound: -0.5763909, upper bound: 0.5610612
NS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 31.15
Output dim: 1, lower bound: -0.5670589, upper bound: 0.5712968
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 31.15
Output dim: 1, lower bound: -0.5777304, upper bound: 0.5723081

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -8.6033831, -7.0447607, -8.6133461, -7.0381179, -1.1445961, 1.1465631
1: 1.3107524, 2.5210690, 1.2933316, 2.5506535, -1.1435900, 1.1446671
2: -6.3031178, -5.1122103, -6.3150063, -5.1024742, -0.9619300, 0.9684415
3: -10.5453749, -9.0160065, -10.5767756, -8.9954872, -1.1048851, 1.1037879
4: -4.5394626, -3.2861288, -4.5545187, -3.2593136, -0.9384587, 0.9345560
5: -8.6787357, -7.4618387, -8.6886940, -7.4493651, -1.0072074, 0.9945230
6: -8.2501020, -6.7616839, -8.2876701, -6.7411642, -0.9866743, 0.9854848
7: -7.3695269, -6.3164067, -7.3883233, -6.2847643, -0.8039896, 0.8147290
8: -0.2305627, 1.0697584, -0.2653120, 1.0852206, -1.2838097, 1.3029056
9: -5.1120610, -4.0344772, -5.1304684, -4.0040741, -0.7842405, 0.7862070

Time for backsubstitution: 22.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 5760
type: B, layer: 1, pos: 5844

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 1, pos: 832

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.5525626, upper bound: 0.5752550
time: 3.95 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5525626, upper bound: 0.5767136
time: 3.91 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -8.6068001, -7.0415573, -8.6134682, -7.0370569, -1.1496167, 1.1437063
1: 1.3037491, 2.5260949, 1.2911139, 2.5510082, -1.1480284, 1.1518412
2: -6.3097453, -5.1045370, -6.3154354, -5.0998654, -0.9708638, 0.9730151
3: -10.5480328, -9.0131989, -10.5776262, -8.9952812, -1.1062737, 1.1060743
4: -4.5446124, -3.2826691, -4.5561829, -3.2593048, -0.9384711, 0.9397383
5: -8.6792812, -7.4603605, -8.6888399, -7.4491568, -1.0079188, 0.9964733
6: -8.2509785, -6.7614164, -8.2877998, -6.7410779, -0.9882498, 0.9862523
7: -7.3711119, -6.3153315, -7.3887506, -6.2845840, -0.8055038, 0.8162706
8: -0.2350373, 1.0751641, -0.2657633, 1.0869226, -1.2900839, 1.3069134
9: -5.1147981, -4.0314231, -5.1314540, -4.0038505, -0.7840455, 0.7901967

Time for backsubstitution: 22.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 5760
type: B, layer: 1, pos: 5844

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 1, pos: 832

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5630963, upper bound: 0.5761825
time: 4.28 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5630958, upper bound: 0.5777302
time: 4.58 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -8.6071692, -7.0429311, -8.6136322, -7.0374784, -1.1488824, 1.1453855
1: 1.3053541, 2.5270391, 1.2915201, 2.5507603, -1.1437407, 1.1525645
2: -6.3070693, -5.1092653, -6.3152347, -5.1015091, -0.9672096, 0.9660916
3: -10.5540400, -9.0040607, -10.5767841, -8.9914398, -1.1177499, 1.1010814
4: -4.5457368, -3.2808325, -4.5566216, -3.2593036, -0.9380693, 0.9420137
5: -8.6850605, -7.4574323, -8.6905231, -7.4493375, -1.0076003, 1.0007472
6: -8.2638664, -6.7507887, -8.2922783, -6.7411008, -0.9847264, 0.9935479
7: -7.3794894, -6.3022833, -7.3883781, -6.2800350, -0.8115990, 0.8123033
8: -0.2353816, 1.0742066, -0.2654815, 1.0866616, -1.2900977, 1.3023062
9: -5.1199937, -4.0269203, -5.1305904, -4.0015116, -0.7904081, 0.7857325

Time for backsubstitution: 22.68 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 58.84 + 555.37 = 614.22 seconds
