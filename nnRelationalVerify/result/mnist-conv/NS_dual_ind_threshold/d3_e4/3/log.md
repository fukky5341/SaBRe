## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.872541919


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (6.6375160, 9.0313435, 6.6375160, 9.0313435, -1.8442516, 1.8442519)
1: (-17.4851456, -13.7643309, -17.4851456, -13.7643309, -2.7918735, 2.7918735)
2: (-3.2759049, -0.5142205, -3.2759049, -0.5142205, -2.3578615, 2.3578615)
3: (-10.8677959, -7.9381061, -10.8677959, -7.9381061, -2.6013856, 2.6013861)
4: (-12.5387917, -9.0154037, -12.5387917, -9.0154037, -2.7610822, 2.7610822)
5: (-4.9653873, -2.6635807, -4.9653873, -2.6635807, -2.0475702, 2.0475702)
6: (-3.0826335, -0.5545902, -3.0826335, -0.5545902, -2.2829180, 2.2829187)
7: (-9.3434553, -5.3956985, -9.3434553, -5.3956985, -3.1810565, 3.1810570)
8: (-2.6018829, -0.3418674, -2.6018829, -0.3418674, -2.0486369, 2.0486372)
9: (-4.4801102, -1.7481186, -4.4801102, -1.7481186, -2.3078995, 2.3078992)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.07 + 38.63 = 61.71 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.8742898, upper bound: 0.8742899

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 471

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8742861, upper bound: 0.8698156
time: 5.30 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8742861, upper bound: 0.8742862
time: 8.34 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 13.72 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 13.72
Output dim: 0, lower bound: -0.8742861, upper bound: 0.8698156
NS_A2, status: Status.UNKNOWN, split count: 1, time: 13.72
Output dim: 0, lower bound: -0.8742861, upper bound: 0.8742862

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 6.6551008, 9.0135641, 6.6483846, 9.0208187, -1.7911239, 1.7899327
1: -17.4738808, -13.8005123, -17.4782677, -13.7856493, -2.6310873, 2.6196427
2: -3.2627926, -0.5239112, -3.2672889, -0.5199797, -2.3435087, 2.3436897
3: -10.8545094, -7.9736710, -10.8565397, -7.9592099, -2.4428215, 2.4299312
4: -12.5303125, -9.0268307, -12.5336456, -9.0226107, -2.7368712, 2.7304783
5: -4.9079671, -2.6729443, -4.9315729, -2.6704021, -1.8402214, 1.8614385
6: -3.0038815, -0.5669255, -3.0362353, -0.5638537, -2.0689974, 2.1008341
7: -9.3258781, -5.4602880, -9.3304539, -5.4337683, -2.9822493, 2.9613752
8: -2.5947638, -0.3486171, -2.5973587, -0.3460078, -2.0295734, 2.0295675
9: -4.4674053, -1.7695547, -4.4723735, -1.7615669, -2.2728467, 2.2686410

Time for backsubstitution: 20.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 471

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8698157, upper bound: 0.8698163
time: 5.75 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8698157, upper bound: 0.8698164
time: 12.85 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 6.5904884, 9.0354805, 6.6375179, 9.0313339, -1.8851533, 1.8454711
1: -17.5395660, -13.7612858, -17.4851456, -13.7643471, -2.8484921, 2.7954130
2: -3.2866726, -0.4902341, -3.2758987, -0.5142205, -2.3659143, 2.3812420
3: -10.9079895, -7.9272223, -10.8677921, -7.9381189, -2.6484585, 2.6109838
4: -12.5829248, -8.9888020, -12.5387926, -9.0154057, -2.8057680, 2.7976289
5: -4.9724522, -2.5807762, -4.9653707, -2.6635835, -2.0558090, 2.0907953
6: -3.0983093, -0.4341698, -3.0826259, -0.5545940, -2.2864566, 2.3228679
7: -9.4304600, -5.3878288, -9.3434505, -5.3957224, -3.2355380, 3.1883569
8: -2.6340671, -0.3340940, -2.6018806, -0.3418708, -2.0750732, 2.0553889
9: -4.5135827, -1.7374012, -4.4801068, -1.7481221, -2.3314004, 2.3200212

Time for backsubstitution: 20.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 471

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8698189, upper bound: 0.8742866
time: 6.22 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8698157, upper bound: 0.8742864
time: 5.69 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 32.88 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 32.88
Output dim: 0, lower bound: -0.8698157, upper bound: 0.8698163
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 32.88
Output dim: 0, lower bound: -0.8698157, upper bound: 0.8698164
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 32.88
Output dim: 0, lower bound: -0.8698189, upper bound: 0.8742866
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 32.88
Output dim: 0, lower bound: -0.8698157, upper bound: 0.8742864

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 6.5945582, 9.0354357, 6.6551008, 9.0135641, -1.8634107, 1.8060632
1: -17.5384178, -13.7615089, -17.4738808, -13.8005123, -2.8112431, 2.6560750
2: -3.2863765, -0.4930609, -3.2627926, -0.5239112, -2.3559713, 2.3710480
3: -10.9072332, -7.9287024, -10.8545094, -7.9736710, -2.6075902, 2.4747343
4: -12.5824928, -8.9957685, -12.5303125, -9.0268307, -2.7818503, 2.7572050
5: -4.9721651, -2.5824437, -4.9079671, -2.6729443, -1.8909018, 2.0331187
6: -3.0973392, -0.4360590, -3.0038815, -0.5669255, -2.1323237, 2.2416043
7: -9.4290791, -5.3902326, -9.3258781, -5.4602880, -3.1710272, 3.0285296
8: -2.6290088, -0.3342514, -2.5947638, -0.3486171, -2.0629892, 2.0429227
9: -4.5133839, -1.7397040, -4.4674053, -1.7695547, -2.3064563, 2.2943473

Time for backsubstitution: 22.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 5859

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8678158, upper bound: 0.8742837
time: 11.46 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8698133, upper bound: 0.8742838
time: 5.90 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 6.5904884, 9.0354805, 6.5904884, 9.0354805, -1.8893695, 1.8864166
1: -17.5395660, -13.7612858, -17.5395660, -13.7612858, -2.8487978, 2.8487988
2: -3.2866726, -0.4902341, -3.2866726, -0.4902341, -2.3823905, 2.3823903
3: -10.9079895, -7.9272223, -10.9079895, -7.9272223, -2.6587729, 2.6587729
4: -12.5829248, -8.9888020, -12.5829248, -8.9888020, -2.8182039, 2.8182049
5: -4.9724522, -2.5807762, -4.9724522, -2.5807762, -2.0993729, 2.1021502
6: -3.0983093, -0.4341698, -3.0983093, -0.4341698, -2.3265631, 2.3384748
7: -9.4304600, -5.3878288, -9.4304600, -5.3878288, -3.2493157, 3.2428913
8: -2.6340671, -0.3340940, -2.6340671, -0.3340940, -2.0823336, 2.0821416
9: -4.5135827, -1.7374012, -4.5135827, -1.7374012, -2.3309526, 2.3309526

Time for backsubstitution: 23.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 5859

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8678158, upper bound: 0.8742840
time: 24.68 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8698133, upper bound: 0.8742842
time: 7.76 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 55.83 seconds
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 55.83
Output dim: 0, lower bound: -0.8678158, upper bound: 0.8742837
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 55.83
Output dim: 0, lower bound: -0.8698133, upper bound: 0.8742838
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 55.83
Output dim: 0, lower bound: -0.8678158, upper bound: 0.8742840
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 55.83
Output dim: 0, lower bound: -0.8698133, upper bound: 0.8742842

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: 6.6008067, 9.0345602, 6.6581059, 9.0131464, -1.8564358, 1.8019688
1: -17.5293732, -13.7631664, -17.4695187, -13.8013029, -2.8012877, 2.6500416
2: -3.2809944, -0.4964566, -3.2601876, -0.5255138, -2.3490739, 2.3652978
3: -10.9038839, -7.9311028, -10.8528919, -7.9748163, -2.6028585, 2.4708734
4: -12.5798378, -9.0216751, -12.5290480, -9.0393133, -2.7665958, 2.7299190
5: -4.9655228, -2.5834088, -4.9047637, -2.6734004, -1.8827598, 2.0280497
6: -3.0877478, -0.4375148, -2.9992504, -0.5676160, -2.1219068, 2.2349851
7: -9.4262466, -5.4261031, -9.3245449, -5.4775720, -3.1479912, 2.9912097
8: -2.6183681, -0.3359675, -2.5896344, -0.3494496, -2.0512319, 2.0357399
9: -4.5112367, -1.7585897, -4.4663811, -1.7786603, -2.2937775, 2.2744832

Time for backsubstitution: 22.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8667401, upper bound: 0.8742847
time: 9.68 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8678144, upper bound: 0.8742823
time: 7.97 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: 6.5919046, 9.0498762, 6.6551075, 9.0135632, -1.8637607, 1.8165379
1: -17.5407658, -13.7462196, -17.4738712, -13.8005123, -2.8108015, 2.6715870
2: -3.2896268, -0.4825482, -3.2627914, -0.5239139, -2.3590322, 2.3767900
3: -10.9094734, -7.9161005, -10.8545046, -7.9736710, -2.6087403, 2.4901862
4: -12.6521320, -8.9946280, -12.5303116, -9.0268574, -2.8057396, 2.7455206
5: -4.9742489, -2.5712771, -4.9079618, -2.6729443, -1.8911214, 2.0352232
6: -3.0986762, -0.4161015, -3.0038753, -0.5669270, -2.1312213, 2.2461996
7: -9.5273190, -5.3893204, -9.3258781, -5.4603300, -3.1843920, 3.0143974
8: -2.6306009, -0.3083458, -2.5947585, -0.3486185, -2.0593390, 2.0490568
9: -4.5604944, -1.7375288, -4.4674044, -1.7695752, -2.3133609, 2.2886457

Time for backsubstitution: 23.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8687371, upper bound: 0.8742824
time: 18.60 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8698118, upper bound: 0.8742823
time: 5.96 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: 6.5967355, 9.0346031, 6.5935001, 9.0350599, -1.8823929, 1.8819022
1: -17.5305214, -13.7629423, -17.5352077, -13.7620773, -2.8388414, 2.8427649
2: -3.2812908, -0.4936302, -3.2840781, -0.4918513, -2.3754368, 2.3766139
3: -10.9046402, -7.9296236, -10.9063721, -7.9283752, -2.6540427, 2.6549625
4: -12.5802689, -9.0147047, -12.5816526, -9.0012817, -2.8029437, 2.7908802
5: -4.9658089, -2.5817404, -4.9692502, -2.5812392, -2.0913734, 2.0970738
6: -3.0887156, -0.4356256, -3.0936873, -0.4348674, -2.3163226, 2.3318570
7: -9.4276257, -5.4237032, -9.4291019, -5.4051108, -3.2262826, 3.2055612
8: -2.6234274, -0.3358102, -2.6289420, -0.3349171, -2.0705810, 2.0741284
9: -4.5114355, -1.7562871, -4.5125546, -1.7465063, -2.3198156, 2.3111010

Time for backsubstitution: 22.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8667401, upper bound: 0.8742823
time: 6.94 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8678144, upper bound: 0.8742832
time: 12.37 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: 6.5878334, 9.0499210, 6.5904946, 9.0354795, -1.8896990, 1.8894377
1: -17.5419102, -13.7459974, -17.5395527, -13.7612877, -2.8483562, 2.8643088
2: -3.2899225, -0.4797206, -3.2866695, -0.4902364, -2.3853555, 2.3937664
3: -10.9102325, -7.9146242, -10.9079866, -7.9272242, -2.6599212, 2.6745415
4: -12.6525650, -8.9876604, -12.5829229, -8.9888248, -2.8591559, 2.8064966
5: -4.9745359, -2.5696206, -4.9724460, -2.5807781, -2.0994844, 2.1042531
6: -3.0996425, -0.4142132, -3.0983031, -0.4341702, -2.3255398, 2.3430715
7: -9.5286961, -5.3869090, -9.4304562, -5.3878698, -3.2626791, 3.2287939
8: -2.6356587, -0.3081903, -2.6340623, -0.3340969, -2.0786834, 2.0850000
9: -4.5606909, -1.7352256, -4.5135803, -1.7374227, -2.3572254, 2.3252060

Time for backsubstitution: 23.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8687371, upper bound: 0.8742820
time: 4.81 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8698118, upper bound: 0.8742834
time: 16.98 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 45.39 seconds
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 45.39
Output dim: 0, lower bound: -0.8667401, upper bound: 0.8742847
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 45.39
Output dim: 0, lower bound: -0.8678144, upper bound: 0.8742823
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 45.39
Output dim: 0, lower bound: -0.8687371, upper bound: 0.8742824
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 45.39
Output dim: 0, lower bound: -0.8698118, upper bound: 0.8742823
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 45.39
Output dim: 0, lower bound: -0.8667401, upper bound: 0.8742823
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 45.39
Output dim: 0, lower bound: -0.8678144, upper bound: 0.8742832
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 45.39
Output dim: 0, lower bound: -0.8687371, upper bound: 0.8742820
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 45.39
Output dim: 0, lower bound: -0.8698118, upper bound: 0.8742834

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 6.6054149, 9.0301466, 6.6659698, 9.0047913, -1.8430800, 1.7906814
1: -17.5264816, -13.7699070, -17.4642944, -13.8141804, -2.7819738, 2.6385341
2: -3.2794788, -0.5012259, -3.2581162, -0.5339572, -2.3324823, 2.3513060
3: -10.8995094, -7.9325972, -10.8473892, -7.9772077, -2.5885863, 2.4596195
4: -12.5779018, -9.0254173, -12.5256090, -9.0449924, -2.7590823, 2.7227299
5: -4.9586725, -2.5856571, -4.8917742, -2.6765094, -1.8673267, 2.0121398
6: -3.0826058, -0.4406734, -2.9896941, -0.5719085, -2.1068511, 2.2209115
7: -9.4219875, -5.4347301, -9.3184414, -5.4939013, -3.1227155, 2.9740958
8: -2.6158237, -0.3411379, -2.5859175, -0.3592782, -2.0377955, 2.0268734
9: -4.5088515, -1.7640595, -4.4620333, -1.7879100, -2.2793491, 2.2624185

Time for backsubstitution: 22.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 859

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8637479, upper bound: 0.8731452
time: 5.80 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8656723, upper bound: 0.8731456
time: 5.77 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 6.6001120, 9.0345650, 6.6358757, 9.0180902, -1.8609052, 1.8230853
1: -17.5298653, -13.7631416, -17.5136719, -13.7994518, -2.8013668, 2.6952143
2: -3.2811105, -0.4960330, -3.2811527, -0.5143384, -2.3618627, 2.3754210
3: -10.9041939, -7.9308748, -10.8593159, -7.9577303, -2.6150336, 2.4779413
4: -12.5799885, -9.0206566, -12.5573931, -9.0336246, -2.7710152, 2.7599692
5: -4.9655600, -2.5826604, -4.9155078, -2.6459897, -1.8871176, 2.0369821
6: -3.0879087, -0.4366598, -3.0143094, -0.5453429, -2.1247110, 2.2497382
7: -9.4268398, -5.4258752, -9.3557730, -5.4736099, -3.1506500, 3.0028777
8: -2.6191392, -0.3359256, -2.6104527, -0.3391390, -2.0588932, 2.0465934
9: -4.5113158, -1.7582479, -4.4893179, -1.7723820, -2.2997687, 2.2970078

Time for backsubstitution: 23.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 859

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8647878, upper bound: 0.8731485
time: 7.73 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8667124, upper bound: 0.8731474
time: 5.67 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 6.5965037, 9.0454664, 6.6629729, 9.0052099, -1.8504102, 1.8049228
1: -17.5378723, -13.7529545, -17.4686413, -13.8133907, -2.7914848, 2.6600771
2: -3.2881038, -0.4873294, -3.2607164, -0.5323594, -2.3425102, 2.3622861
3: -10.9051008, -7.9175901, -10.8490028, -7.9760628, -2.5944681, 2.4789305
4: -12.6501961, -8.9983692, -12.5268717, -9.0325336, -2.7979538, 2.7383351
5: -4.9673972, -2.5735512, -4.8949714, -2.6760569, -1.8756876, 2.0193081
6: -3.0935349, -0.4192629, -2.9943209, -0.5712214, -2.1161628, 2.2321320
7: -9.5230350, -5.3979454, -9.3197746, -5.4766593, -3.1591196, 2.9972820
8: -2.6280556, -0.3135147, -2.5910397, -0.3584437, -2.0459018, 2.0393748
9: -4.5581007, -1.7430022, -4.4630542, -1.7788318, -2.2989516, 2.2765915

Time for backsubstitution: 23.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 859

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8657438, upper bound: 0.8731462
time: 10.79 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8676709, upper bound: 0.8731463
time: 14.46 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 6.5912094, 9.0498829, 6.6328702, 9.0185108, -1.8682346, 1.8306093
1: -17.5412598, -13.7461958, -17.5180168, -13.7986603, -2.8108816, 2.7166727
2: -3.2897446, -0.4821241, -3.2837653, -0.5127344, -2.3717959, 2.3864105
3: -10.9097843, -7.9158735, -10.8609285, -7.9565721, -2.6209221, 2.4972551
4: -12.6522846, -8.9936104, -12.5586557, -9.0211697, -2.8100920, 2.7755666
5: -4.9742856, -2.5705295, -4.9187055, -2.6455312, -1.8954797, 2.0441694
6: -3.0988357, -0.4152479, -3.0189321, -0.5446553, -2.1340261, 2.2609568
7: -9.5279102, -5.3890820, -9.3571043, -5.4563665, -3.1870489, 3.0260658
8: -2.6313739, -0.3083043, -2.6155758, -0.3383141, -2.0669975, 2.0574129
9: -4.5605717, -1.7371855, -4.4903393, -1.7632978, -2.3193552, 2.3111944

Time for backsubstitution: 23.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 859

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8667885, upper bound: 0.8731458
time: 5.51 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8687124, upper bound: 0.8731471
time: 10.62 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 6.6008453, 9.0302010, 6.6014581, 9.0266199, -1.8694091, 1.8692235
1: -17.5277691, -13.7696524, -17.5298786, -13.7749519, -2.8196683, 2.8275204
2: -3.2799568, -0.4980822, -3.2814848, -0.5002944, -2.3587189, 2.3630462
3: -10.9008236, -7.9309530, -10.8990755, -7.9309464, -2.6398969, 2.6395621
4: -12.5784092, -9.0177078, -12.5780640, -9.0070419, -2.7956600, 2.7843914
5: -4.9590063, -2.5836415, -4.9562006, -2.5849271, -2.0792518, 2.0817108
6: -3.0836873, -0.4383349, -3.0840564, -0.4401159, -2.3041317, 2.3182125
7: -9.4238663, -5.4322662, -9.4216881, -5.4215488, -3.2015433, 3.1848435
8: -2.6214485, -0.3409424, -2.6250901, -0.3447638, -2.0574007, 2.0643015
9: -4.5091105, -1.7613885, -4.5080242, -1.7562938, -2.3048725, 2.2995653

Time for backsubstitution: 24.41 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 61.71 + 542.81 = 604.51 seconds
