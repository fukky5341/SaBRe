## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.3552878565


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.3612347, -6.1413960, -7.3612347, -6.1413960, -0.7229214, 0.7229214)
1: (-6.7012358, -5.3436651, -6.7012358, -5.3436651, -1.1029615, 1.1029606)
2: (-4.7904668, -3.6836765, -4.7904668, -3.6836765, -0.8028684, 0.8028684)
3: (-5.1534758, -3.8338423, -5.1534758, -3.8338423, -0.7839131, 0.7839131)
4: (-10.7349405, -9.5796747, -10.7349405, -9.5796747, -0.6515818, 0.6515818)
5: (1.3470602, 2.2432051, 1.3470602, 2.2432051, -0.6103714, 0.6103711)
6: (0.1716712, 1.3331558, 0.1716712, 1.3331558, -0.6167908, 0.6167908)
7: (-12.6086416, -11.2634993, -12.6086416, -11.2634993, -0.8833489, 0.8833489)
8: (6.0859089, 7.0352283, 6.0859089, 7.0352283, -0.8516474, 0.8516474)
9: (-8.6910915, -7.4509592, -8.6910915, -7.4509592, -1.0465469, 1.0465469)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.62 + 36.34 = 58.95 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.3556433, upper bound: 0.3556435

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 6212
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4614

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 542

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3556411, upper bound: 0.3542100
time: 5.42 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3556411, upper bound: 0.3556404
time: 4.59 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 10.23 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 10.23
Output dim: 8, lower bound: -0.3556411, upper bound: 0.3542100
NS_A2, status: Status.UNKNOWN, split count: 1, time: 10.23
Output dim: 8, lower bound: -0.3556411, upper bound: 0.3556404

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -7.3501544, -6.1493382, -7.3549566, -6.1415854, -0.7112823, 0.7086926
1: -6.6940956, -5.3489256, -6.6975384, -5.3439026, -1.0955210, 1.0941868
2: -4.7834120, -3.6887298, -4.7866006, -3.6839380, -0.7955704, 0.7931728
3: -5.1407027, -3.8481073, -5.1527863, -3.8422637, -0.7630167, 0.7689466
4: -10.7148981, -9.5955238, -10.7228374, -9.5800114, -0.6310413, 0.6235394
5: 1.3560526, 2.2318153, 1.3473964, 2.2364359, -0.5947950, 0.5987415
6: 0.1795981, 1.3211713, 0.1718802, 1.3269817, -0.6024127, 0.6043971
7: -12.5972309, -11.2718582, -12.6022482, -11.2638721, -0.8713889, 0.8685408
8: 6.0933132, 7.0272436, 6.0863895, 7.0304656, -0.8396015, 0.8432007
9: -8.6850538, -7.4543557, -8.6883678, -7.4512520, -1.0396404, 1.0403242

Time for backsubstitution: 21.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 6212
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 4614

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6182

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3555321, upper bound: 0.3534478
time: 4.64 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3556404, upper bound: 0.3542083
time: 4.94 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -7.3612232, -6.1413946, -7.3612266, -6.1413946, -0.7085247, 0.7219603
1: -6.7012305, -5.3436656, -6.7012329, -5.3436646, -1.0969591, 1.1029577
2: -4.7904620, -3.6836762, -4.7904639, -3.6836760, -0.7945137, 0.8025360
3: -5.1534753, -3.8338542, -5.1534748, -3.8338487, -0.7802844, 0.7634988
4: -10.7349195, -9.5796728, -10.7349281, -9.5796738, -0.6219268, 0.6462188
5: 1.3470609, 2.2431931, 1.3470592, 2.2431979, -0.6070225, 0.5946522
6: 0.1716713, 1.3331468, 0.1716726, 1.3331505, -0.6167150, 0.6018190
7: -12.6086330, -11.2635012, -12.6086388, -11.2634993, -0.8688607, 0.8833447
8: 6.0859103, 7.0352201, 6.0859084, 7.0352235, -0.8516431, 0.8427205
9: -8.6910868, -7.4509592, -8.6910877, -7.4509573, -1.0414033, 1.0465436

Time for backsubstitution: 21.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 6212
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 4614

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6182

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3555321, upper bound: 0.3548795
time: 5.41 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3556404, upper bound: 0.3556394
time: 6.21 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 33.27 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 33.27
Output dim: 8, lower bound: -0.3555321, upper bound: 0.3534478
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 33.27
Output dim: 8, lower bound: -0.3556404, upper bound: 0.3542083
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 33.27
Output dim: 8, lower bound: -0.3555321, upper bound: 0.3548795
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 33.27
Output dim: 8, lower bound: -0.3556404, upper bound: 0.3556394

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -7.3493423, -6.1511812, -7.3535380, -6.1446648, -0.7067151, 0.7046857
1: -6.6898656, -5.3498559, -6.6905231, -5.3454881, -1.0849228, 1.0818739
2: -4.7816758, -3.6894832, -4.7836914, -3.6852388, -0.7914758, 0.7884226
3: -5.1345243, -3.8491871, -5.1424108, -3.8441501, -0.7545786, 0.7573900
4: -10.7139845, -9.6027365, -10.7211990, -9.5921144, -0.6183779, 0.6153398
5: 1.3581460, 2.2309368, 1.3509250, 2.2349026, -0.5902557, 0.5937166
6: 0.1798124, 1.3192189, 0.1722441, 1.3237244, -0.5964990, 0.5996594
7: -12.5952024, -11.2732859, -12.5987282, -11.2662735, -0.8654518, 0.8619885
8: 6.0963202, 7.0263247, 6.0914569, 7.0290747, -0.8347816, 0.8369660
9: -8.6839428, -7.4605236, -8.6864719, -7.4615946, -1.0273056, 1.0314708

Time for backsubstitution: 21.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 6212
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4614

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 522

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3542004, upper bound: 0.3534444
time: 4.67 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3555274, upper bound: 0.3534439
time: 6.94 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -7.3501511, -6.1493406, -7.3576875, -6.1401377, -0.7128811, 0.7111936
1: -6.6940894, -5.3489232, -6.6996341, -5.3419399, -1.0939388, 1.0951996
2: -4.7834072, -3.6887298, -4.7874818, -3.6808641, -0.7987452, 0.7936902
3: -5.1406898, -3.8481088, -5.1542778, -3.8337674, -0.7685592, 0.7679734
4: -10.7149000, -9.5955324, -10.7333441, -9.5792236, -0.6294484, 0.6320610
5: 1.3560562, 2.2318139, 1.3461552, 2.2410526, -0.5990949, 0.5994773
6: 0.1795980, 1.3211684, 0.1703508, 1.3277285, -0.6028261, 0.6043472
7: -12.5972271, -11.2718601, -12.6040649, -11.2631931, -0.8712382, 0.8709378
8: 6.0933189, 7.0272427, 6.0852690, 7.0350924, -0.8449268, 0.8432918
9: -8.6850529, -7.4543667, -8.6985798, -7.4503856, -1.0382385, 1.0512033

Time for backsubstitution: 21.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 6212
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4614

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 522

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3543098, upper bound: 0.3542049
time: 7.05 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3556356, upper bound: 0.3542043
time: 4.74 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -7.3604021, -6.1432390, -7.3598089, -6.1444750, -0.7039685, 0.7176535
1: -6.6970091, -5.3445992, -6.6942239, -5.3452559, -1.0863752, 1.0906420
2: -4.7887316, -3.6844318, -4.7875600, -3.6849763, -0.7904072, 0.7977910
3: -5.1472921, -3.8349290, -5.1430969, -3.8357303, -0.7707438, 0.7519474
4: -10.7340012, -9.5868893, -10.7332888, -9.5917797, -0.6092596, 0.6367378
5: 1.3491545, 2.2423182, 1.3505888, 2.2416656, -0.6021051, 0.5896313
6: 0.1718837, 1.3311993, 0.1720349, 1.3299038, -0.6107893, 0.5970404
7: -12.6066160, -11.2649279, -12.6051197, -11.2659006, -0.8629370, 0.8767972
8: 6.0889201, 7.0342999, 6.0909786, 7.0338335, -0.8468180, 0.8364844
9: -8.6899891, -7.4571290, -8.6891975, -7.4613008, -1.0290804, 1.0376983

Time for backsubstitution: 21.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 6212
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4614

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 522

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3542004, upper bound: 0.3548756
time: 8.06 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3555274, upper bound: 0.3548756
time: 5.69 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -7.3612223, -6.1413999, -7.3639579, -6.1399517, -0.7101212, 0.7228436
1: -6.7012262, -5.3436680, -6.7033272, -5.3417034, -1.0953817, 1.1039743
2: -4.7904568, -3.6836762, -4.7913456, -3.6806014, -0.7976904, 0.8030572
3: -5.1534610, -3.8338540, -5.1549625, -3.8253465, -0.7820275, 0.7625251
4: -10.7349215, -9.5796852, -10.7454376, -9.5788879, -0.6203346, 0.6492436
5: 1.3470659, 2.2431927, 1.3458223, 2.2478170, -0.6090181, 0.5953870
6: 0.1716716, 1.3331451, 0.1701428, 1.3339005, -0.6161017, 0.6017675
7: -12.6086302, -11.2635021, -12.6104546, -11.2628212, -0.8687086, 0.8845830
8: 6.0859141, 7.0352182, 6.0847769, 7.0398502, -0.8555989, 0.8428087
9: -8.6910839, -7.4509711, -8.7013035, -7.4500947, -1.0399976, 1.0574274

Time for backsubstitution: 21.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 6212
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4614

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 522

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3543097, upper bound: 0.3556362
time: 6.01 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3556356, upper bound: 0.3556361
time: 4.94 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 32.97 seconds
NS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 32.97
Output dim: 8, lower bound: -0.3542004, upper bound: 0.3534444
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 32.97
Output dim: 8, lower bound: -0.3555274, upper bound: 0.3534439
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 32.97
Output dim: 8, lower bound: -0.3543098, upper bound: 0.3542049
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 32.97
Output dim: 8, lower bound: -0.3556356, upper bound: 0.3542043
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 32.97
Output dim: 8, lower bound: -0.3542004, upper bound: 0.3548756
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 32.97
Output dim: 8, lower bound: -0.3555274, upper bound: 0.3548756
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 32.97
Output dim: 8, lower bound: -0.3543097, upper bound: 0.3556362
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 32.97
Output dim: 8, lower bound: -0.3556356, upper bound: 0.3556361

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7.3558397, -6.1443982, -7.3535366, -6.1446629, -0.7130327, 0.7152054
1: -6.7006707, -5.3426580, -6.6905193, -5.3454885, -1.0956402, 1.0961394
2: -4.8157816, -3.6831784, -4.7836914, -3.6852436, -0.8031063, 0.7969165
3: -5.1423430, -3.8021536, -5.1424031, -3.8441491, -0.7629027, 0.7661295
4: -10.7204027, -9.5950089, -10.7212000, -9.5921154, -0.6271555, 0.6232362
5: 1.3439088, 2.2328403, 1.3509259, 2.2349000, -0.5963681, 0.5956836
6: 0.1471508, 1.3232573, 0.1722441, 1.3237178, -0.6037331, 0.6085835
7: -12.5995140, -11.2625666, -12.5987244, -11.2662754, -0.8703213, 0.8715544
8: 6.0893421, 7.0306358, 6.0914602, 7.0290756, -0.8475966, 0.8400617
9: -8.7121277, -7.4568186, -8.6864719, -7.4615974, -1.0544376, 1.0346985

Time for backsubstitution: 21.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 6212
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 4614
type: B, layer: 1, pos: 536

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 542

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3540961, upper bound: 0.3534437
time: 10.50 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3540960, upper bound: 0.3534444
time: 5.28 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -7.3566713, -6.1425662, -7.3576870, -6.1401391, -0.7191982, 0.7203953
1: -6.7048860, -5.3417268, -6.6996312, -5.3419390, -1.1045265, 1.1080461
2: -4.8175383, -3.6824119, -4.7874808, -3.6808686, -0.8096623, 0.8022137
3: -5.1485052, -3.8010693, -5.1542692, -3.8337655, -0.7768285, 0.7767229
4: -10.7213316, -9.5878067, -10.7333422, -9.5792227, -0.6383224, 0.6363821
5: 1.3418226, 2.2336946, 1.3461573, 2.2410495, -0.6032851, 0.6014237
6: 0.1469394, 1.3252152, 0.1703522, 1.3277236, -0.6090043, 0.6132925
7: -12.6015034, -11.2611446, -12.6040621, -11.2631931, -0.8760657, 0.8793168
8: 6.0863371, 7.0315514, 6.0852714, 7.0350904, -0.8548903, 0.8463869
9: -8.7132530, -7.4506602, -8.6985798, -7.4503899, -1.0653939, 1.0544381

Time for backsubstitution: 21.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 6212
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 4614

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 542

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3542040, upper bound: 0.3542049
time: 5.13 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3542040, upper bound: 0.3542049
time: 7.01 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7.3669157, -6.1364861, -7.3598094, -6.1444750, -0.7102809, 0.7237313
1: -6.7077279, -5.3373966, -6.6942220, -5.3452549, -1.0969906, 1.1018915
2: -4.8227825, -3.6781387, -4.7875595, -3.6849823, -0.8015423, 0.8062916
3: -5.1551371, -3.7879131, -5.1430883, -3.8357301, -0.7790618, 0.7605925
4: -10.7404394, -9.5791416, -10.7332878, -9.5917797, -0.6180537, 0.6410599
5: 1.3349185, 2.2442017, 1.3505898, 2.2416630, -0.6060002, 0.5915802
6: 0.1392248, 1.3351846, 0.1720340, 1.3298948, -0.6121631, 0.6060865
7: -12.6108770, -11.2542143, -12.6051178, -11.2659025, -0.8677459, 0.8806558
8: 6.0818996, 7.0386090, 6.0909839, 7.0338340, -0.8556237, 0.8395753
9: -8.7182035, -7.4533873, -8.6891966, -7.4613037, -1.0561028, 1.0409598

Time for backsubstitution: 21.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 6212
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 4614
type: B, layer: 1, pos: 536

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 542

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3540963, upper bound: 0.3548758
time: 11.01 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3540963, upper bound: 0.3548764
time: 5.08 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -7.3609896, -6.1416144, -7.3639579, -6.1399517, -0.7097092, 0.7221837
1: -6.6994758, -5.3438864, -6.7033272, -5.3417034, -1.0925531, 1.1027279
2: -4.7901154, -3.6868284, -4.7913456, -3.6806014, -0.7974243, 0.7998986
3: -5.1475668, -3.8339865, -5.1549625, -3.8253465, -0.7760241, 0.7623911
4: -10.7343760, -9.5802269, -10.7454376, -9.5788879, -0.6197977, 0.6486900
5: 1.3475811, 2.2419474, 1.3458223, 2.2478170, -0.6085269, 0.5941365
6: 0.1718673, 1.3298237, 0.1701428, 1.3339005, -0.6158104, 0.5981753
7: -12.6076965, -11.2637405, -12.6104546, -11.2628212, -0.8677158, 0.8843279
8: 6.0876522, 7.0350466, 6.0847769, 7.0398502, -0.8531365, 0.8415618
9: -8.6908684, -7.4531307, -8.7013035, -7.4500947, -1.0399256, 1.0550933

Time for backsubstitution: 21.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 6212
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 4614

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 542

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3528804, upper bound: 0.3556355
time: 4.47 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3528804, upper bound: 0.3556363
time: 5.25 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7.3677368, -6.1346521, -7.3639565, -6.1399527, -0.7164345, 0.7289143
1: -6.7119355, -5.3364668, -6.7033229, -5.3417025, -1.1058693, 1.1137919
2: -4.8245330, -3.6773720, -4.7913451, -3.6806064, -0.8081007, 0.8115883
3: -5.1613064, -3.7868340, -5.1549568, -3.8253479, -0.7903502, 0.7711709
4: -10.7413673, -9.5719366, -10.7454367, -9.5788870, -0.6292214, 0.6535707
5: 1.3328340, 2.2450526, 1.3458240, 2.2478132, -0.6129146, 0.5973158
6: 0.1390119, 1.3371360, 0.1701421, 1.3338921, -0.6174197, 0.6108189
7: -12.6128540, -11.2527924, -12.6104527, -11.2628231, -0.8734756, 0.8884077
8: 6.0788918, 7.0395241, 6.0847816, 7.0398502, -0.8629179, 0.8459005
9: -8.7193155, -7.4472294, -8.7013035, -7.4500966, -1.0670371, 1.0606952

Time for backsubstitution: 21.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 6212
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 4614

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 542

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3542037, upper bound: 0.3556357
time: 4.75 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3542037, upper bound: 0.3556363
time: 5.00 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 31.74 seconds
NS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 31.74
Output dim: 8, lower bound: -0.3540961, upper bound: 0.3534437
NS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 31.74
Output dim: 8, lower bound: -0.3540960, upper bound: 0.3534444
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 31.74
Output dim: 8, lower bound: -0.3542040, upper bound: 0.3542049
NS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 31.74
Output dim: 8, lower bound: -0.3542040, upper bound: 0.3542049
NS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 31.74
Output dim: 8, lower bound: -0.3540963, upper bound: 0.3548758
NS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 31.74
Output dim: 8, lower bound: -0.3540963, upper bound: 0.3548764
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.74
Output dim: 8, lower bound: -0.3528804, upper bound: 0.3556355
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.74
Output dim: 8, lower bound: -0.3528804, upper bound: 0.3556363
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.74
Output dim: 8, lower bound: -0.3542037, upper bound: 0.3556357
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.74
Output dim: 8, lower bound: -0.3542037, upper bound: 0.3556363

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -7.3609896, -6.1416144, -7.3528991, -6.1478872, -0.7149467, 0.7106817
1: -6.6994758, -5.3438864, -6.6961803, -5.3469505, -1.0934706, 1.0954981
2: -4.7901154, -3.6868284, -4.7842855, -3.6856527, -0.7999153, 0.7929001
3: -5.1475668, -3.8339865, -5.1421914, -3.8396063, -0.7617416, 0.7658000
4: -10.7343760, -9.5802269, -10.7254019, -9.5947371, -0.6271193, 0.6285341
5: 1.3475811, 2.2419474, 1.3548111, 2.2364345, -0.5971243, 0.5969589
6: 0.1718673, 1.3298237, 0.1780689, 1.3219048, -0.6035838, 0.6039562
7: -12.6076965, -11.2637405, -12.5990505, -11.2711773, -0.8737307, 0.8727484
8: 6.0876522, 7.0350466, 6.0922174, 7.0318708, -0.8451376, 0.8432126
9: -8.6908684, -7.4531307, -8.6952744, -7.4534888, -1.0416112, 1.0485454

Time for backsubstitution: 21.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6212
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4614

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6212

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3523450, upper bound: 0.3556342
time: 4.69 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3528791, upper bound: 0.3556346
time: 4.63 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -7.3609896, -6.1416144, -7.3639517, -6.1399512, -0.7097096, 0.7103648
1: -6.6994758, -5.3438864, -6.7033253, -5.3417029, -1.0925541, 1.0967312
2: -4.7901154, -3.6868284, -4.7913442, -3.6806018, -0.7974243, 0.7918611
3: -5.1475668, -3.8339865, -5.1549630, -3.8253508, -0.7667832, 0.7623904
4: -10.7343760, -9.5802269, -10.7454290, -9.5788889, -0.6197970, 0.6326926
5: 1.3475811, 2.2419474, 1.3458242, 2.2478123, -0.5984693, 0.5941365
6: 0.1718673, 1.3298237, 0.1701412, 1.3338964, -0.6019471, 0.5981753
7: -12.6076965, -11.2637405, -12.6104527, -11.2628241, -0.8677158, 0.8710065
8: 6.0876522, 7.0350466, 6.0847788, 7.0398459, -0.8456059, 0.8415618
9: -8.6908684, -7.4531307, -8.7013025, -7.4500933, -1.0399251, 1.0499511

Time for backsubstitution: 21.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6212
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4614

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6212

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3523450, upper bound: 0.3556352
time: 6.17 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3528791, upper bound: 0.3556352
time: 5.29 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -7.3677368, -6.1346521, -7.3528967, -6.1478896, -0.7177644, 0.7174118
1: -6.7119355, -5.3364668, -6.6961780, -5.3469505, -1.1067858, 1.1065516
2: -4.8245330, -3.6773720, -4.7842846, -3.6856580, -0.8068390, 0.8045902
3: -5.1613064, -3.7868340, -5.1421828, -3.8396068, -0.7760677, 0.7683737
4: -10.7413673, -9.5719366, -10.7253981, -9.5947380, -0.6363196, 0.6334152
5: 1.3328340, 2.2450526, 1.3548120, 2.2364287, -0.6015129, 0.6001542
6: 0.1390119, 1.3371360, 0.1780686, 1.3219008, -0.6051934, 0.6130781
7: -12.6128540, -11.2527924, -12.5990486, -11.2711782, -0.8795357, 0.8768289
8: 6.0788918, 7.0395241, 6.0922232, 7.0318704, -0.8549190, 0.8454523
9: -8.7193155, -7.4472294, -8.6952744, -7.4534922, -1.0659599, 1.0541477

Time for backsubstitution: 21.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6212
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4614

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6212

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3536709, upper bound: 0.3556346
time: 4.80 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3542024, upper bound: 0.3556346
time: 4.64 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -7.3677368, -6.1346521, -7.3639531, -6.1399498, -0.7164350, 0.7216353
1: -6.7119355, -5.3364668, -6.7033215, -5.3417044, -1.1058683, 1.1137924
2: -4.8245330, -3.6773720, -4.7913427, -3.6806076, -0.8080997, 0.8035746
3: -5.1613064, -3.7868340, -5.1549535, -3.8253527, -0.7811694, 0.7711446
4: -10.7413673, -9.5719366, -10.7454300, -9.5788889, -0.6292210, 0.6411223
5: 1.3328340, 2.2450526, 1.3458247, 2.2478085, -0.6124935, 0.5973155
6: 0.1390119, 1.3371360, 0.1701437, 1.3338897, -0.6133947, 0.6108191
7: -12.6128540, -11.2527924, -12.6104498, -11.2628231, -0.8734746, 0.8836546
8: 6.0788918, 7.0395241, 6.0847807, 7.0398474, -0.8627768, 0.8459010
9: -8.7193155, -7.4472294, -8.7013025, -7.4500971, -1.0670075, 1.0555539

Time for backsubstitution: 21.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6212
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4614

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6212

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3536704, upper bound: 0.3556346
time: 6.67 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3542024, upper bound: 0.3556352
time: 4.88 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 33.67 seconds
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 33.67
Output dim: 8, lower bound: -0.3523450, upper bound: 0.3556342
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 33.67
Output dim: 8, lower bound: -0.3528791, upper bound: 0.3556346
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 33.67
Output dim: 8, lower bound: -0.3523450, upper bound: 0.3556352
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 33.67
Output dim: 8, lower bound: -0.3528791, upper bound: 0.3556352
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 33.67
Output dim: 8, lower bound: -0.3536709, upper bound: 0.3556346
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 33.67
Output dim: 8, lower bound: -0.3542024, upper bound: 0.3556346
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 33.67
Output dim: 8, lower bound: -0.3536704, upper bound: 0.3556346
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 33.67
Output dim: 8, lower bound: -0.3542024, upper bound: 0.3556352

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -7.3562050, -6.1416869, -7.3500705, -6.1479263, -0.7099967, 0.7071524
1: -6.6960282, -5.3442974, -6.6941328, -5.3471918, -1.0888195, 1.0921087
2: -4.7897506, -3.6924253, -4.7840738, -3.6889725, -0.7951689, 0.7868004
3: -5.1445551, -3.8341644, -5.1403866, -3.8397095, -0.7583566, 0.7634783
4: -10.7284479, -9.5815353, -10.7218962, -9.5955000, -0.6203172, 0.6227951
5: 1.3495049, 2.2417746, 1.3559616, 2.2363317, -0.5923033, 0.5925069
6: 0.1725168, 1.3266094, 0.1784526, 1.3199973, -0.6003363, 0.6001184
7: -12.6052961, -11.2647800, -12.5976200, -11.2717962, -0.8702536, 0.8693917
8: 6.0901966, 7.0349298, 6.0937443, 7.0318022, -0.8411117, 0.8401012
9: -8.6876478, -7.4540548, -8.6933670, -7.4540324, -1.0372691, 1.0449772

Time for backsubstitution: 21.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 6212
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 4614

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 522

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3523450, upper bound: 0.3543083
time: 4.76 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3523450, upper bound: 0.3556346
time: 4.42 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 58.95 + 544.86 = 603.81 seconds
