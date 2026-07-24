## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.4091261895


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2431149, 1.2431149)
1: (-6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3747201, 1.3747203)
2: (-6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3585410, 1.3585410)
3: (-5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9945278, 0.9945278)
4: (-7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2638829, 1.2638830)
5: (-10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0836921, 1.0836918)
6: (-17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2666290, 1.2666286)
7: (5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9420798, 0.9420798)
8: (-6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0508149, 1.0508149)
9: (-5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2966568, 1.2966571)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.01 + 33.88 = 56.89 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.4111793, upper bound: 0.4111788

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 6153
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 577

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4111644, upper bound: 0.4073552
time: 5.96 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4111772, upper bound: 0.4111764
time: 4.17 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 10.27 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 10.27
Output dim: 7, lower bound: -0.4111644, upper bound: 0.4073552
NS_A2, status: Status.UNKNOWN, split count: 1, time: 10.27
Output dim: 7, lower bound: -0.4111772, upper bound: 0.4111764

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -11.4713364, -9.2479630, -11.4776888, -9.2470217, -1.2322178, 1.2386285
1: -6.5234871, -4.7154136, -6.5250597, -4.7153144, -1.3706651, 1.3716414
2: -6.2258978, -4.2193713, -6.2329211, -4.2185659, -1.3465347, 1.3536549
3: -5.3554783, -3.7503870, -5.3563819, -3.7483101, -0.9917011, 0.9919336
4: -7.4019489, -5.1485782, -7.4044294, -5.1484003, -1.2603869, 1.2622690
5: -10.4817085, -8.6021004, -10.4880018, -8.6009693, -1.0730054, 1.0791507
6: -17.1340351, -14.7069778, -17.1377068, -14.7063675, -1.2601812, 1.2637982
7: 5.0498304, 6.2543149, 5.0490975, 6.2576618, -0.9394505, 0.9361970
8: -6.4465561, -4.6751695, -6.4514070, -4.6742043, -1.0425539, 1.0474153
9: -5.4514384, -3.7926455, -5.4517465, -3.7882156, -1.2933538, 1.2890840

Time for backsubstitution: 21.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6153
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6153

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4106955, upper bound: 0.4073550
time: 4.18 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4111637, upper bound: 0.4073545
time: 6.95 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -11.4845047, -9.1906223, -11.4819841, -9.2464142, -1.2446003, 1.2678988
1: -6.5281162, -4.7085686, -6.5261402, -4.7152481, -1.3831148, 1.3803573
2: -6.2393103, -4.1525435, -6.2376690, -4.2180405, -1.3586717, 1.3874538
3: -5.3599420, -3.7378776, -5.3569946, -3.7469800, -0.9972823, 1.0039974
4: -7.4078951, -5.1316452, -7.4061103, -5.1482797, -1.2666938, 1.2807376
5: -10.4926357, -8.5582104, -10.4922523, -8.6001921, -1.0830860, 1.0992011
6: -17.1419563, -14.6815901, -17.1401939, -14.7059708, -1.2674901, 1.2799522
7: 5.0200677, 6.2601023, 5.0486202, 6.2599230, -0.9583642, 0.9408420
8: -6.4556766, -4.6370983, -6.4546862, -4.6735826, -1.0500560, 1.0732734
9: -5.4970675, -3.7843614, -5.4519520, -3.7852218, -1.3281312, 1.2956686

Time for backsubstitution: 21.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6153
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6153

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4107146, upper bound: 0.4073542
time: 4.99 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4111765, upper bound: 0.4111757
time: 4.07 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 31.05 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 31.05
Output dim: 7, lower bound: -0.4106955, upper bound: 0.4073550
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 31.05
Output dim: 7, lower bound: -0.4111637, upper bound: 0.4073545
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 31.05
Output dim: 7, lower bound: -0.4107146, upper bound: 0.4073542
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 31.05
Output dim: 7, lower bound: -0.4111765, upper bound: 0.4111757

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -11.4700165, -9.2492104, -11.4693718, -9.2561817, -1.2175839, 1.2206173
1: -6.5180459, -4.7181168, -6.5046539, -4.7264876, -1.3509951, 1.3469348
2: -6.2163858, -4.2204247, -6.2137637, -4.2391005, -1.3175964, 1.3329322
3: -5.3477955, -3.7533927, -5.3398666, -3.7620056, -0.9721076, 0.9739186
4: -7.3883152, -5.1493273, -7.3760376, -5.1653996, -1.2269566, 1.2343297
5: -10.4782486, -8.6216927, -10.4519997, -8.6363688, -1.0333593, 1.0258524
6: -17.1238384, -14.7092209, -17.1189365, -14.7262077, -1.2277884, 1.2432973
7: 5.0529299, 6.2531338, 5.0612583, 6.2508955, -0.9251580, 0.9217434
8: -6.4440889, -4.6775174, -6.4401846, -4.6820278, -1.0225394, 1.0259938
9: -5.4493985, -3.8036077, -5.4287062, -3.8082352, -1.2702816, 1.2520523

Time for backsubstitution: 22.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 577

## Relational analysis of NS_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4068798, upper bound: 0.4073548
time: 4.92 seconds

## Relational analysis of NS_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4068798, upper bound: 0.4073541
time: 5.18 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -11.4713383, -9.2479630, -11.4776878, -9.2470245, -1.2237597, 1.2399451
1: -6.5234838, -4.7154164, -6.5250549, -4.7153177, -1.3681047, 1.3716319
2: -6.2258911, -4.2193708, -6.2329106, -4.2185664, -1.3423722, 1.3480008
3: -5.3554697, -3.7503872, -5.3563681, -3.7483115, -0.9926381, 0.9915922
4: -7.4019399, -5.1485796, -7.4044156, -5.1484013, -1.2523274, 1.2439215
5: -10.4817076, -8.6021118, -10.4880009, -8.6009874, -1.0451479, 1.0611137
6: -17.1340294, -14.7069788, -17.1376953, -14.7063675, -1.2491417, 1.2554868
7: 5.0498323, 6.2543144, 5.0491009, 6.2576599, -0.9371507, 0.9329817
8: -6.4465570, -4.6751719, -6.4514027, -4.6742063, -1.0467393, 1.0309074
9: -5.4514370, -3.7926555, -5.4517441, -3.7882311, -1.2746148, 1.2890704

Time for backsubstitution: 23.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 577

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4073552, upper bound: 0.4073545
time: 4.10 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4073552, upper bound: 0.4073547
time: 4.56 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -11.4831848, -9.1918716, -11.4736681, -9.2555752, -1.2299707, 1.2498846
1: -6.5227089, -4.7112727, -6.5057325, -4.7264233, -1.3634031, 1.3556442
2: -6.2297988, -4.1536016, -6.2185102, -4.2385750, -1.3297176, 1.3668710
3: -5.3522630, -3.7408845, -5.3404803, -3.7606781, -0.9776967, 0.9859773
4: -7.3942604, -5.1323934, -7.3777170, -5.1652813, -1.2332582, 1.2527310
5: -10.4891758, -8.5778046, -10.4562502, -8.6355906, -1.0434458, 1.0403366
6: -17.1317577, -14.6838322, -17.1214256, -14.7258158, -1.2350974, 1.2594948
7: 5.0231643, 6.2589197, 5.0607853, 6.2531557, -0.9440448, 0.9263854
8: -6.4532146, -4.6394472, -6.4434633, -4.6814089, -1.0300512, 1.0519700
9: -5.4950285, -3.7953103, -5.4289122, -3.8052411, -1.3050673, 1.2586589

Time for backsubstitution: 22.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4106149, upper bound: 0.4095454
time: 4.29 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4107124, upper bound: 0.4111737
time: 3.48 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -11.4845057, -9.1906223, -11.4819851, -9.2464190, -1.2361434, 1.2616487
1: -6.5281138, -4.7085710, -6.5261350, -4.7152524, -1.3805542, 1.3803475
2: -6.2393036, -4.1525431, -6.2376580, -4.2180414, -1.3545232, 1.3756440
3: -5.3599358, -3.7378793, -5.3569813, -3.7469821, -0.9982200, 1.0001724
4: -7.4078856, -5.1316462, -7.4060955, -5.1482797, -1.2571077, 1.2562770
5: -10.4926338, -8.5582218, -10.4922476, -8.6002083, -1.0552399, 1.0742669
6: -17.1419525, -14.6815929, -17.1401844, -14.7059746, -1.2564602, 1.2631797
7: 5.0200701, 6.2601008, 5.0486240, 6.2599211, -0.9501820, 0.9376290
8: -6.4556742, -4.6371007, -6.4546843, -4.6735840, -1.0542803, 1.0520022
9: -5.4970665, -3.7843702, -5.4519491, -3.7852380, -1.3051915, 1.2956555

Time for backsubstitution: 23.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4095466, upper bound: 0.4110882
time: 5.10 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4111745, upper bound: 0.4111737
time: 5.23 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 33.60 seconds
NS_A1_B1_B1, status: Status.VERIFIED, split count: 3, time: 33.60
Output dim: 7, lower bound: -0.4068798, upper bound: 0.4073548
NS_A1_B1_B2, status: Status.VERIFIED, split count: 3, time: 33.60
Output dim: 7, lower bound: -0.4068798, upper bound: 0.4073541
NS_A1_B2_B1, status: Status.VERIFIED, split count: 3, time: 33.60
Output dim: 7, lower bound: -0.4073552, upper bound: 0.4073545
NS_A1_B2_B2, status: Status.VERIFIED, split count: 3, time: 33.60
Output dim: 7, lower bound: -0.4073552, upper bound: 0.4073547
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 33.60
Output dim: 7, lower bound: -0.4106149, upper bound: 0.4095454
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 33.60
Output dim: 7, lower bound: -0.4107124, upper bound: 0.4111737
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 33.60
Output dim: 7, lower bound: -0.4095466, upper bound: 0.4110882
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 33.60
Output dim: 7, lower bound: -0.4111745, upper bound: 0.4111737

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -11.4796009, -9.1973486, -11.4725237, -9.2573175, -1.2229717, 1.2429364
1: -6.5149651, -4.7147522, -6.5032358, -4.7275262, -1.3525560, 1.3493230
2: -6.2277861, -4.1546087, -6.2178726, -4.2389107, -1.3276243, 1.3645010
3: -5.3351393, -3.7423391, -5.3349791, -3.7611134, -0.9606261, 0.9793994
4: -7.3912382, -5.1420474, -7.3767385, -5.1683807, -1.2251930, 1.2417057
5: -10.4847965, -8.5802717, -10.4548264, -8.6363659, -1.0378888, 1.0362639
6: -17.1295185, -14.6948891, -17.1207390, -14.7293682, -1.2287517, 1.2469717
7: 5.0258732, 6.2542858, 5.0616417, 6.2516623, -0.9390466, 0.9195886
8: -6.4392366, -4.6403995, -6.4389658, -4.6816893, -1.0157647, 1.0451455
9: -5.4900141, -3.8047998, -5.4273705, -3.8082919, -1.2963300, 1.2474954

Time for backsubstitution: 22.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 577

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4067799, upper bound: 0.4095316
time: 4.81 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4067799, upper bound: 0.4095322
time: 3.77 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -11.5325422, -9.1887999, -11.4736576, -9.2555923, -1.2549386, 1.2535620
1: -6.5362635, -4.6633229, -6.5057144, -4.7264280, -1.3835368, 1.3744322
2: -6.2542152, -4.1504211, -6.2185059, -4.2385788, -1.3523414, 1.3700092
3: -5.3615465, -3.6758113, -5.3404622, -3.7606802, -0.9848733, 0.9917492
4: -7.4406204, -5.1301775, -7.3777103, -5.1652946, -1.2494073, 1.2536755
5: -10.4942284, -8.5539351, -10.4562435, -8.6355953, -1.0464330, 1.0459020
6: -17.1801224, -14.6772251, -17.1214218, -14.7258463, -1.2565205, 1.2654529
7: 5.0013723, 6.2612519, 5.0607905, 6.2531500, -0.9504638, 0.9281309
8: -6.4568558, -4.5923471, -6.4434366, -4.6814079, -1.0331144, 1.0578208
9: -5.5348001, -3.7889864, -5.4289041, -3.8052530, -1.3117871, 1.2635839

Time for backsubstitution: 21.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 577

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4068775, upper bound: 0.4111638
time: 3.32 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4068775, upper bound: 0.4111738
time: 5.43 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: -11.4833755, -9.1923790, -11.4784212, -9.2519007, -1.2291164, 1.2545277
1: -6.5256057, -4.7096558, -6.5183883, -4.7187195, -1.3742335, 1.3695583
2: -6.2386580, -4.1528807, -6.2356458, -4.2190547, -1.3521240, 1.3735474
3: -5.3544507, -3.7383156, -5.3398757, -3.7484374, -0.9923124, 0.9831319
4: -7.4069128, -5.1347423, -7.4030838, -5.1579332, -1.2460964, 1.2478398
5: -10.4912167, -8.5589800, -10.4878635, -8.6026669, -1.0511942, 1.0687057
6: -17.1412430, -14.6851463, -17.1379356, -14.7170076, -1.2439039, 1.2564118
7: 5.0209436, 6.2586145, 5.0513649, 6.2552900, -0.9433327, 0.9327781
8: -6.4511733, -4.6373806, -6.4407039, -4.6745381, -1.0474544, 1.0376161
9: -5.4955401, -3.7874126, -5.4469404, -3.7947080, -1.2939115, 1.2871679

Time for backsubstitution: 23.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 577

## Relational analysis of NS_A2_B2_B1_B1

### Relational analysis result of NS_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4057235, upper bound: 0.4110751
time: 4.46 seconds

## Relational analysis of NS_A2_B2_B1_B2

### Relational analysis result of NS_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4057235, upper bound: 0.4110753
time: 6.16 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: -11.4844990, -9.1906376, -11.5313501, -9.2434206, -1.2407246, 1.2782564
1: -6.5280957, -4.7085752, -6.5396004, -4.6673574, -1.3957109, 1.4001304
2: -6.2392993, -4.1525450, -6.2620640, -4.2148476, -1.3576407, 1.3902040
3: -5.3599167, -3.7378821, -5.3662891, -3.6819301, -1.0059080, 1.0075185
4: -7.4078784, -5.1316619, -7.4524217, -5.1460557, -1.2580402, 1.2634714
5: -10.4926291, -8.5582256, -10.4973783, -8.5762997, -1.0607858, 1.0771754
6: -17.1419487, -14.6816206, -17.1883965, -14.6994228, -1.2623572, 1.2757945
7: 5.0200758, 6.2600956, 5.0268331, 6.2623014, -0.9519324, 0.9455718
8: -6.4556494, -4.6371017, -6.4583139, -4.6264825, -1.0601275, 1.0550429
9: -5.4970589, -3.7843833, -5.4917436, -3.7789168, -1.3098607, 1.3026178

Time for backsubstitution: 23.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 577

## Relational analysis of NS_A2_B2_B2_B1

### Relational analysis result of NS_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4073530, upper bound: 0.4111609
time: 4.25 seconds

## Relational analysis of NS_A2_B2_B2_B2

### Relational analysis result of NS_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4073530, upper bound: 0.4111748
time: 4.34 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 32.19 seconds
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 32.19
Output dim: 7, lower bound: -0.4067799, upper bound: 0.4095316
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 32.19
Output dim: 7, lower bound: -0.4067799, upper bound: 0.4095322
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 32.19
Output dim: 7, lower bound: -0.4068775, upper bound: 0.4111638
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 32.19
Output dim: 7, lower bound: -0.4068775, upper bound: 0.4111738
NS_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 32.19
Output dim: 7, lower bound: -0.4057235, upper bound: 0.4110751
NS_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 32.19
Output dim: 7, lower bound: -0.4057235, upper bound: 0.4110753
NS_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 32.19
Output dim: 7, lower bound: -0.4073530, upper bound: 0.4111609
NS_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 32.19
Output dim: 7, lower bound: -0.4073530, upper bound: 0.4111748

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -11.4796028, -9.1976814, -11.4618740, -9.2588625, -1.2239778, 1.2317684
1: -6.5149336, -4.7147508, -6.5005875, -4.7276912, -1.3431816, 1.3459737
2: -6.2277856, -4.1550088, -6.2061005, -4.2402382, -1.3284650, 1.3521593
3: -5.3351378, -3.7424302, -5.3334570, -3.7645183, -0.9588389, 0.9769911
4: -7.3912387, -5.1420465, -7.3725753, -5.1686783, -1.2238269, 1.2383542
5: -10.4847965, -8.5806847, -10.4442835, -8.6382771, -1.0385866, 1.0252848
6: -17.1295185, -14.6950178, -17.1145782, -14.7303658, -1.2294755, 1.2404948
7: 5.0259776, 6.2542868, 5.0628414, 6.2460556, -0.9331799, 0.9205775
8: -6.4392357, -4.6405716, -6.4308400, -4.6832757, -1.0172441, 1.0367316
9: -5.4898267, -3.8048019, -5.4268599, -3.8157198, -1.2887101, 1.2490602

Time for backsubstitution: 23.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4585

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4061148, upper bound: 0.4095296
time: 3.97 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4067779, upper bound: 0.4095296
time: 4.07 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -11.4796009, -9.1973486, -11.4750481, -9.2015152, -1.2476749, 1.2455156
1: -6.5149651, -4.7147522, -6.5052705, -4.7208467, -1.3549824, 1.3545477
2: -6.2277861, -4.1546087, -6.2195005, -4.1734042, -1.3568027, 1.3655188
3: -5.3351393, -3.7423391, -5.3379178, -3.7520099, -0.9698762, 0.9826605
4: -7.3912382, -5.1420474, -7.3785105, -5.1517391, -1.2367852, 1.2392734
5: -10.4847965, -8.5802717, -10.4552135, -8.5943890, -1.0533087, 1.0366402
6: -17.1295185, -14.6948891, -17.1225014, -14.7049713, -1.2416179, 1.2488314
7: 5.0258732, 6.2542858, 5.0330553, 6.2518425, -0.9391015, 0.9359140
8: -6.4392366, -4.6403995, -6.4399676, -4.6452026, -1.0383344, 1.0459526
9: -5.4900141, -3.8047998, -5.4724965, -3.8074439, -1.2971931, 1.2775233

Time for backsubstitution: 22.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4585

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4061148, upper bound: 0.4095298
time: 6.14 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4067779, upper bound: 0.4095298
time: 4.41 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -11.5325394, -9.1891317, -11.4630108, -9.2571363, -1.2511997, 1.2423946
1: -6.5362320, -4.6633224, -6.5030656, -4.7265916, -1.3785481, 1.3710762
2: -6.2542143, -4.1508198, -6.2067337, -4.2399058, -1.3479631, 1.3576679
3: -5.3615465, -3.6759019, -5.3389416, -3.7640865, -0.9830849, 0.9893421
4: -7.4406195, -5.1301775, -7.3735461, -5.1655917, -1.2489283, 1.2503235
5: -10.4942274, -8.5543470, -10.4457016, -8.6375065, -1.0471308, 1.0349233
6: -17.1801224, -14.6773577, -17.1152611, -14.7268467, -1.2546966, 1.2589759
7: 5.0014768, 6.2612519, 5.0619917, 6.2475419, -0.9445970, 0.9291198
8: -6.4568548, -4.5925198, -6.4353113, -4.6829948, -1.0345945, 1.0494065
9: -5.5346131, -3.7889888, -5.4283910, -3.8126814, -1.3041663, 1.2651522

Time for backsubstitution: 22.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4585

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4062123, upper bound: 0.4111620
time: 3.68 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4068754, upper bound: 0.4111586
time: 4.26 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -11.5325422, -9.1887999, -11.4761858, -9.1997871, -1.2714446, 1.2561373
1: -6.5362635, -4.6633229, -6.5077491, -4.7197495, -1.3859797, 1.3788897
2: -6.2542152, -4.1504211, -6.2201362, -4.1730714, -1.3735204, 1.3710293
3: -5.3615465, -3.6758113, -5.3434000, -3.7515764, -0.9942474, 0.9950091
4: -7.4406204, -5.1301775, -7.3794847, -5.1486545, -1.2559352, 1.2512460
5: -10.4942284, -8.5539351, -10.4566307, -8.5936155, -1.0619237, 1.0462785
6: -17.1801224, -14.6772251, -17.1231842, -14.7014465, -1.2610087, 1.2673123
7: 5.0013723, 6.2612519, 5.0322065, 6.2533288, -0.9505181, 0.9444827
8: -6.4568558, -4.5923471, -6.4444366, -4.6449213, -1.0557766, 1.0586275
9: -5.5348001, -3.7889864, -5.4740243, -3.8044047, -1.3126504, 1.2935545

Time for backsubstitution: 21.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4585

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4062123, upper bound: 0.4111594
time: 3.69 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4068754, upper bound: 0.4111626
time: 3.36 seconds

## BFS NS instance: NS_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -11.4833755, -9.1927128, -11.4677715, -9.2534466, -1.2300134, 1.2433603
1: -6.5255737, -4.7096558, -6.5157347, -4.7188835, -1.3648400, 1.3662090
2: -6.2386570, -4.1532812, -6.2238755, -4.2203856, -1.3477364, 1.3612062
3: -5.3544493, -3.7384064, -5.3383584, -3.7518449, -0.9905245, 0.9807270
4: -7.4069109, -5.1347427, -7.3989215, -5.1582308, -1.2456174, 1.2444868
5: -10.4912186, -8.5593929, -10.4773216, -8.6045771, -1.0470276, 1.0577266
6: -17.1412430, -14.6852789, -17.1317768, -14.7180119, -1.2420793, 1.2499344
7: 5.0210476, 6.2586145, 5.0525742, 6.2496815, -0.9374657, 0.9326046
8: -6.4511719, -4.6375542, -6.4325743, -4.6761236, -1.0452542, 1.0292032
9: -5.4953518, -3.7874134, -5.4464293, -3.8021326, -1.2862878, 1.2853026

Time for backsubstitution: 21.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 4585

## Relational analysis of NS_A2_B2_B1_B1_B1

### Relational analysis result of NS_A2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4057217, upper bound: 0.4104093
time: 7.37 seconds

## Relational analysis of NS_A2_B2_B1_B1_B2

### Relational analysis result of NS_A2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4057217, upper bound: 0.4110732
time: 3.66 seconds

## BFS NS instance: NS_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -11.4833755, -9.1923790, -11.4809475, -9.1961021, -1.2502487, 1.2571100
1: -6.5256057, -4.7096558, -6.5203614, -4.7120352, -1.3766880, 1.3747804
2: -6.2386580, -4.1528807, -6.2372808, -4.1535568, -1.3732855, 1.3745518
3: -5.3544507, -3.7383156, -5.3428335, -3.7393367, -0.9964558, 0.9864070
4: -7.4069128, -5.1347423, -7.4048648, -5.1412950, -1.2526231, 1.2500373
5: -10.4912167, -8.5589800, -10.4882479, -8.5606852, -1.0591252, 1.0690786
6: -17.1412430, -14.6851463, -17.1396980, -14.6926413, -1.2483888, 1.2582710
7: 5.0209436, 6.2586145, 5.0228057, 6.2554698, -0.9433866, 0.9434826
8: -6.4511733, -4.6373806, -6.4416909, -4.6380506, -1.0570579, 1.0384288
9: -5.4955401, -3.7874126, -5.4920673, -3.7938497, -1.2947605, 1.3038306

Time for backsubstitution: 22.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4585

## Relational analysis of NS_A2_B2_B1_B2_B1

### Relational analysis result of NS_A2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4057217, upper bound: 0.4104227
time: 5.90 seconds

## Relational analysis of NS_A2_B2_B1_B2_B2

### Relational analysis result of NS_A2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4057217, upper bound: 0.4110735
time: 6.26 seconds

## BFS NS instance: NS_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -11.4844961, -9.1909733, -11.5207043, -9.2449675, -1.2405572, 1.2670918
1: -6.5280628, -4.7085757, -6.5369492, -4.6675220, -1.3927782, 1.3967887
2: -6.2392993, -4.1529450, -6.2502966, -4.2161775, -1.3532529, 1.3778634
3: -5.3599157, -3.7379723, -5.3647738, -3.6853352, -1.0035259, 1.0051175
4: -7.4078770, -5.1316614, -7.4482617, -5.1463537, -1.2575610, 1.2601191
5: -10.4926281, -8.5586395, -10.4868336, -8.5782166, -1.0566196, 1.0661960
6: -17.1419487, -14.6817532, -17.1822376, -14.7004251, -1.2605357, 1.2693176
7: 5.0201802, 6.2600956, 5.0280409, 6.2566929, -0.9460657, 0.9439594
8: -6.4556494, -4.6372738, -6.4501848, -4.6280680, -1.0579271, 1.0466299
9: -5.4968691, -3.7843821, -5.4912305, -3.7863371, -1.3022370, 1.3007522

Time for backsubstitution: 22.55 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.89 + 545.67 = 602.56 seconds
