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
execution time: IAR + RelationalAnalysis = 22.95 + 33.21 = 56.16 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.4111793, upper bound: 0.4111788

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 577

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4111644, upper bound: 0.4073552
time: 5.80 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4111772, upper bound: 0.4111764
time: 4.16 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 10.05 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 10.05
Output dim: 7, lower bound: -0.4111644, upper bound: 0.4073552
NS_A2, status: Status.UNKNOWN, split count: 1, time: 10.05
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

Time for backsubstitution: 21.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 577

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4073558, upper bound: 0.4073553
time: 4.10 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4073558, upper bound: 0.4073585
time: 3.40 seconds

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

Time for backsubstitution: 21.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 577

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4073558, upper bound: 0.4111638
time: 3.89 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4073558, upper bound: 0.4111769
time: 4.22 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 29.99 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 29.99
Output dim: 7, lower bound: -0.4073558, upper bound: 0.4073553
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 29.99
Output dim: 7, lower bound: -0.4073558, upper bound: 0.4073585
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 29.99
Output dim: 7, lower bound: -0.4073558, upper bound: 0.4111638
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 29.99
Output dim: 7, lower bound: -0.4073558, upper bound: 0.4111769

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -11.4845066, -9.1909533, -11.4713364, -9.2479630, -1.2456067, 1.2567325
1: -6.5280838, -4.7085686, -6.5234871, -4.7154136, -1.3737273, 1.3770020
2: -6.2393088, -4.1529427, -6.2258978, -4.2193713, -1.3595080, 1.3751137
3: -5.3599424, -3.7379684, -5.3554783, -3.7503870, -0.9954925, 1.0015936
4: -7.4078927, -5.1316452, -7.4019489, -5.1485782, -1.2653322, 1.2773876
5: -10.4926357, -8.5586205, -10.4817085, -8.6021004, -1.0837862, 1.0882215
6: -17.1419563, -14.6817236, -17.1340351, -14.7069778, -1.2682128, 1.2734745
7: 5.0201721, 6.2601023, 5.0498304, 6.2543149, -0.9524975, 0.9418242
8: -6.4556761, -4.6372709, -6.4465561, -4.6751695, -1.0515351, 1.0648600
9: -5.4968801, -3.7843616, -5.4514384, -3.7926455, -1.3205135, 1.2972336

Time for backsubstitution: 22.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4072683, upper bound: 0.4095320
time: 4.55 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4073539, upper bound: 0.4111646
time: 3.33 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -11.4845047, -9.1906223, -11.4845047, -9.1906223, -1.2693551, 1.2704655
1: -6.5281162, -4.7085686, -6.5281162, -4.7085686, -1.3855660, 1.3855662
2: -6.2393103, -4.1525435, -6.2393103, -4.1525435, -1.3875961, 1.3884816
3: -5.3599420, -3.7378776, -5.3599420, -3.7378776, -1.0074937, 1.0072676
4: -7.4078951, -5.1316452, -7.4078951, -5.1316452, -1.2782881, 1.2782881
5: -10.4926357, -8.5582104, -10.4926357, -8.5582104, -1.0986023, 1.0995730
6: -17.1419563, -14.6815901, -17.1419563, -14.6815901, -1.2808220, 1.2818114
7: 5.0200677, 6.2601023, 5.0200677, 6.2601023, -0.9584191, 0.9571342
8: -6.4556766, -4.6370983, -6.4556766, -4.6370983, -1.0725296, 1.0740864
9: -5.4970675, -3.7843614, -5.4970675, -3.7843614, -1.3290057, 1.3271422

Time for backsubstitution: 23.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4072683, upper bound: 0.4095359
time: 3.47 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4073539, upper bound: 0.4111648
time: 3.29 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 30.02 seconds
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 30.02
Output dim: 7, lower bound: -0.4072683, upper bound: 0.4095320
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 30.02
Output dim: 7, lower bound: -0.4073539, upper bound: 0.4111646
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 30.02
Output dim: 7, lower bound: -0.4072683, upper bound: 0.4095359
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 30.02
Output dim: 7, lower bound: -0.4073539, upper bound: 0.4111648

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -11.4809494, -9.1964340, -11.4702044, -9.2497187, -1.2386358, 1.2497731
1: -6.5203352, -4.7120318, -6.5209789, -4.7164998, -1.3629358, 1.3707130
2: -6.2372904, -4.1539555, -6.2252536, -4.2197075, -1.3573856, 1.3727174
3: -5.3428459, -3.7394240, -5.3499908, -3.7508235, -0.9784527, 0.9950271
4: -7.4048791, -5.1412945, -7.4009771, -5.1516743, -1.2573056, 1.2663772
5: -10.4882498, -8.5610771, -10.4802895, -8.6028585, -1.0782614, 1.0841799
6: -17.1397114, -14.6927729, -17.1333313, -14.7105227, -1.2617693, 1.2609285
7: 5.0229063, 6.2554698, 5.0507050, 6.2528286, -0.9474785, 0.9349755
8: -6.4416943, -4.6382217, -6.4420567, -4.6754527, -1.0372400, 1.0580248
9: -5.4918828, -3.7938349, -5.4499097, -3.7956877, -1.3117821, 1.2860279

Time for backsubstitution: 23.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4585

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4072664, upper bound: 0.4088663
time: 3.91 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4072665, upper bound: 0.4095303
time: 3.96 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -11.5338678, -9.1879578, -11.4713306, -9.2479773, -1.2660003, 1.2602895
1: -6.5414791, -4.6606803, -6.5234690, -4.7154198, -1.3986821, 1.3957115
2: -6.2637186, -4.1497469, -6.2258945, -4.2193727, -1.3766489, 1.3782289
3: -5.3692231, -3.6729128, -5.3554592, -3.7503893, -1.0026574, 1.0073452
4: -7.4542217, -5.1294127, -7.4019413, -5.1485920, -1.2822258, 1.2783263
5: -10.4977636, -8.5347281, -10.4817028, -8.6021042, -1.0867858, 1.0937706
6: -17.1901703, -14.6751509, -17.1340294, -14.7070045, -1.2868848, 1.2794217
7: 4.9983835, 6.2624812, 5.0498347, 6.2543092, -0.9588275, 0.9436190
8: -6.4593067, -4.5901718, -6.4465327, -4.6751719, -1.0545695, 1.0707141
9: -5.5366602, -3.7780561, -5.4514294, -3.7926576, -1.3272843, 1.3020415

Time for backsubstitution: 22.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 4585

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4073522, upper bound: 0.4104954
time: 4.05 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4073522, upper bound: 0.4111593
time: 3.68 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -11.4809475, -9.1961021, -11.4833755, -9.1923771, -1.2622881, 1.2635098
1: -6.5203662, -4.7120318, -6.5256085, -4.7096539, -1.3747907, 1.3792911
2: -6.2372899, -4.1535559, -6.2386642, -4.1528797, -1.3854783, 1.3860817
3: -5.3428459, -3.7393346, -5.3544574, -3.7383151, -0.9904594, 1.0007033
4: -7.4048796, -5.1412945, -7.4069214, -5.1347418, -1.2702646, 1.2672318
5: -10.4882498, -8.5606661, -10.4912167, -8.5589676, -1.0930629, 1.0955315
6: -17.1397114, -14.6926384, -17.1412525, -14.6851454, -1.2739851, 1.2692649
7: 5.0228009, 6.2554698, 5.0209403, 6.2586155, -0.9533999, 0.9502782
8: -6.4416947, -4.6380491, -6.4511752, -4.6373801, -1.0581434, 1.0672507
9: -5.4920712, -3.7938361, -5.4955406, -3.7874038, -1.3202736, 1.3158946

Time for backsubstitution: 21.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 4585

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4074211, upper bound: 0.4088807
time: 4.23 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4074211, upper bound: 0.4095446
time: 4.30 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -11.5338650, -9.1876249, -11.4844971, -9.1906414, -1.2862377, 1.2740223
1: -6.5415111, -4.6606793, -6.5280986, -4.7085733, -1.4061990, 1.4035226
2: -6.2637186, -4.1493454, -6.2393045, -4.1525450, -1.4021952, 1.3915967
3: -5.3692226, -3.6728232, -5.3599248, -3.7378814, -1.0148034, 1.0130198
4: -7.4542236, -5.1294122, -7.4078875, -5.1316600, -1.2892306, 1.2791874
5: -10.4977646, -8.5343180, -10.4926300, -8.5582132, -1.1016061, 1.1051224
6: -17.1901703, -14.6750240, -17.1419525, -14.6816196, -1.2931867, 1.2877581
7: 4.9982786, 6.2624812, 5.0200729, 6.2600961, -0.9647486, 0.9589481
8: -6.4593077, -4.5900002, -6.4556499, -4.6371002, -1.0755668, 1.0799404
9: -5.5368500, -3.7780552, -5.4970589, -3.7843728, -1.3357770, 1.3319281

Time for backsubstitution: 21.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 4585

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4075069, upper bound: 0.4105085
time: 4.07 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4075068, upper bound: 0.4111729
time: 3.98 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 29.91 seconds
NS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 29.91
Output dim: 7, lower bound: -0.4072664, upper bound: 0.4088663
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.91
Output dim: 7, lower bound: -0.4072665, upper bound: 0.4095303
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.91
Output dim: 7, lower bound: -0.4073522, upper bound: 0.4104954
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.91
Output dim: 7, lower bound: -0.4073522, upper bound: 0.4111593
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 29.91
Output dim: 7, lower bound: -0.4074211, upper bound: 0.4088807
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.91
Output dim: 7, lower bound: -0.4074211, upper bound: 0.4095446
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.91
Output dim: 7, lower bound: -0.4075069, upper bound: 0.4105085
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.91
Output dim: 7, lower bound: -0.4075068, upper bound: 0.4111729

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -11.4809437, -9.1964407, -11.4701977, -9.2497282, -1.2345419, 1.2399521
1: -6.5203319, -4.7120447, -6.5209732, -4.7165232, -1.3444355, 1.3617567
2: -6.2372751, -4.1539583, -6.2252240, -4.2197127, -1.3414199, 1.3445572
3: -5.3428369, -3.7394261, -5.3499708, -3.7508249, -0.9762642, 0.9786711
4: -7.4048767, -5.1413088, -7.4009719, -5.1517000, -1.2322545, 1.2487072
5: -10.4882460, -8.5610847, -10.4802866, -8.6028700, -1.0579462, 1.0696800
6: -17.1397038, -14.6927738, -17.1333160, -14.7105312, -1.2531142, 1.2541952
7: 5.0229092, 6.2554688, 5.0507131, 6.2528257, -0.9406354, 0.9333019
8: -6.4416857, -4.6382236, -6.4420400, -4.6754541, -1.0372312, 1.0525973
9: -5.4918785, -3.7938557, -5.4499025, -3.7957251, -1.2477369, 1.2655401

Time for backsubstitution: 21.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 6153

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4072657, upper bound: 0.4090821
time: 4.20 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4072657, upper bound: 0.4095296
time: 4.04 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -11.5298595, -9.1920223, -11.4407120, -9.2574844, -1.2528770, 1.2199215
1: -6.5372162, -4.6729155, -6.4879484, -4.7400608, -1.3672476, 1.3463832
2: -6.2504745, -4.1520853, -6.1988621, -4.2688813, -1.3185246, 1.3474303
3: -5.3566065, -3.6739023, -5.3301415, -3.7690558, -0.9705844, 0.9812372
4: -7.4512053, -5.1424870, -7.3685141, -5.1723566, -1.2550006, 1.2268460
5: -10.4950886, -8.5441027, -10.4518690, -8.6196728, -1.0640216, 1.0537626
6: -17.1844673, -14.6786156, -17.1185684, -14.7239017, -1.2603779, 1.2561954
7: 5.0028563, 6.2603593, 5.0606422, 6.2385612, -0.9363534, 0.9297135
8: -6.4517784, -4.5934048, -6.4229288, -4.6886139, -1.0301781, 1.0445595
9: -5.5331697, -3.8083041, -5.4078870, -3.8463333, -1.2677386, 1.2206683

Time for backsubstitution: 21.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 6153

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4073516, upper bound: 0.4100269
time: 3.86 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4073515, upper bound: 0.4104947
time: 3.70 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -11.5338631, -9.1879616, -11.4713268, -9.2479858, -1.2582297, 1.2505052
1: -6.5414762, -4.6606932, -6.5234618, -4.7154436, -1.3745232, 1.3788625
2: -6.2637024, -4.1497498, -6.2258635, -4.2193780, -1.3581376, 1.3500303
3: -5.3692122, -3.6729143, -5.3554392, -3.7503912, -1.0006163, 0.9909531
4: -7.4542198, -5.1294270, -7.4019361, -5.1486192, -1.2521138, 1.2606584
5: -10.4977617, -8.5347347, -10.4816990, -8.6021166, -1.0663948, 1.0792841
6: -17.1901627, -14.6751585, -17.1340199, -14.7070103, -1.2716551, 1.2726892
7: 4.9983883, 6.2624793, 5.0498428, 6.2543058, -0.9520140, 0.9419066
8: -6.4592981, -4.5901737, -6.4465160, -4.6751728, -1.0545607, 1.0652823
9: -5.5366564, -3.7780766, -5.4514227, -3.7926950, -1.2631667, 1.2815839

Time for backsubstitution: 21.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 6153

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4073516, upper bound: 0.4106899
time: 4.24 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4073514, upper bound: 0.4111584
time: 4.13 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -11.4809456, -9.1961060, -11.4833679, -9.1923866, -1.2547805, 1.2536981
1: -6.5203638, -4.7120442, -6.5256033, -4.7096767, -1.3546882, 1.3695893
2: -6.2372746, -4.1535583, -6.2386346, -4.1528840, -1.3669755, 1.3579091
3: -5.3428359, -3.7393363, -5.3544383, -3.7383168, -0.9815230, 0.9843481
4: -7.4048781, -5.1413083, -7.4069161, -5.1347661, -1.2436633, 1.2542713
5: -10.4882460, -8.5606718, -10.4912138, -8.5589800, -1.0700374, 1.0810323
6: -17.1397038, -14.6926413, -17.1412373, -14.6851492, -1.2594182, 1.2625312
7: 5.0228052, 6.2554688, 5.0209489, 6.2586117, -0.9465566, 0.9442024
8: -6.4416857, -4.6380501, -6.4511576, -4.6373835, -1.0510402, 1.0618234
9: -5.4920683, -3.7938550, -5.4955349, -3.7874410, -1.2562072, 1.2840381

Time for backsubstitution: 22.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 6153

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4074204, upper bound: 0.4091018
time: 3.73 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4074204, upper bound: 0.4095443
time: 4.25 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -11.5298576, -9.1916904, -11.4538908, -9.2001572, -1.2731118, 1.2336774
1: -6.5372472, -4.6729150, -6.4926162, -4.7332201, -1.3747585, 1.3541837
2: -6.2504740, -4.1516848, -6.2122631, -4.2020450, -1.3440876, 1.3607880
3: -5.3566070, -3.6738126, -5.3346090, -3.7565451, -0.9801273, 0.9869084
4: -7.4512062, -5.1424875, -7.3744588, -5.1554246, -1.2620046, 1.2306783
5: -10.4950876, -8.5436907, -10.4627981, -8.5757847, -1.0788462, 1.0651155
6: -17.1844673, -14.6784830, -17.1264935, -14.6985006, -1.2667060, 1.2645320
7: 5.0027509, 6.2603593, 5.0308895, 6.2443485, -0.9422746, 0.9450622
8: -6.4517779, -4.5932322, -6.4320331, -4.6505384, -1.0504484, 1.0537603
9: -5.5333581, -3.8083026, -5.4535241, -3.8380625, -1.2762177, 1.2457581

Time for backsubstitution: 22.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 6153

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4075062, upper bound: 0.4100461
time: 4.13 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4075062, upper bound: 0.4105085
time: 3.60 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -11.5338631, -9.1876297, -11.4844933, -9.1906462, -1.2784667, 1.2642460
1: -6.5415087, -4.6606941, -6.5280910, -4.7085972, -1.3799024, 1.3866978
2: -6.2637038, -4.1493483, -6.2392774, -4.1525478, -1.3836911, 1.3633840
3: -5.3692126, -3.6728251, -5.3599043, -3.7378824, -1.0058720, 0.9966273
4: -7.4542217, -5.1294260, -7.4078808, -5.1316872, -1.2591128, 1.2662240
5: -10.4977617, -8.5343237, -10.4926243, -8.5582256, -1.0784860, 1.0906359
6: -17.1901627, -14.6750250, -17.1419411, -14.6816244, -1.2779562, 1.2810254
7: 4.9982839, 6.2624793, 5.0200796, 6.2600937, -0.9579353, 0.9528052
8: -6.4592986, -4.5899992, -6.4556346, -4.6371036, -1.0684640, 1.0745080
9: -5.5368452, -3.7780752, -5.4970527, -3.7844117, -1.2716384, 1.3000841

Time for backsubstitution: 21.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 6153

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4075062, upper bound: 0.4107097
time: 4.38 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4075061, upper bound: 0.4111722
time: 4.00 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 30.42 seconds
NS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 30.42
Output dim: 7, lower bound: -0.4072657, upper bound: 0.4090821
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -0.4072657, upper bound: 0.4095296
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -0.4073516, upper bound: 0.4100269
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -0.4073515, upper bound: 0.4104947
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -0.4073516, upper bound: 0.4106899
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -0.4073514, upper bound: 0.4111584
NS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 30.42
Output dim: 7, lower bound: -0.4074204, upper bound: 0.4091018
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -0.4074204, upper bound: 0.4095443
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -0.4075062, upper bound: 0.4100461
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -0.4075062, upper bound: 0.4105085
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -0.4075062, upper bound: 0.4107097
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -0.4075061, upper bound: 0.4111722

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -11.4809437, -9.1964397, -11.4701977, -9.2497311, -1.2282412, 1.2277977
1: -6.5203276, -4.7120490, -6.5209703, -4.7165241, -1.3444247, 1.3576777
2: -6.2372632, -4.1539593, -6.2252173, -4.2197142, -1.3295984, 1.3326461
3: -5.3428230, -3.7394276, -5.3499632, -3.7508256, -0.9724374, 0.9741973
4: -7.4048605, -5.1413097, -7.4009628, -5.1517010, -1.2121713, 1.2302439
5: -10.4882460, -8.5611057, -10.4802856, -8.6028833, -1.0329912, 1.0342433
6: -17.1396904, -14.6927776, -17.1333103, -14.7105331, -1.2363863, 1.2343078
7: 5.0229149, 6.2554665, 5.0507150, 6.2528248, -0.9320028, 0.9251090
8: -6.4416823, -4.6382251, -6.4420376, -4.6754560, -1.0179639, 1.0439498
9: -5.4918756, -3.7938714, -5.4499021, -3.7957349, -1.2331576, 1.2425377

Time for backsubstitution: 21.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4057212, upper bound: 0.4095297
time: 3.73 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4057212, upper bound: 0.4095329
time: 3.64 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -11.5213871, -9.2012053, -11.4393883, -9.2586136, -1.2347083, 1.2052054
1: -6.5170498, -4.6837773, -6.4824004, -4.7426929, -1.3418717, 1.3261504
2: -6.2312918, -4.1725903, -6.1893902, -4.2698975, -1.2977138, 1.3182118
3: -5.3399343, -3.6874909, -5.3224936, -3.7719927, -0.9525023, 0.9608358
4: -7.4231577, -5.1595840, -7.3546276, -5.1730871, -1.2272992, 1.1925771
5: -10.4589615, -8.5795031, -10.4484100, -8.6392660, -1.0079129, 1.0135250
6: -17.1663094, -14.6983643, -17.1084576, -14.7261467, -1.2404697, 1.2236549
7: 5.0149865, 6.2535448, 5.0636640, 6.2374520, -0.9219971, 0.9152884
8: -6.4400539, -4.6012239, -6.4211373, -4.6909351, -1.0084496, 1.0254233
9: -5.5102291, -3.8283730, -5.4059672, -3.8572364, -1.2293310, 1.1977434

Time for backsubstitution: 21.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4057212, upper bound: 0.4099292
time: 3.46 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4057212, upper bound: 0.4100270
time: 3.45 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -11.5298576, -9.1920261, -11.4407120, -9.2574854, -1.2464795, 1.2079909
1: -6.5372114, -4.6729183, -6.4879465, -4.7400632, -1.3662138, 1.3426758
2: -6.2504635, -4.1520858, -6.1988564, -4.2688823, -1.3081713, 1.3355205
3: -5.3565931, -3.6739037, -5.3301344, -3.7690568, -0.9702449, 0.9766808
4: -7.4511890, -5.1424875, -7.3685060, -5.1723576, -1.2307665, 1.2083838
5: -10.4950857, -8.5441170, -10.4518681, -8.6196823, -1.0415988, 1.0186172
6: -17.1844559, -14.6786156, -17.1185627, -14.7239037, -1.2441287, 1.2363091
7: 5.0028596, 6.2603579, 5.0606461, 6.2385612, -0.9279959, 0.9258466
8: -6.4517756, -4.5934048, -6.4229307, -4.6886144, -1.0138535, 1.0371152
9: -5.5331674, -3.8083181, -5.4078856, -3.8463423, -1.2530022, 1.2018926

Time for backsubstitution: 22.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4057212, upper bound: 0.4104090
time: 8.02 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4057212, upper bound: 0.4104983
time: 3.40 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -11.5254087, -9.1970100, -11.4700003, -9.2492352, -1.2401478, 1.2358555
1: -6.5214224, -4.6716218, -6.5180197, -4.7181463, -1.3499107, 1.3589089
2: -6.2444963, -4.1702700, -6.2163529, -4.2204309, -1.3375750, 1.3219632
3: -5.3525229, -3.6865661, -5.3477569, -3.7533960, -0.9823904, 0.9706943
4: -7.4258943, -5.1464615, -7.3883028, -5.1493683, -1.2243659, 1.2272344
5: -10.4616537, -8.5701275, -10.4782381, -8.6217089, -1.0077052, 1.0393980
6: -17.1716995, -14.6949158, -17.1238174, -14.7092590, -1.2516694, 1.2403604
7: 5.0103631, 6.2556124, 5.0529423, 6.2531242, -0.9376478, 0.9275590
8: -6.4480929, -4.5980024, -6.4440470, -4.6775217, -1.0331244, 1.0455039
9: -5.5135565, -3.7981725, -5.4493833, -3.8036556, -1.2250350, 1.2584087

Time for backsubstitution: 21.97 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.16 + 544.49 = 600.65 seconds
