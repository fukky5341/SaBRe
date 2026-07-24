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
execution time: IAR + LP analysis = 13.88 + 33.16 = 47.05 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -1.6885415, upper bound: 1.6885395


# Binary Search by BASE starts (time budget: 3552.95 seconds, max iter: 100)

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
rel_dist={7: [-1.086373347751306, 1.0863705612652526]}

## Binary Search Result
Binary search time: 195.59 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual) starts
Time budget: 3357.36 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6178
type: A, layer: 1, pos: 6178
type: B, layer: 1, pos: 484
type: A, layer: 1, pos: 484
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6178

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4384856, upper bound: 1.4295511
time: 4.12 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4397741, upper bound: 1.4397737
time: 4.10 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8.40 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 8.40
Output dim: 7, lower bound: -1.4384856, upper bound: 1.4295511
IS_B2, status: Status.UNKNOWN, split count: 1, time: 8.40
Output dim: 7, lower bound: -1.4397741, upper bound: 1.4397737

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -7.1167135, -4.3948116, -7.1096139, -4.3970814, -2.6053743, 2.5981588
1: -7.2918773, -5.0625052, -7.2482762, -5.0689411, -2.2229362, 2.1857710
2: -6.1067004, -4.0262651, -6.0847487, -4.0312190, -1.9074836, 1.8909681
3: -6.1529870, -3.5656323, -6.1070867, -3.5716767, -2.5813103, 2.5414543
4: -6.4903264, -4.0586848, -6.4741940, -4.0632663, -2.4270601, 2.4155092
5: -6.5214386, -4.3049474, -6.5166945, -4.3161192, -2.2053194, 2.2117472
6: -11.4815340, -8.7031775, -11.4735241, -8.7056084, -2.7759256, 2.7703466
7: 2.7555785, 4.8178535, 2.7820542, 4.8129635, -2.0573850, 2.0357993
8: -4.4057159, -2.0530095, -4.4016609, -2.0721102, -2.2365389, 2.2553084
9: -2.7913332, -1.0601335, -2.7858586, -1.0757484, -1.7155848, 1.7257252

Time for backsubstitution: 14.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 484
type: B, layer: 1, pos: 484
type: A, layer: 1, pos: 6178
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 484

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4276878, upper bound: 1.4286936
time: 4.10 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4384771, upper bound: 1.4295437
time: 4.27 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -7.1188107, -4.3942251, -7.1353807, -4.3758678, -2.6328206, 2.6685865
1: -7.3048606, -5.0606899, -7.3234901, -4.9762707, -2.3285899, 2.2628002
2: -6.1131382, -4.0248537, -6.1204958, -3.9811716, -1.9513216, 1.9240859
3: -6.1664476, -3.5639722, -6.1979318, -3.4905531, -2.6758945, 2.6339595
4: -6.4951210, -4.0573521, -6.5056715, -4.0280843, -2.4670367, 2.4483194
5: -6.5228047, -4.3014235, -6.5651617, -4.2898498, -2.2329550, 2.2637382
6: -11.4839745, -8.7024994, -11.5080719, -8.6733360, -2.8106384, 2.8055725
7: 2.7477808, 4.8194666, 2.7353382, 4.8440499, -2.0962691, 2.0841284
8: -4.4071112, -2.0474181, -4.4523430, -2.0392408, -2.2723336, 2.3080888
9: -2.7929382, -1.0555971, -2.8258357, -1.0504484, -1.7424898, 1.7702386

Time for backsubstitution: 14.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 484
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 484

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4287774, upper bound: 1.4386681
time: 3.96 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4397656, upper bound: 1.4397656
time: 3.91 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 22.19 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 22.19
Output dim: 7, lower bound: -1.4276878, upper bound: 1.4286936
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 22.19
Output dim: 7, lower bound: -1.4384771, upper bound: 1.4295437
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 22.19
Output dim: 7, lower bound: -1.4287774, upper bound: 1.4386681
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 22.19
Output dim: 7, lower bound: -1.4397656, upper bound: 1.4397656

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -7.0943785, -4.4508586, -7.1048584, -4.4102497, -2.5988698, 2.5155947
1: -7.2781496, -5.0944562, -7.2452073, -5.0765495, -2.2016001, 2.1507511
2: -6.0977430, -4.0368423, -6.0828886, -4.0337739, -1.8818631, 1.8694868
3: -6.1286755, -3.6026235, -6.1011467, -3.5801091, -2.5485663, 2.4985232
4: -6.4748774, -4.1108203, -6.4708867, -4.0755334, -2.3993440, 2.3600664
5: -6.4950600, -4.3261499, -6.5101223, -4.3211603, -2.1738997, 2.1839724
6: -11.4422150, -8.7098093, -11.4643517, -8.7071190, -2.7350960, 2.7545424
7: 2.7844098, 4.8076410, 2.7888451, 4.8106918, -2.0262821, 2.0187960
8: -4.4001493, -2.0669212, -4.4004230, -2.0753741, -2.2130165, 2.2275379
9: -2.7787511, -1.0757506, -2.7825842, -1.0795026, -1.6992486, 1.7068336

Time for backsubstitution: 14.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6178
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6178

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4188953, upper bound: 1.4286934
time: 4.23 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4188953, upper bound: 1.4286936
time: 4.30 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -7.1810560, -4.3886976, -7.1096125, -4.3970876, -2.6564105, 2.5975931
1: -7.3203130, -5.0540600, -7.2482729, -5.0689492, -2.2513638, 2.1942129
2: -6.1390324, -4.0203843, -6.0847468, -4.0312223, -1.9452171, 1.8880780
3: -6.1968780, -3.5566118, -6.1070819, -3.5716882, -2.6251898, 2.5504701
4: -6.5343971, -4.0445900, -6.4741912, -4.0632777, -2.4711194, 2.4296012
5: -6.5354171, -4.2912874, -6.5166893, -4.3161240, -2.2192931, 2.2254019
6: -11.4980125, -8.6781216, -11.4735146, -8.7056112, -2.7924013, 2.7953930
7: 2.7386179, 4.8388195, 2.7820611, 4.8129625, -2.0743446, 2.0567584
8: -4.4242043, -2.0446377, -4.4016595, -2.0721130, -2.2687473, 2.2564282
9: -2.8046117, -1.0515718, -2.7858565, -1.0757511, -1.7288606, 1.7342846

Time for backsubstitution: 14.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6178
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6178

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4295480, upper bound: 1.4295440
time: 4.49 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4295480, upper bound: 1.4295438
time: 5.25 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -7.0964489, -4.4503407, -7.1305518, -4.3891129, -2.6269722, 2.5759869
1: -7.2911444, -5.0926909, -7.3204861, -4.9839249, -2.3072195, 2.2277951
2: -6.1041842, -4.0354781, -6.1186171, -3.9837077, -1.9244165, 1.9029574
3: -6.1421585, -3.6009605, -6.1920447, -3.4989896, -2.6431689, 2.5910842
4: -6.4796715, -4.1095390, -6.5024080, -4.0403891, -2.4392824, 2.3928690
5: -6.4963503, -4.3226309, -6.5585332, -4.2948804, -2.2014699, 2.2359023
6: -11.4446554, -8.7091513, -11.4988708, -8.6748686, -2.7697868, 2.7897196
7: 2.7766683, 4.8092537, 2.7421601, 4.8417754, -2.0651071, 2.0670936
8: -4.4016089, -2.0613432, -4.4511261, -2.0425081, -2.2488041, 2.2797372
9: -2.7802920, -1.0712206, -2.8224840, -1.0541885, -1.7261035, 1.7512634

Time for backsubstitution: 14.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 484
type: A, layer: 1, pos: 6178
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 484

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4287757, upper bound: 1.4287752
time: 4.20 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4287756, upper bound: 1.4386680
time: 4.12 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -7.1831546, -4.3880963, -7.1353774, -4.3758774, -2.6814003, 2.6678600
1: -7.3332815, -5.0522251, -7.3234859, -4.9762788, -2.3570027, 2.2712607
2: -6.1454792, -4.0189538, -6.1204934, -3.9811749, -1.9739385, 1.9212122
3: -6.2103443, -3.5549514, -6.1979260, -3.4905646, -2.7197797, 2.6429746
4: -6.5391860, -4.0432386, -6.5056701, -4.0280972, -2.5110888, 2.4624314
5: -6.5367990, -4.2877522, -6.5651541, -4.2898545, -2.2469444, 2.2774019
6: -11.5003824, -8.6774368, -11.5080614, -8.6733379, -2.8270445, 2.8306246
7: 2.7308028, 4.8404393, 2.7353449, 4.8440475, -2.1132448, 2.1050944
8: -4.4256763, -2.0390491, -4.4523420, -2.0392427, -2.3045506, 2.3092103
9: -2.8062215, -1.0470444, -2.8258333, -1.0504514, -1.7557701, 1.7787889

Time for backsubstitution: 14.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 484
type: A, layer: 1, pos: 6178
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 484

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4386683, upper bound: 1.4287750
time: 3.90 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4386682, upper bound: 1.4287750
time: 4.01 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 22.27 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 22.27
Output dim: 7, lower bound: -1.4188953, upper bound: 1.4286934
IS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 22.27
Output dim: 7, lower bound: -1.4188953, upper bound: 1.4286936
IS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 22.27
Output dim: 7, lower bound: -1.4295480, upper bound: 1.4295440
IS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 22.27
Output dim: 7, lower bound: -1.4295480, upper bound: 1.4295438
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 22.27
Output dim: 7, lower bound: -1.4287757, upper bound: 1.4287752
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 22.27
Output dim: 7, lower bound: -1.4287756, upper bound: 1.4386680
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 22.27
Output dim: 7, lower bound: -1.4386683, upper bound: 1.4287750
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 22.27
Output dim: 7, lower bound: -1.4386682, upper bound: 1.4287750

## BFS IS instance: IS_B1_A1_A1

### Backsubstitution after applying IS history:
0: -7.0873675, -4.4529572, -7.1048584, -4.4102497, -2.5700498, 2.5003083
1: -7.2345042, -5.1007624, -7.2452073, -5.0765495, -2.1579547, 2.1444449
2: -6.0758004, -4.0416355, -6.0828886, -4.0337739, -1.8600917, 1.8645539
3: -6.0826745, -3.6086719, -6.1011467, -3.5801091, -2.5025654, 2.4924748
4: -6.4587431, -4.1152668, -6.4708867, -4.0755334, -2.3832097, 2.3556199
5: -6.4905062, -4.3373127, -6.5101223, -4.3211603, -2.1693459, 2.1728096
6: -11.4341793, -8.7121630, -11.4643517, -8.7071190, -2.7270603, 2.7521887
7: 2.8107338, 4.8027568, 2.7888451, 4.8106918, -1.9999580, 2.0139117
8: -4.3960266, -2.0859809, -4.4004230, -2.0753741, -2.2085247, 2.2036774
9: -2.7734327, -1.0913429, -2.7825842, -1.0795026, -1.6939301, 1.6912413

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 484
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 484

## Relational analysis of IS_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4188953, upper bound: 1.4188881
time: 4.02 seconds

## Relational analysis of IS_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4188954, upper bound: 1.4286935
time: 3.95 seconds

## BFS IS instance: IS_B1_A1_A2

### Backsubstitution after applying IS history:
0: -7.1115336, -4.4355459, -7.1048584, -4.4102497, -2.5935974, 2.5176685
1: -7.3082838, -5.0106854, -7.2452073, -5.0765495, -2.2317343, 2.2345219
2: -6.1114631, -3.9931753, -6.0828886, -4.0337739, -1.8961821, 1.8999412
3: -6.1708059, -3.5282285, -6.1011467, -3.5801091, -2.5906968, 2.5729182
4: -6.4885826, -4.0830750, -6.4708867, -4.0755334, -2.4130492, 2.3878117
5: -6.5350027, -4.3126602, -6.5101223, -4.3211603, -2.2138424, 2.1974621
6: -11.4671707, -8.6813583, -11.4643517, -8.7071190, -2.7600517, 2.7829933
7: 2.7662263, 4.8332481, 2.7888451, 4.8106918, -2.0444655, 2.0444031
8: -4.4457912, -2.0537186, -4.4004230, -2.0753741, -2.2520790, 2.2352724
9: -2.8091018, -1.0671957, -2.7825842, -1.0795026, -1.7295992, 1.7153885

Time for backsubstitution: 14.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 484

## Relational analysis of IS_B1_A1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4188953, upper bound: 1.4188884
time: 4.05 seconds

## Relational analysis of IS_B1_A1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4188954, upper bound: 1.4286937
time: 4.10 seconds

## BFS IS instance: IS_B1_A2_A1

### Backsubstitution after applying IS history:
0: -7.1739616, -4.3910222, -7.1096125, -4.3970876, -2.6339056, 2.5794597
1: -7.2767582, -5.0605559, -7.2482729, -5.0689492, -2.2078090, 2.1877170
2: -6.1170545, -4.0253410, -6.0847468, -4.0312223, -1.9235392, 1.8827512
3: -6.1509519, -3.5626521, -6.1070819, -3.5716882, -2.5792637, 2.5444298
4: -6.5182810, -4.0492325, -6.4741912, -4.0632777, -2.4550033, 2.4249587
5: -6.5306158, -4.3024988, -6.5166893, -4.3161240, -2.2144918, 2.2141905
6: -11.4897709, -8.6805611, -11.4735146, -8.7056112, -2.7841597, 2.7929535
7: 2.7651443, 4.8339019, 2.7820611, 4.8129625, -2.0478182, 2.0518408
8: -4.4201031, -2.0637283, -4.4016595, -2.0721130, -2.2642760, 2.2330987
9: -2.7991209, -1.0671490, -2.7858565, -1.0757511, -1.7233698, 1.7187074

Time for backsubstitution: 14.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 484
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 484

## Relational analysis of IS_B1_A2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4286980, upper bound: 1.4188884
time: 4.21 seconds

## Relational analysis of IS_B1_A2_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4286985, upper bound: 1.4219949
time: 4.78 seconds

## BFS IS instance: IS_B1_A2_A2

### Backsubstitution after applying IS history:
0: -7.1996927, -4.3699932, -7.1096125, -4.3970876, -2.6542790, 2.5991971
1: -7.3518257, -4.9677014, -7.2482729, -5.0689492, -2.2828765, 2.2805715
2: -6.1527548, -3.9752190, -6.0847468, -4.0312223, -1.9577045, 1.9199409
3: -6.2412105, -3.4821949, -6.1070819, -3.5716882, -2.6695223, 2.6248870
4: -6.5496721, -4.0138793, -6.4741912, -4.0632777, -2.4863944, 2.4603119
5: -6.5792551, -4.2765975, -6.5166893, -4.3161240, -2.2631311, 2.2400918
6: -11.5247574, -8.6482992, -11.4735146, -8.7056112, -2.8191462, 2.8252153
7: 2.7186079, 4.8644338, 2.7820611, 4.8129625, -2.0943546, 2.0823727
8: -4.4700241, -2.0309587, -4.4016595, -2.0721130, -2.2937078, 2.2652669
9: -2.8384423, -1.0418103, -2.7858565, -1.0757511, -1.7626913, 1.7440462

Time for backsubstitution: 14.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 484

## Relational analysis of IS_B1_A2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4286978, upper bound: 1.4188879
time: 4.70 seconds

## Relational analysis of IS_B1_A2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4286985, upper bound: 1.4219947
time: 4.60 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -7.0964489, -4.4503407, -7.1124840, -4.4339428, -2.5774512, 2.5576134
1: -7.2911444, -5.0926909, -7.3096709, -5.0097485, -2.2813959, 2.2169800
2: -6.1041842, -4.0354781, -6.1114635, -3.9922206, -1.9086390, 1.8883061
3: -6.1421585, -3.6009605, -6.1724005, -3.5275595, -2.6145990, 2.5714400
4: -6.4796715, -4.1095390, -6.4900475, -4.0819707, -2.3977008, 2.3805084
5: -6.4963503, -4.3226309, -6.5364413, -4.3113866, -2.1849637, 2.2138104
6: -11.4446554, -8.7091513, -11.4677963, -8.6802998, -2.7643557, 2.7586451
7: 2.7766683, 4.8092537, 2.7652836, 4.8338270, -2.0571587, 2.0439701
8: -4.4016089, -2.0613432, -4.4467916, -2.0534730, -2.2276607, 2.2673662
9: -2.7802920, -1.0712206, -2.8109851, -1.0662770, -1.7140150, 1.7397645

Time for backsubstitution: 14.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6178
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6178

## Relational analysis of IS_B2_A1_B1_A1

### Relational analysis result of IS_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4188904, upper bound: 1.4276856
time: 4.05 seconds

## Relational analysis of IS_B2_A1_B1_A2

### Relational analysis result of IS_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4188904, upper bound: 1.4188885
time: 4.11 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.0964489, -4.4503407, -7.1997185, -4.3695583, -2.6493440, 2.6066914
1: -7.2911444, -5.0926909, -7.3518538, -4.9676790, -2.3234653, 2.2591629
2: -6.1041842, -4.0354781, -6.1527548, -3.9751024, -1.9268742, 1.9305234
3: -6.1421585, -3.6009605, -6.2412319, -3.4815264, -2.6606321, 2.6402714
4: -6.4796715, -4.1095390, -6.5496879, -4.0138531, -2.4658184, 2.4401488
5: -6.4963503, -4.3226309, -6.5792732, -4.2764201, -2.2199302, 2.2566423
6: -11.4446554, -8.7091513, -11.5248861, -8.6482897, -2.7963657, 2.8157349
7: 2.7766683, 4.8092537, 2.7185712, 4.8650117, -2.0883434, 2.0906825
8: -4.4016089, -2.0613432, -4.4710274, -2.0309582, -2.2562599, 2.2948649
9: -2.7802920, -1.0712206, -2.8390017, -1.0417790, -1.7385130, 1.7677810

Time for backsubstitution: 14.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6178
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6178

## Relational analysis of IS_B2_A1_B2_A1

### Relational analysis result of IS_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4188906, upper bound: 1.4375306
time: 4.72 seconds

## Relational analysis of IS_B2_A1_B2_A2

### Relational analysis result of IS_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4188906, upper bound: 1.4286937
time: 4.12 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.1831546, -4.3880963, -7.1124840, -4.4339428, -2.6246519, 2.6335096
1: -7.3332815, -5.0522251, -7.3096709, -5.0097485, -2.3235331, 2.2574458
2: -6.1454792, -4.0189538, -6.1114635, -3.9922206, -1.9454374, 1.9006829
3: -6.2103443, -3.5549514, -6.1724005, -3.5275595, -2.6827848, 2.6174490
4: -6.5391860, -4.0432386, -6.4900475, -4.0819707, -2.4572153, 2.4468088
5: -6.5367990, -4.2877522, -6.5364413, -4.3113866, -2.2254124, 2.2486892
6: -11.5003824, -8.6774368, -11.4677963, -8.6802998, -2.8200827, 2.7903595
7: 2.7308028, 4.8404393, 2.7652836, 4.8338270, -2.1030242, 2.0751557
8: -4.4256763, -2.0390491, -4.4467916, -2.0534730, -2.2606144, 2.2900355
9: -2.8062215, -1.0470444, -2.8109851, -1.0662770, -1.7399445, 1.7639407

Time for backsubstitution: 14.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_B2_A2_B1_B1

### Relational analysis result of IS_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4290805, upper bound: 1.4287649
time: 4.23 seconds

## Relational analysis of IS_B2_A2_B1_B2

### Relational analysis result of IS_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4386576, upper bound: 1.4287649
time: 4.23 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.1831546, -4.3880963, -7.1997185, -4.3695583, -2.6720853, 2.6934447
1: -7.3332815, -5.0522251, -7.3518538, -4.9676790, -2.3656025, 2.2996287
2: -6.1454792, -4.0189538, -6.1527548, -3.9751024, -1.9801598, 1.9680049
3: -6.2103443, -3.5549514, -6.2412319, -3.4815264, -2.7288179, 2.6862805
4: -6.5391860, -4.0432386, -6.5496879, -4.0138531, -2.5253329, 2.5064492
5: -6.5367990, -4.2877522, -6.5792732, -4.2764201, -2.2603788, 2.2915211
6: -11.5003824, -8.6774368, -11.5248861, -8.6482897, -2.8520927, 2.8474493
7: 2.7308028, 4.8404393, 2.7185712, 4.8650117, -2.1342089, 2.1218681
8: -4.4256763, -2.0390491, -4.4710274, -2.0309582, -2.3145251, 2.3342960
9: -2.8062215, -1.0470444, -2.8390017, -1.0417790, -1.7644424, 1.7919573

Time for backsubstitution: 14.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_B2_A2_B2_B1

### Relational analysis result of IS_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4290810, upper bound: 1.4188774
time: 4.94 seconds

## Relational analysis of IS_B2_A2_B2_B2

### Relational analysis result of IS_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4386580, upper bound: 1.4322580
time: 4.07 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 23.34 seconds
IS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.34
Output dim: 7, lower bound: -1.4188953, upper bound: 1.4188881
IS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.34
Output dim: 7, lower bound: -1.4188954, upper bound: 1.4286935
IS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.34
Output dim: 7, lower bound: -1.4188953, upper bound: 1.4188884
IS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.34
Output dim: 7, lower bound: -1.4188954, upper bound: 1.4286937
IS_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.34
Output dim: 7, lower bound: -1.4286980, upper bound: 1.4188884
IS_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.34
Output dim: 7, lower bound: -1.4286985, upper bound: 1.4219949
IS_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.34
Output dim: 7, lower bound: -1.4286978, upper bound: 1.4188879
IS_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.34
Output dim: 7, lower bound: -1.4286985, upper bound: 1.4219947
IS_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 23.34
Output dim: 7, lower bound: -1.4188904, upper bound: 1.4276856
IS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 23.34
Output dim: 7, lower bound: -1.4188904, upper bound: 1.4188885
IS_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 23.34
Output dim: 7, lower bound: -1.4188906, upper bound: 1.4375306
IS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 23.34
Output dim: 7, lower bound: -1.4188906, upper bound: 1.4286937
IS_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 23.34
Output dim: 7, lower bound: -1.4290805, upper bound: 1.4287649
IS_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 23.34
Output dim: 7, lower bound: -1.4386576, upper bound: 1.4287649
IS_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 23.34
Output dim: 7, lower bound: -1.4290810, upper bound: 1.4188774
IS_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 23.34
Output dim: 7, lower bound: -1.4386580, upper bound: 1.4322580

## BFS IS instance: IS_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -7.0873675, -4.4529572, -7.0873675, -4.4529572, -2.4720907, 2.4720907
1: -7.2345042, -5.1007624, -7.2345042, -5.1007624, -2.1337419, 2.1337419
2: -6.0758004, -4.0416355, -6.0758004, -4.0416355, -1.8499591, 1.8499594
3: -6.0826745, -3.6086719, -6.0826745, -3.6086719, -2.4740026, 2.4740026
4: -6.4587431, -4.1152668, -6.4587431, -4.1152668, -2.3434763, 2.3434763
5: -6.4905062, -4.3373127, -6.4905062, -4.3373127, -2.1531935, 2.1531935
6: -11.4341793, -8.7121630, -11.4341793, -8.7121630, -2.7220163, 2.7220163
7: 2.8107338, 4.8027568, 2.8107338, 4.8027568, -1.9920230, 1.9920230
8: -4.3960266, -2.0859809, -4.3960266, -2.0859809, -2.1902523, 2.1902523
9: -2.7734327, -1.0913429, -2.7734327, -1.0913429, -1.6820898, 1.6820898

Time for backsubstitution: 14.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_B1_A1_A1_B1_B1

### Relational analysis result of IS_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4092036, upper bound: 1.4188827
time: 4.31 seconds

## Relational analysis of IS_B1_A1_A1_B1_B2

### Relational analysis result of IS_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4188840, upper bound: 1.4188822
time: 4.36 seconds

## BFS IS instance: IS_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -7.0873675, -4.4529572, -7.1739616, -4.3910222, -2.5920997, 2.5531535
1: -7.2345042, -5.1007624, -7.2767582, -5.0605559, -2.1739483, 2.1759958
2: -6.0758004, -4.0416355, -6.1170545, -4.0253410, -1.8622675, 1.8921533
3: -6.0826745, -3.6086719, -6.1509519, -3.5626521, -2.5200224, 2.5422800
4: -6.4587431, -4.1152668, -6.5182810, -4.0492325, -2.4095106, 2.4030142
5: -6.4905062, -4.3373127, -6.5306158, -4.3024988, -2.1880074, 2.1933031
6: -11.4341793, -8.7121630, -11.4897709, -8.6805611, -2.7536182, 2.7776079
7: 2.8107338, 4.8027568, 2.7651443, 4.8339019, -2.0231681, 2.0376124
8: -4.3960266, -2.0859809, -4.4201031, -2.0637283, -2.2158303, 2.2231107
9: -2.7734327, -1.0913429, -2.7991209, -1.0671490, -1.7062837, 1.7077780

Time for backsubstitution: 14.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 466

## Relational analysis of IS_B1_A1_A1_B2_A1

### Relational analysis result of IS_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4188846, upper bound: 1.4191958
time: 4.45 seconds

## Relational analysis of IS_B1_A1_A1_B2_A2

### Relational analysis result of IS_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4188846, upper bound: 1.4286875
time: 4.11 seconds

## BFS IS instance: IS_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -7.1115336, -4.4355459, -7.0873675, -4.4529572, -2.4957294, 2.4887772
1: -7.3082838, -5.0106854, -7.2345042, -5.1007624, -2.2075214, 2.2238188
2: -6.1114631, -3.9931753, -6.0758004, -4.0416355, -1.8857768, 1.8862712
3: -6.1708059, -3.5282285, -6.0826745, -3.6086719, -2.5621340, 2.5544460
4: -6.4885826, -4.0830750, -6.4587431, -4.1152668, -2.3733158, 2.3756680
5: -6.5350027, -4.3126602, -6.4905062, -4.3373127, -2.1976900, 2.1772571
6: -11.4671707, -8.6813583, -11.4341793, -8.7121630, -2.7550077, 2.7528210
7: 2.7662263, 4.8332481, 2.8107338, 4.8027568, -2.0365305, 2.0225143
8: -4.4457912, -2.0537186, -4.3960266, -2.0859809, -2.2359686, 2.2218473
9: -2.8091018, -1.0671957, -2.7734327, -1.0913429, -1.7177589, 1.7062371

Time for backsubstitution: 14.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_B1_A1_A2_B1_B1

### Relational analysis result of IS_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4180250, upper bound: 1.4188777
time: 4.24 seconds

## Relational analysis of IS_B1_A1_A2_B1_B2

### Relational analysis result of IS_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4276770, upper bound: 1.4188774
time: 4.33 seconds

## BFS IS instance: IS_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -7.1115336, -4.4355459, -7.1739616, -4.3910222, -2.6156468, 2.5658410
1: -7.3082838, -5.0106854, -7.2767582, -5.0605559, -2.2477279, 2.2660728
2: -6.1114631, -3.9931753, -6.1170545, -4.0253410, -1.8983583, 1.9232688
3: -6.1708059, -3.5282285, -6.1509519, -3.5626521, -2.6081538, 2.6227233
4: -6.4885826, -4.0830750, -6.5182810, -4.0492325, -2.4393501, 2.4352059
5: -6.5350027, -4.3126602, -6.5306158, -4.3024988, -2.2325039, 2.2179556
6: -11.4671707, -8.6813583, -11.4897709, -8.6805611, -2.7866096, 2.8084126
7: 2.7662263, 4.8332481, 2.7651443, 4.8339019, -2.0676756, 2.0681038
8: -4.4457912, -2.0537186, -4.4201031, -2.0637283, -2.2592764, 2.2547059
9: -2.8091018, -1.0671957, -2.7991209, -1.0671490, -1.7419528, 1.7319252

Time for backsubstitution: 14.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_B1_A1_A2_B2_B1

### Relational analysis result of IS_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4180250, upper bound: 1.4286834
time: 4.28 seconds

## Relational analysis of IS_B1_A1_A2_B2_B2

### Relational analysis result of IS_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4276769, upper bound: 1.4286832
time: 4.13 seconds

## BFS IS instance: IS_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -7.1739616, -4.3910222, -7.0873675, -4.4529572, -2.5531538, 2.5920994
1: -7.2767582, -5.0605559, -7.2345042, -5.1007624, -2.1759958, 2.1739483
2: -6.1170545, -4.0253410, -6.0758004, -4.0416355, -1.8921530, 1.8622675
3: -6.1509519, -3.5626521, -6.0826745, -3.6086719, -2.5422800, 2.5200224
4: -6.5182810, -4.0492325, -6.4587431, -4.1152668, -2.4030142, 2.4095106
5: -6.5306158, -4.3024988, -6.4905062, -4.3373127, -2.1933031, 2.1880074
6: -11.4897709, -8.6805611, -11.4341793, -8.7121630, -2.7776079, 2.7536182
7: 2.7651443, 4.8339019, 2.8107338, 4.8027568, -2.0376124, 2.0231681
8: -4.4201031, -2.0637283, -4.3960266, -2.0859809, -2.2231107, 2.2158303
9: -2.7991209, -1.0671490, -2.7734327, -1.0913429, -1.7077780, 1.7062837

Time for backsubstitution: 14.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_B1_A2_A1_B1_B1

### Relational analysis result of IS_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4191955, upper bound: 1.4188825
time: 4.58 seconds

## Relational analysis of IS_B1_A2_A1_B1_B2

### Relational analysis result of IS_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4286871, upper bound: 1.4188823
time: 4.42 seconds

## BFS IS instance: IS_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -7.1739616, -4.3910222, -7.1739616, -4.3910222, -2.6191616, 2.6191616
1: -7.2767582, -5.0605559, -7.2767582, -5.0605559, -2.2162023, 2.2162023
2: -6.1170545, -4.0253410, -6.1170545, -4.0253410, -1.9296317, 1.9296312
3: -6.1509519, -3.5626521, -6.1509519, -3.5626521, -2.5882998, 2.5882998
4: -6.5182810, -4.0492325, -6.5182810, -4.0492325, -2.4690485, 2.4690485
5: -6.5306158, -4.3024988, -6.5306158, -4.3024988, -2.2281170, 2.2281170
6: -11.4897709, -8.6805611, -11.4897709, -8.6805611, -2.8092098, 2.8092098
7: 2.7651443, 4.8339019, 2.7651443, 4.8339019, -2.0687575, 2.0687575
8: -4.4201031, -2.0637283, -4.4201031, -2.0637283, -2.2741084, 2.2741086
9: -2.7991209, -1.0671490, -2.7991209, -1.0671490, -1.7319719, 1.7319719

Time for backsubstitution: 14.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 466

## Relational analysis of IS_B1_A2_A1_B2_A1

### Relational analysis result of IS_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4286880, upper bound: 1.4127950
time: 4.54 seconds

## Relational analysis of IS_B1_A2_A1_B2_A2

### Relational analysis result of IS_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4286880, upper bound: 1.4219894
time: 4.83 seconds

## BFS IS instance: IS_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -7.1996927, -4.3699932, -7.0873675, -4.4529572, -2.5736377, 2.6115711
1: -7.3518257, -4.9677014, -7.2345042, -5.1007624, -2.2510633, 2.2668028
2: -6.1527548, -3.9752190, -6.0758004, -4.0416355, -1.9279945, 1.8984519
3: -6.2412105, -3.4821949, -6.0826745, -3.6086719, -2.6325386, 2.6004796
4: -6.5496721, -4.0138793, -6.4587431, -4.1152668, -2.4344053, 2.4448638
5: -6.5792551, -4.2765975, -6.4905062, -4.3373127, -2.2419424, 2.2139087
6: -11.5247574, -8.6482992, -11.4341793, -8.7121630, -2.8125944, 2.7858801
7: 2.7186079, 4.8644338, 2.8107338, 4.8027568, -2.0841489, 2.0537000
8: -4.4700241, -2.0309587, -4.3960266, -2.0859809, -2.2634687, 2.2479985
9: -2.8384423, -1.0418103, -2.7734327, -1.0913429, -1.7470994, 1.7316225

Time for backsubstitution: 14.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_B1_A2_A2_B1_B1

### Relational analysis result of IS_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4279963, upper bound: 1.4188775
time: 4.54 seconds

## Relational analysis of IS_B1_A2_A2_B1_B2

### Relational analysis result of IS_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4375198, upper bound: 1.4188774
time: 4.00 seconds

## BFS IS instance: IS_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -7.1996927, -4.3699932, -7.1739616, -4.3910222, -2.6446586, 2.6388993
1: -7.3518257, -4.9677014, -7.2767582, -5.0605559, -2.2912698, 2.3090568
2: -6.1527548, -3.9752190, -6.1170545, -4.0253410, -1.9638839, 1.9517167
3: -6.2412105, -3.4821949, -6.1509519, -3.5626521, -2.6785583, 2.6687570
4: -6.5496721, -4.0138793, -6.5182810, -4.0492325, -2.5004396, 2.5044017
5: -6.5792551, -4.2765975, -6.5306158, -4.3024988, -2.2767563, 2.2540183
6: -11.5247574, -8.6482992, -11.4897709, -8.6805611, -2.8441963, 2.8414717
7: 2.7186079, 4.8644338, 2.7651443, 4.8339019, -2.1152940, 2.0992894
8: -4.4700241, -2.0309587, -4.4201031, -2.0637283, -2.3035231, 2.3062766
9: -2.8384423, -1.0418103, -2.7991209, -1.0671490, -1.7712933, 1.7573106

Time for backsubstitution: 14.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_B1_A2_A2_B2_B1

### Relational analysis result of IS_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4279969, upper bound: 1.4188774
time: 4.69 seconds

## Relational analysis of IS_B1_A2_A2_B2_B2

### Relational analysis result of IS_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4375203, upper bound: 1.4188776
time: 4.45 seconds

## BFS IS instance: IS_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7.0873675, -4.4529572, -7.1124582, -4.4343777, -2.5396752, 2.5072246
1: -7.2345042, -5.1007624, -7.3096418, -5.0097733, -2.2247310, 2.2088795
2: -6.0758004, -4.0416355, -6.1114631, -3.9923365, -1.8802176, 1.8857770
3: -6.0826745, -3.6086719, -6.1723776, -3.5282285, -2.5544460, 2.5637057
4: -6.4587431, -4.1152668, -6.4900303, -4.0819979, -2.3767452, 2.3747635
5: -6.4905062, -4.3373127, -6.5364227, -4.3115625, -2.1789436, 2.1991100
6: -11.4341793, -8.7121630, -11.4676676, -8.6803122, -2.7538671, 2.7555046
7: 2.8107338, 4.8027568, 2.7653210, 4.8332481, -2.0225143, 2.0374358
8: -4.3960266, -2.0859809, -4.4457912, -2.0534739, -2.2195745, 2.2359686
9: -2.7734327, -1.0913429, -2.8104258, -1.0663083, -1.7071245, 1.7190828

Time for backsubstitution: 14.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 466

## Relational analysis of IS_B2_A1_B1_A1_A1

### Relational analysis result of IS_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4188795, upper bound: 1.4180243
time: 4.21 seconds

## Relational analysis of IS_B2_A1_B1_A1_A2

### Relational analysis result of IS_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4188795, upper bound: 1.4276771
time: 4.33 seconds

## BFS IS instance: IS_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.1115494, -4.4351096, -7.1124840, -4.4339428, -2.6266148, 2.5664952
1: -7.3082991, -5.0106626, -7.3096709, -5.0097485, -2.2985506, 2.2990084
2: -6.1114635, -3.9930665, -6.1114635, -3.9922206, -1.9080596, 1.9165411
3: -6.1708164, -3.5275595, -6.1724005, -3.5275595, -2.6432569, 2.6448410
4: -6.4885845, -4.0830474, -6.4900475, -4.0819707, -2.4066138, 2.4070001
5: -6.5350218, -4.3124952, -6.5364413, -4.3113866, -2.2230759, 2.2239461
6: -11.4672985, -8.6813564, -11.4677963, -8.6802998, -2.7869987, 2.7864399
7: 2.7661929, 4.8338270, 2.7652836, 4.8338270, -2.0676341, 2.0685434
8: -4.4467916, -2.0537195, -4.4467916, -2.0534730, -2.2415528, 2.2442145
9: -2.8096611, -1.0671734, -2.8109851, -1.0662770, -1.7433841, 1.7438117

Time for backsubstitution: 14.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_B2_A1_B1_A2_B1

### Relational analysis result of IS_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4091987, upper bound: 1.4188793
time: 4.33 seconds

## Relational analysis of IS_B2_A1_B1_A2_B2

### Relational analysis result of IS_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4188791, upper bound: 1.4219876
time: 4.21 seconds

## BFS IS instance: IS_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.0873675, -4.4529572, -7.1996927, -4.3699932, -2.6115713, 2.5736382
1: -7.2345042, -5.1007624, -7.3518257, -4.9677014, -2.2668028, 2.2510633
2: -6.0758004, -4.0416355, -6.1527548, -3.9752190, -1.8984518, 1.9279945
3: -6.0826745, -3.6086719, -6.2412105, -3.4821949, -2.6004796, 2.6325386
4: -6.4587431, -4.1152668, -6.5496721, -4.0138793, -2.4448638, 2.4344053
5: -6.4905062, -4.3373127, -6.5792551, -4.2765975, -2.2139087, 2.2419424
6: -11.4341793, -8.7121630, -11.5247574, -8.6482992, -2.7858801, 2.8125944
7: 2.8107338, 4.8027568, 2.7186079, 4.8644338, -2.0537000, 2.0841489
8: -4.3960266, -2.0859809, -4.4700241, -2.0309587, -2.2479987, 2.2634685
9: -2.7734327, -1.0913429, -2.8384423, -1.0418103, -1.7316225, 1.7470994

Time for backsubstitution: 14.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 466

## Relational analysis of IS_B2_A1_B2_A1_A1

### Relational analysis result of IS_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4188795, upper bound: 1.4279965
time: 4.29 seconds

## Relational analysis of IS_B2_A1_B2_A1_A2

### Relational analysis result of IS_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4188795, upper bound: 1.4375205
time: 4.47 seconds

## BFS IS instance: IS_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.1115494, -4.4351096, -7.1997185, -4.3695583, -2.6809459, 2.6155729
1: -7.3082991, -5.0106626, -7.3518538, -4.9676790, -2.3406200, 2.3411913
2: -6.1114635, -3.9930665, -6.1527548, -3.9751024, -1.9277830, 1.9555333
3: -6.1708164, -3.5275595, -6.2412319, -3.4815264, -2.6892900, 2.7136724
4: -6.4885845, -4.0830474, -6.5496879, -4.0138531, -2.4747314, 2.4666405
5: -6.5350218, -4.3124952, -6.5792732, -4.2764201, -2.2585733, 2.2667780
6: -11.4672985, -8.6813564, -11.5248861, -8.6482897, -2.8190088, 2.8435297
7: 2.7661929, 4.8338270, 2.7185712, 4.8650117, -2.0988188, 2.1152558
8: -4.4467916, -2.0537195, -4.4710274, -2.0309582, -2.2701521, 2.2771168
9: -2.8096611, -1.0671734, -2.8390017, -1.0417790, -1.7678821, 1.7718283

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 466

## Relational analysis of IS_B2_A1_B2_A2_A1

### Relational analysis result of IS_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4188794, upper bound: 1.4220909
time: 4.49 seconds

## Relational analysis of IS_B2_A1_B2_A2_A2

### Relational analysis result of IS_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4188795, upper bound: 1.4316958
time: 4.25 seconds

## BFS IS instance: IS_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -7.1808686, -4.3931084, -7.1033707, -4.4534731, -2.6017985, 2.6174083
1: -7.3270140, -5.0538316, -7.2855148, -5.0161743, -2.3108397, 2.2316833
2: -6.1376801, -4.0200467, -6.0811687, -3.9967079, -1.9266100, 1.8662353
3: -6.2034817, -3.5568948, -6.1459827, -3.5349965, -2.6684852, 2.5890880
4: -6.5370135, -4.0498948, -6.4813428, -4.1080570, -2.4289565, 2.4314480
5: -6.5351348, -4.2896099, -6.5296783, -4.3185248, -2.2166100, 2.2400684
6: -11.4965191, -8.6836567, -11.4522171, -8.7045059, -2.7920132, 2.7685604
7: 2.7323623, 4.8292756, 2.7714953, 4.7902269, -2.0578647, 2.0577803
8: -4.4192729, -2.0403762, -4.4218078, -2.0588770, -2.2451925, 2.2579918
9: -2.7980146, -1.0486125, -2.7790065, -1.0728186, -1.7251960, 1.7303940

Time for backsubstitution: 14.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6178

## Relational analysis of IS_B2_A2_B1_B1_A1

### Relational analysis result of IS_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4191905, upper bound: 1.4276772
time: 4.49 seconds

## Relational analysis of IS_B2_A2_B1_B1_A2

### Relational analysis result of IS_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4191906, upper bound: 1.4188795
time: 4.54 seconds

## BFS IS instance: IS_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -7.1831522, -4.3881016, -7.1331038, -4.4276876, -2.6290424, 2.6414614
1: -7.3332772, -5.0522256, -7.3164535, -5.0066042, -2.3266730, 2.2642279
2: -6.1454778, -4.0189538, -6.1144357, -3.9650075, -1.9522364, 1.9009001
3: -6.2103395, -3.5549521, -6.1885905, -3.5238309, -2.6865087, 2.6336384
4: -6.5391846, -4.0432429, -6.5035110, -4.0732484, -2.4659362, 2.4602680
5: -6.5367985, -4.2877531, -6.5446815, -4.3061771, -2.2306213, 2.2569284
6: -11.5003815, -8.6774406, -11.4938602, -8.6785336, -2.8218479, 2.8164196
7: 2.7308033, 4.8404303, 2.7390676, 4.8433981, -2.1125948, 2.1013627
8: -4.4256721, -2.0390511, -4.4533453, -2.0370193, -2.2800012, 2.2920377
9: -2.8062165, -1.0470455, -2.8163342, -1.0534602, -1.7527562, 1.7692888

Time for backsubstitution: 14.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6178

## Relational analysis of IS_B2_A2_B1_B2_A1

### Relational analysis result of IS_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4286824, upper bound: 1.4276772
time: 5.23 seconds

## Relational analysis of IS_B2_A2_B1_B2_A2

### Relational analysis result of IS_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4286824, upper bound: 1.4188791
time: 4.70 seconds

## BFS IS instance: IS_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -7.1808686, -4.3931084, -7.1904774, -4.3890514, -2.6491117, 2.6772645
1: -7.3270140, -5.0538316, -7.3277006, -4.9740982, -2.3529158, 2.2738690
2: -6.1376801, -4.0200467, -6.1224656, -3.9796007, -1.9612808, 1.9335601
3: -6.2034817, -3.5568948, -6.2148943, -3.4888985, -2.7145832, 2.6579995
4: -6.5370135, -4.0498948, -6.5409966, -4.0397944, -2.4972191, 2.4911017
5: -6.5351348, -4.2896099, -6.5727043, -4.2836108, -2.2515240, 2.2830944
6: -11.4965191, -8.6836567, -11.5093269, -8.6725035, -2.8240156, 2.8256702
7: 2.7323623, 4.8292756, 2.7248144, 4.8214893, -2.0891271, 2.1044612
8: -4.4192729, -2.0403762, -4.4461222, -2.0363350, -2.2990870, 2.3022542
9: -2.7980146, -1.0486125, -2.8070433, -1.0483285, -1.7496861, 1.7584308

Time for backsubstitution: 14.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6178

## Relational analysis of IS_B2_A2_B2_B1_A1

### Relational analysis result of IS_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4203446, upper bound: 1.4309961
time: 4.57 seconds

## Relational analysis of IS_B2_A2_B2_B1_A2

### Relational analysis result of IS_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4203446, upper bound: 1.4219849
time: 4.76 seconds

## BFS IS instance: IS_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -7.1831522, -4.3881016, -7.2201357, -4.3632202, -2.6768799, 2.7013211
1: -7.3332772, -5.0522256, -7.3585672, -4.9644861, -2.3687911, 2.3063416
2: -6.1454778, -4.0189538, -6.1556811, -3.9478798, -1.9869161, 1.9682653
3: -6.2103395, -3.5549521, -6.2571878, -3.4777350, -2.7326045, 2.7022357
4: -6.5391846, -4.0432429, -6.5631661, -4.0049477, -2.5342369, 2.5199232
5: -6.5367985, -4.2877531, -6.5879488, -4.2711306, -2.2656679, 2.3001957
6: -11.5003815, -8.6774406, -11.5509644, -8.6465340, -2.8538475, 2.8735237
7: 2.7308033, 4.8404303, 2.6923342, 4.8744860, -2.1436827, 2.1480961
8: -4.4256721, -2.0390511, -4.4776030, -2.0144415, -2.3268480, 2.3362823
9: -2.8062165, -1.0470455, -2.8443823, -1.0289630, -1.7772535, 1.7973368

Time for backsubstitution: 14.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6178

## Relational analysis of IS_B2_A2_B2_B2_A1

### Relational analysis result of IS_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4295327, upper bound: 1.4309959
time: 4.45 seconds

## Relational analysis of IS_B2_A2_B2_B2_A2

### Relational analysis result of IS_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4295327, upper bound: 1.4249733
time: 4.40 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 23.25 seconds
IS_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.4092036, upper bound: 1.4188827
IS_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.4188840, upper bound: 1.4188822
IS_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.4188846, upper bound: 1.4191958
IS_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.4188846, upper bound: 1.4286875
IS_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.4180250, upper bound: 1.4188777
IS_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.4276770, upper bound: 1.4188774
IS_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.4180250, upper bound: 1.4286834
IS_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.4276769, upper bound: 1.4286832
IS_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.4191955, upper bound: 1.4188825
IS_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.4286871, upper bound: 1.4188823
IS_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.4286880, upper bound: 1.4127950
IS_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.4286880, upper bound: 1.4219894
IS_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.4279963, upper bound: 1.4188775
IS_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.4375198, upper bound: 1.4188774
IS_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.4279969, upper bound: 1.4188774
IS_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.4375203, upper bound: 1.4188776
IS_B2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.4188795, upper bound: 1.4180243
IS_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.4188795, upper bound: 1.4276771
IS_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.4091987, upper bound: 1.4188793
IS_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.4188791, upper bound: 1.4219876
IS_B2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.4188795, upper bound: 1.4279965
IS_B2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.4188795, upper bound: 1.4375205
IS_B2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.4188794, upper bound: 1.4220909
IS_B2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.4188795, upper bound: 1.4316958
IS_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.4191905, upper bound: 1.4276772
IS_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.4191906, upper bound: 1.4188795
IS_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.4286824, upper bound: 1.4276772
IS_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.4286824, upper bound: 1.4188791
IS_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.4203446, upper bound: 1.4309961
IS_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.4203446, upper bound: 1.4219849
IS_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.4295327, upper bound: 1.4309959
IS_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.4295327, upper bound: 1.4249733

## BFS IS instance: IS_B1_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -7.0851660, -4.4578733, -7.0785074, -4.4720697, -2.4251995, 2.4576061
1: -7.2282372, -5.1023021, -7.2102599, -5.1066265, -2.1216106, 2.0495412
2: -6.0680065, -4.0426846, -6.0455213, -4.0458918, -1.8351703, 1.7569247
3: -6.0757732, -3.6106358, -6.0556769, -3.6161091, -2.4596641, 2.4450412
4: -6.4565878, -4.1218572, -6.4499207, -4.1408119, -2.3157759, 2.3280635
5: -6.4889255, -4.3391752, -6.4842033, -4.3443871, -2.1343751, 2.1369393
6: -11.4304247, -8.7183628, -11.4190016, -8.7362728, -2.6683722, 2.7006388
7: 2.8122315, 4.7915897, 2.8168187, 4.7595048, -1.9472733, 1.9747710
8: -4.3897266, -2.0872946, -4.3715687, -2.0913925, -2.1748705, 2.1579001
9: -2.7653537, -1.0928988, -2.7423232, -1.0978405, -1.6675131, 1.6494243

Time for backsubstitution: 14.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 466

## Relational analysis of IS_B1_A1_A1_B1_B1_A1

### Relational analysis result of IS_B1_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4092036, upper bound: 1.4092036
time: 4.36 seconds

## Relational analysis of IS_B1_A1_A1_B1_B1_A2

### Relational analysis result of IS_B1_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4092036, upper bound: 1.4188844
time: 4.31 seconds

## BFS IS instance: IS_B1_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -7.0873675, -4.4529629, -7.1080518, -4.4468966, -2.4771276, 2.4930065
1: -7.2344985, -5.1007633, -7.2412229, -5.0976734, -2.1368251, 2.1404595
2: -6.0757961, -4.0416365, -6.0788193, -4.0143332, -1.8763785, 1.8500433
3: -6.0826688, -3.6086726, -6.0983977, -3.6049092, -2.4777596, 2.4897251
4: -6.4587421, -4.1152725, -6.4720488, -4.1066399, -2.3521023, 2.3567762
5: -6.4905066, -4.3373160, -6.4986053, -4.3324761, -2.1536446, 2.1612892
6: -11.4341784, -8.7121668, -11.4604044, -8.7104778, -2.7237005, 2.7482376
7: 2.8107338, 4.8027482, 2.7847095, 4.8119922, -2.0012584, 2.0180387
8: -4.3960209, -2.0859823, -4.4025812, -2.0695806, -2.2094622, 2.1920996
9: -2.7734270, -1.0913440, -2.7786160, -1.0785308, -1.6948962, 1.6872720

Time for backsubstitution: 14.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 466

## Relational analysis of IS_B1_A1_A1_B1_B2_A1

### Relational analysis result of IS_B1_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4188844, upper bound: 1.4092030
time: 4.66 seconds

## Relational analysis of IS_B1_A1_A1_B1_B2_A2

### Relational analysis result of IS_B1_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4188844, upper bound: 1.4188840
time: 4.52 seconds

## BFS IS instance: IS_B1_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.0785074, -4.4720697, -7.1717172, -4.3960247, -2.5785012, 2.4903514
1: -7.2102599, -5.1066265, -7.2704754, -5.0621748, -2.1173511, 2.1638489
2: -6.0455213, -4.0458918, -6.1092515, -4.0264206, -1.7546654, 1.8773668
3: -6.0556769, -3.6161091, -6.1440129, -3.5645986, -2.4910784, 2.5279038
4: -6.4499207, -4.1408119, -6.5161047, -4.0558791, -2.3940415, 2.3752928
5: -6.4842033, -4.3443871, -6.5289621, -4.3043866, -2.1727369, 2.1845751
6: -11.4190016, -8.7362728, -11.4859734, -8.6867752, -2.7322264, 2.7300568
7: 2.8168187, 4.7595048, 2.7667072, 4.8227520, -2.0059333, 1.9927976
8: -4.3715687, -2.0913925, -4.4137993, -2.0650563, -2.1831799, 2.2077122
9: -2.7423232, -1.0978405, -2.7909262, -1.0687170, -1.6736062, 1.6930857

Time for backsubstitution: 14.14 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.0716936588287354
rel_dist={7: [-1.4397835588004604, 1.43978321464756]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6178
type: A, layer: 1, pos: 6178
type: B, layer: 1, pos: 484
type: A, layer: 1, pos: 484
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6178

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2398932, upper bound: 1.2349364
time: 4.60 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2399057, upper bound: 1.2399048
time: 6.18 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 10.99 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 10.99
Output dim: 7, lower bound: -1.2398932, upper bound: 1.2349364
IS_B2, status: Status.UNKNOWN, split count: 1, time: 10.99
Output dim: 7, lower bound: -1.2399057, upper bound: 1.2399048

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -7.1142826, -4.3955135, -7.1096139, -4.3970814, -2.3932633, 2.3883133
1: -7.2771664, -5.0646043, -7.2482762, -5.0689411, -2.2082253, 2.1836720
2: -6.0994167, -4.0278893, -6.0847487, -4.0312190, -1.7708793, 1.7598615
3: -6.1377277, -3.5675666, -6.1070867, -3.5716767, -2.4312911, 2.4052806
4: -6.4849176, -4.0602245, -6.4741940, -4.0632663, -2.4216514, 2.4139695
5: -6.5198631, -4.3088722, -6.5166945, -4.3161192, -2.0998192, 2.1078200
6: -11.4788694, -8.7039680, -11.4735241, -8.7056084, -2.7101107, 2.7039394
7: 2.7643838, 4.8159389, 2.7820542, 4.8129635, -1.9849620, 1.9713237
8: -4.4040737, -2.0593462, -4.4016609, -2.0721102, -2.0576615, 2.0706146
9: -2.7894466, -1.0652866, -2.7858586, -1.0757484, -1.7136981, 1.7205720

Time for backsubstitution: 14.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 484
type: B, layer: 1, pos: 484
type: A, layer: 1, pos: 6178
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 484

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2317718, upper bound: 1.2328339
time: 4.33 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2398875, upper bound: 1.2349315
time: 4.06 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -7.1188111, -4.3942261, -7.1353807, -4.3758678, -2.4291315, 2.4528189
1: -7.3048563, -5.0606909, -7.3234901, -4.9762707, -2.3006763, 2.2627993
2: -6.1131306, -4.0248547, -6.1204958, -3.9811716, -1.8192883, 1.7910328
3: -6.1664376, -3.5639713, -6.1979318, -3.4905531, -2.5439205, 2.4898109
4: -6.4951191, -4.0573525, -6.5056715, -4.0280843, -2.4670348, 2.4483190
5: -6.5228043, -4.3014250, -6.5651617, -4.2898498, -2.1376190, 2.1688244
6: -11.4839725, -8.7024975, -11.5080719, -8.6733360, -2.7502360, 2.7453413
7: 2.7477853, 4.8194652, 2.7353382, 4.8440499, -2.0342484, 2.0188448
8: -4.4071112, -2.0474262, -4.4523430, -2.0392408, -2.0880237, 2.1271732
9: -2.7929382, -1.0555989, -2.8258357, -1.0504484, -1.7424898, 1.7702368

Time for backsubstitution: 13.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 484
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 6178
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 484

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2318192, upper bound: 1.2377887
time: 5.00 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2399000, upper bound: 1.2399001
time: 5.28 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 23.85 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 23.85
Output dim: 7, lower bound: -1.2317718, upper bound: 1.2328339
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 23.85
Output dim: 7, lower bound: -1.2398875, upper bound: 1.2349315
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 23.85
Output dim: 7, lower bound: -1.2318192, upper bound: 1.2377887
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 23.85
Output dim: 7, lower bound: -1.2399000, upper bound: 1.2399001

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -7.0919800, -4.4514866, -7.0993013, -4.4249544, -2.3652120, 2.2937164
1: -7.2634258, -5.0964966, -7.2416663, -5.0850334, -2.1783924, 2.1311004
2: -6.0904617, -4.0384083, -6.0806971, -4.0365782, -1.7395337, 1.7303727
3: -6.1133947, -3.6045573, -6.0946574, -3.5895321, -2.4572401, 2.3559043
4: -6.4694729, -4.1122980, -6.4669900, -4.0892258, -2.3802471, 2.3546920
5: -6.4935718, -4.3300700, -6.5028386, -4.3267193, -1.9910913, 2.0538425
6: -11.4395199, -8.7105713, -11.4541140, -8.7088451, -2.6443567, 2.6509380
7: 2.7931547, 4.8057294, 2.7964339, 4.8081160, -1.9429808, 1.9317496
8: -4.3984351, -2.0732393, -4.3990116, -2.0790038, -2.0271606, 2.0380137
9: -2.7769406, -1.0808952, -2.7789350, -1.0836171, -1.6933235, 1.6980398

Time for backsubstitution: 14.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 484
type: A, layer: 1, pos: 6178
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 484

## Relational analysis of IS_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2317707, upper bound: 1.2268332
time: 4.28 seconds

## Relational analysis of IS_B1_A1_B2

### Relational analysis result of IS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2317707, upper bound: 1.2328339
time: 4.68 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -7.1786280, -4.3894200, -7.1096091, -4.3970947, -2.4384689, 2.3822470
1: -7.3056154, -5.0561805, -7.2482710, -5.0689554, -2.2366600, 2.1920905
2: -6.1317353, -4.0220294, -6.0847464, -4.0312243, -1.8049994, 1.7569520
3: -6.1816111, -3.5585442, -6.1070786, -3.5716960, -2.4740858, 2.4161258
4: -6.5289927, -4.0461526, -6.4741907, -4.0632887, -2.4657040, 2.4280381
5: -6.5338225, -4.2952242, -6.5166831, -4.3161273, -2.0965099, 2.1262865
6: -11.4952784, -8.6789169, -11.4735060, -8.7056122, -2.7225323, 2.7252579
7: 2.7474482, 4.8368969, 2.7820649, 4.8129606, -1.9957023, 1.9814174
8: -4.4225488, -2.0509710, -4.4016571, -2.0721140, -2.0874062, 2.0717077
9: -2.8027220, -1.0567130, -2.7858539, -1.0757536, -1.7269684, 1.7291409

Time for backsubstitution: 14.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 6178
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 484
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2341893, upper bound: 1.2349244
time: 4.48 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2398812, upper bound: 1.2349248
time: 4.17 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -7.0964479, -4.4503403, -7.1249075, -4.4039059, -2.4027796, 2.3492670
1: -7.2911406, -5.0926909, -7.3170376, -4.9924612, -2.2743392, 2.1882622
2: -6.1041765, -4.0354786, -6.1164045, -3.9865301, -1.7862968, 1.7623377
3: -6.1421475, -3.6009612, -6.1854467, -3.5084155, -2.5500984, 2.4404263
4: -6.4796691, -4.1095400, -6.4985623, -4.0541220, -2.4255471, 2.3890224
5: -6.4963498, -4.3226337, -6.5511837, -4.3004322, -2.0202785, 2.1157856
6: -11.4446554, -8.7091522, -11.4885941, -8.6766186, -2.6853132, 2.6890798
7: 2.7766731, 4.8092537, 2.7497845, 4.8391938, -1.9934769, 1.9789877
8: -4.4016075, -2.0613527, -4.4497356, -2.0461435, -2.0575075, 2.0941100
9: -2.7802911, -1.0712228, -2.8186810, -1.0582882, -1.7220029, 1.7474582

Time for backsubstitution: 14.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 6178
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 484

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2318181, upper bound: 1.2318174
time: 4.45 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2318181, upper bound: 1.2377887
time: 4.63 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -7.1831522, -4.3880939, -7.1353755, -4.3758826, -2.4725530, 2.4465766
1: -7.3332806, -5.0522261, -7.3234854, -4.9762840, -2.3145609, 2.2712593
2: -6.1454744, -4.0189538, -6.1204939, -3.9811771, -1.8393705, 1.7881563
3: -6.2103338, -3.5549517, -6.1979227, -3.4905729, -2.5676250, 2.5006192
4: -6.5391836, -4.0432382, -6.5056686, -4.0281067, -2.5110769, 2.4624305
5: -6.5367980, -4.2877555, -6.5651503, -4.2898579, -2.1343753, 2.1805205
6: -11.5003815, -8.6774368, -11.5080557, -8.6733379, -2.7627487, 2.7666583
7: 2.7308073, 4.8404398, 2.7353508, 4.8440480, -2.0450130, 2.0285556
8: -4.4256759, -2.0390582, -4.4523401, -2.0392456, -2.1177902, 2.1282945
9: -2.8062210, -1.0470474, -2.8258319, -1.0504546, -1.7557664, 1.7787845

Time for backsubstitution: 14.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 6178
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 484
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2341926, upper bound: 1.2398936
time: 5.21 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2398937, upper bound: 1.2398937
time: 5.45 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 25.01 seconds
IS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 25.01
Output dim: 7, lower bound: -1.2317707, upper bound: 1.2268332
IS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 25.01
Output dim: 7, lower bound: -1.2317707, upper bound: 1.2328339
IS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 25.01
Output dim: 7, lower bound: -1.2341893, upper bound: 1.2349244
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 25.01
Output dim: 7, lower bound: -1.2398812, upper bound: 1.2349248
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 25.01
Output dim: 7, lower bound: -1.2318181, upper bound: 1.2318174
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 25.01
Output dim: 7, lower bound: -1.2318181, upper bound: 1.2377887
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 25.01
Output dim: 7, lower bound: -1.2341926, upper bound: 1.2398936
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 25.01
Output dim: 7, lower bound: -1.2398937, upper bound: 1.2398937

## BFS IS instance: IS_B1_A1_B1

### Backsubstitution after applying IS history:
0: -7.0919800, -4.4514866, -7.0873675, -4.4529572, -2.2749648, 2.2705314
1: -7.2634258, -5.0964966, -7.2345042, -5.1007624, -2.1474919, 2.1233685
2: -6.0904617, -4.0384083, -6.0758004, -4.0416355, -1.7315228, 1.7204516
3: -6.1133947, -3.6045573, -6.0826745, -3.6086719, -2.4370632, 2.4113119
4: -6.4694729, -4.1122980, -6.4587431, -4.1152668, -2.3542061, 2.3464451
5: -6.4935718, -4.3300700, -6.4905062, -4.3373127, -1.9744365, 1.9804935
6: -11.4395199, -8.7105713, -11.4341793, -8.7121630, -2.6260910, 2.6199460
7: 2.7931547, 4.8057294, 2.8107338, 4.8027568, -1.9335370, 1.9202335
8: -4.3984351, -2.0732393, -4.3960266, -2.0859809, -2.0156069, 2.0289538
9: -2.7769406, -1.0808952, -2.7734327, -1.0913429, -1.6855977, 1.6925375

Time for backsubstitution: 14.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6178
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6178

## Relational analysis of IS_B1_A1_B1_A1

### Relational analysis result of IS_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2268372, upper bound: 1.2268333
time: 4.58 seconds

## Relational analysis of IS_B1_A1_B1_A2

### Relational analysis result of IS_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2268393, upper bound: 1.2268330
time: 4.66 seconds

## BFS IS instance: IS_B1_A1_B2

### Backsubstitution after applying IS history:
0: -7.0919800, -4.4514866, -7.1727805, -4.3910670, -2.3926442, 2.3444521
1: -7.2634258, -5.0964966, -7.2758951, -5.0605955, -2.2028303, 2.1675520
2: -6.0904617, -4.0384083, -6.1161089, -4.0253973, -1.7473407, 1.7616756
3: -6.1133947, -3.6045573, -6.1501617, -3.5627110, -2.4853559, 2.4101861
4: -6.4694729, -4.1122980, -6.5173273, -4.0493546, -2.4201183, 2.4050293
5: -6.4935718, -4.3300700, -6.5304084, -4.3028469, -2.0099893, 2.0736256
6: -11.4395199, -8.7105713, -11.4894371, -8.6807251, -2.6721492, 2.6905303
7: 2.7931547, 4.8057294, 2.7653708, 4.8337879, -1.9660170, 1.9679055
8: -4.3984351, -2.0732393, -4.4195957, -2.0637841, -2.0413942, 2.0613015
9: -2.7769406, -1.0808952, -2.7990479, -1.0674130, -1.7095276, 1.7181528

Time for backsubstitution: 14.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6178
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6178

## Relational analysis of IS_B1_A1_B2_A1

### Relational analysis result of IS_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2268372, upper bound: 1.2328340
time: 4.37 seconds

## Relational analysis of IS_B1_A1_B2_A2

### Relational analysis result of IS_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2268372, upper bound: 1.2328336
time: 4.39 seconds

## BFS IS instance: IS_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.1742024, -4.3990984, -7.1004934, -4.4165487, -2.4136062, 2.3634257
1: -7.2934947, -5.0592585, -7.2239132, -5.0750589, -2.2184358, 2.1646547
2: -6.1166573, -4.0241442, -6.0544538, -4.0355463, -1.7776670, 1.7211113
3: -6.1682425, -3.5622740, -6.0800228, -3.5790098, -2.4455590, 2.3735487
4: -6.5247240, -4.0589952, -6.4652872, -4.0890179, -2.4357061, 2.4062920
5: -6.5306015, -4.2988262, -6.5098391, -4.3232937, -2.0692136, 2.1014550
6: -11.4877834, -8.6909361, -11.4581842, -8.7297812, -2.6898355, 2.6984634
7: 2.7505012, 4.8153543, 2.7883921, 4.7697535, -1.9453394, 1.9491105
8: -4.4103584, -2.0535822, -4.3772392, -2.0776043, -2.0648885, 2.0384665
9: -2.7869010, -1.0598133, -2.7541101, -1.0822996, -1.7046014, 1.6942968

Time for backsubstitution: 14.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 484
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6178

## Relational analysis of IS_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2292435, upper bound: 1.2349246
time: 4.60 seconds

## Relational analysis of IS_B1_A2_B1_A2

### Relational analysis result of IS_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2292435, upper bound: 1.2349269
time: 4.98 seconds

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.1786251, -4.3894291, -7.1302667, -4.3909240, -2.4425073, 2.4033384
1: -7.3056059, -5.0561819, -7.2549381, -5.0657640, -2.2398419, 2.1987562
2: -6.1317291, -4.0220318, -6.0877571, -4.0038705, -1.8115551, 1.7554188
3: -6.1816020, -3.5585468, -6.1228428, -3.5679901, -2.4783769, 2.4255145
4: -6.5289912, -4.0461617, -6.4875002, -4.0545506, -2.4744406, 2.4413385
5: -6.5338211, -4.2952275, -6.5252781, -4.3112626, -2.0958970, 2.1375062
6: -11.4952726, -8.6789207, -11.4998035, -8.7039375, -2.7226944, 2.7516465
7: 2.7474506, 4.8368793, 2.7559752, 4.8221083, -1.9989090, 1.9993105
8: -4.4225402, -2.0509720, -4.4081435, -2.0556488, -2.1009674, 2.0734816
9: -2.8027124, -1.0567147, -2.7911742, -1.0629299, -1.7397826, 1.7344595

Time for backsubstitution: 14.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6178
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 484
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6178

## Relational analysis of IS_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2349269, upper bound: 1.2349249
time: 4.74 seconds

## Relational analysis of IS_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2349269, upper bound: 1.2349269
time: 4.86 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -7.0964479, -4.4503403, -7.1124840, -4.4339428, -2.3696527, 2.3366632
1: -7.2911406, -5.0926909, -7.3096709, -5.0097485, -2.2566211, 2.1805496
2: -6.1041765, -4.0354786, -6.1114635, -3.9922206, -1.7766051, 1.7523785
3: -6.1421475, -3.6009612, -6.1724005, -3.5275595, -2.5336146, 2.4266977
4: -6.4796691, -4.1095400, -6.4900475, -4.0819707, -2.3976984, 2.3805075
5: -6.4963498, -4.3226337, -6.5364413, -4.3113866, -2.0041900, 2.0901587
6: -11.4446554, -8.7091522, -11.4677963, -8.6802998, -2.6766825, 2.6630201
7: 2.7766731, 4.8092537, 2.7652836, 4.8338270, -1.9840326, 1.9616420
8: -4.4016075, -2.0613527, -4.4467916, -2.0534730, -2.0433493, 2.0863316
9: -2.7802911, -1.0712228, -2.8109851, -1.0662770, -1.7101171, 1.7397623

Time for backsubstitution: 14.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 6178
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_B2_A1_B1_B1

### Relational analysis result of IS_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2261218, upper bound: 1.2318113
time: 4.98 seconds

## Relational analysis of IS_B2_A1_B1_B2

### Relational analysis result of IS_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2318129, upper bound: 1.2318113
time: 4.81 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.0964479, -4.4503403, -7.1985369, -4.3696084, -2.4290400, 2.3848014
1: -7.2911406, -5.0926909, -7.3509946, -4.9677210, -2.2953367, 2.2245023
2: -6.1041765, -4.0354786, -6.1518106, -3.9751611, -1.7935014, 1.7936316
3: -6.1421475, -3.6009612, -6.2404485, -3.4815874, -2.5668340, 2.4941821
4: -6.4796691, -4.1095400, -6.5487370, -4.0139771, -2.4656920, 2.4391971
5: -6.4963498, -4.3226337, -6.5790606, -4.2767553, -2.0393806, 2.1360159
6: -11.4446554, -8.7091522, -11.5245428, -8.6484566, -2.7131338, 2.7293572
7: 2.7766731, 4.8092537, 2.7187972, 4.8648977, -2.0102611, 2.0151126
8: -4.4016075, -2.0613527, -4.4705148, -2.0310116, -2.0718956, 2.1134088
9: -2.7802911, -1.0712228, -2.8389223, -1.0420437, -1.7379832, 1.7676995

Time for backsubstitution: 14.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 6178
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_B2_A1_B2_B1

### Relational analysis result of IS_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2261218, upper bound: 1.2377827
time: 4.83 seconds

## Relational analysis of IS_B2_A1_B2_B2

### Relational analysis result of IS_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2318129, upper bound: 1.2377826
time: 4.84 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.1786871, -4.3977818, -7.1260443, -4.3953686, -2.4476993, 2.4250033
1: -7.3211741, -5.0552850, -7.2993312, -4.9826541, -2.2897654, 2.2440462
2: -6.1303959, -4.0210938, -6.0901976, -3.9856801, -1.8121471, 1.7523193
3: -6.1970406, -3.5586786, -6.1717005, -3.4978857, -2.5380411, 2.4591894
4: -6.5349183, -4.0560880, -6.4969859, -4.0540257, -2.4808927, 2.4408979
5: -6.5335684, -4.2913275, -6.5582914, -4.2969856, -2.1068826, 2.1543717
6: -11.4928303, -8.6894646, -11.4923830, -8.6975422, -2.7300835, 2.7394505
7: 2.7338581, 4.8188839, 2.7415943, 4.8004966, -1.9944587, 1.9964314
8: -4.4133835, -2.0416651, -4.4275427, -2.0446568, -2.0952396, 2.0944183
9: -2.7903914, -1.0501462, -2.7938061, -1.0569943, -1.7333971, 1.7436599

Time for backsubstitution: 14.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 484
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 466

## Relational analysis of IS_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2341926, upper bound: 1.2341922
time: 5.07 seconds

## Relational analysis of IS_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2341926, upper bound: 1.2398935
time: 5.39 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.1831503, -4.3881054, -7.1559687, -4.3696289, -2.4769568, 2.4542754
1: -7.3332686, -5.0522280, -7.3301811, -4.9731483, -2.3123941, 2.2779531
2: -6.1454673, -4.0189543, -6.1234560, -3.9539576, -1.8461359, 1.7865882
3: -6.2103262, -3.5549548, -6.2141099, -3.4869003, -2.5720291, 2.5112164
4: -6.5391827, -4.0432482, -6.5191507, -4.0193763, -2.5198064, 2.4759026
5: -6.5367956, -4.2877569, -6.5737586, -4.2846513, -2.1338563, 2.1868069
6: -11.5003777, -8.6774454, -11.5341482, -8.6715832, -2.7630062, 2.7929840
7: 2.7308090, 4.8404226, 2.7091260, 4.8535309, -2.0485907, 2.0463750
8: -4.4256673, -2.0390592, -4.4589634, -2.0227432, -2.1253476, 2.1304450
9: -2.8062108, -1.0470488, -2.8312187, -1.0376308, -1.7685800, 1.7841699

Time for backsubstitution: 14.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 484
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_B2_A2_B2_A1

### Relational analysis result of IS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2398870, upper bound: 1.2356605
time: 5.02 seconds

## Relational analysis of IS_B2_A2_B2_A2

### Relational analysis result of IS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2398869, upper bound: 1.2398871
time: 4.56 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 24.10 seconds
IS_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 24.10
Output dim: 7, lower bound: -1.2268372, upper bound: 1.2268333
IS_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 24.10
Output dim: 7, lower bound: -1.2268393, upper bound: 1.2268330
IS_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 24.10
Output dim: 7, lower bound: -1.2268372, upper bound: 1.2328340
IS_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 24.10
Output dim: 7, lower bound: -1.2268372, upper bound: 1.2328336
IS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 24.10
Output dim: 7, lower bound: -1.2292435, upper bound: 1.2349246
IS_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 24.10
Output dim: 7, lower bound: -1.2292435, upper bound: 1.2349269
IS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 24.10
Output dim: 7, lower bound: -1.2349269, upper bound: 1.2349249
IS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 24.10
Output dim: 7, lower bound: -1.2349269, upper bound: 1.2349269
IS_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 24.10
Output dim: 7, lower bound: -1.2261218, upper bound: 1.2318113
IS_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 24.10
Output dim: 7, lower bound: -1.2318129, upper bound: 1.2318113
IS_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 24.10
Output dim: 7, lower bound: -1.2261218, upper bound: 1.2377827
IS_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 24.10
Output dim: 7, lower bound: -1.2318129, upper bound: 1.2377826
IS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 24.10
Output dim: 7, lower bound: -1.2341926, upper bound: 1.2341922
IS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 24.10
Output dim: 7, lower bound: -1.2341926, upper bound: 1.2398935
IS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 24.10
Output dim: 7, lower bound: -1.2398870, upper bound: 1.2356605
IS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 24.10
Output dim: 7, lower bound: -1.2398869, upper bound: 1.2398871

## BFS IS instance: IS_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7.0873675, -4.4529572, -7.0873675, -4.4529572, -2.2582755, 2.2582757
1: -7.2345042, -5.1007624, -7.2345042, -5.1007624, -2.1190104, 2.1190104
2: -6.0758004, -4.0416355, -6.0758004, -4.0416355, -1.7171929, 1.7171931
3: -6.0826745, -3.6086719, -6.0826745, -3.6086719, -2.4066782, 2.4066784
4: -6.4587431, -4.1152668, -6.4587431, -4.1152668, -2.3434763, 2.3434763
5: -6.4905062, -4.3373127, -6.4905062, -4.3373127, -1.9716094, 1.9716096
6: -11.4341793, -8.7121630, -11.4341793, -8.7121630, -2.6181545, 2.6181545
7: 2.8107338, 4.8027568, 2.8107338, 4.8027568, -1.9169216, 1.9169214
8: -4.3960266, -2.0859809, -4.3960266, -2.0859809, -2.0129395, 2.0129397
9: -2.7734327, -1.0913429, -2.7734327, -1.0913429, -1.6820898, 1.6820898

Time for backsubstitution: 14.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 466

## Relational analysis of IS_B1_A1_B1_A1_A1

### Relational analysis result of IS_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2268306, upper bound: 1.2211372
time: 4.47 seconds

## Relational analysis of IS_B1_A1_B1_A1_A2

### Relational analysis result of IS_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2268306, upper bound: 1.2268277
time: 4.32 seconds

## BFS IS instance: IS_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.1115007, -4.4356828, -7.0873675, -4.4529572, -2.2818818, 2.2749760
1: -7.3082476, -5.0106921, -7.2345042, -5.1007624, -2.1792333, 2.1582732
2: -6.1114635, -3.9932475, -6.0758004, -4.0416355, -1.7529819, 1.7505212
3: -6.1707859, -3.5283008, -6.0826745, -3.6086719, -2.4965878, 2.4723938
4: -6.4877615, -4.0830927, -6.4587431, -4.1152668, -2.3724947, 2.3756504
5: -6.5349960, -4.3130980, -6.4905062, -4.3373127, -2.0191245, 1.9950206
6: -11.4671421, -8.6813564, -11.4341793, -8.7121630, -2.6495275, 2.6491299
7: 2.7663136, 4.8331795, 2.8107338, 4.8027568, -1.9618607, 1.9511406
8: -4.4456615, -2.0537205, -4.3960266, -2.0859809, -2.0544052, 2.0445061
9: -2.8090382, -1.0672218, -2.7734327, -1.0913429, -1.7176952, 1.7062110

Time for backsubstitution: 14.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_B1_A1_B1_A2_B1

### Relational analysis result of IS_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2211400, upper bound: 1.2268277
time: 4.40 seconds

## Relational analysis of IS_B1_A1_B1_A2_B2

### Relational analysis result of IS_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2268304, upper bound: 1.2268273
time: 4.63 seconds

## BFS IS instance: IS_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.0873675, -4.4529572, -7.1727805, -4.3910670, -2.3778183, 2.3368306
1: -7.2345042, -5.1007624, -7.2758951, -5.0605955, -2.1739087, 2.1635442
2: -6.0758004, -4.0416355, -6.1161089, -4.0253973, -1.7328877, 1.7584174
3: -6.0826745, -3.6086719, -6.1501617, -3.5627110, -2.4549708, 2.4060256
4: -6.4587431, -4.1152668, -6.5173273, -4.0493546, -2.4093885, 2.4020605
5: -6.4905062, -4.3373127, -6.5304084, -4.3028469, -2.0069537, 2.0617270
6: -11.4341793, -8.7121630, -11.4894371, -8.6807251, -2.6630683, 2.6887422
7: 2.8107338, 4.8027568, 2.7653708, 4.8337879, -1.9494016, 1.9649663
8: -4.3960266, -2.0859809, -4.4195957, -2.0637841, -2.0387406, 2.0452876
9: -2.7734327, -1.0913429, -2.7990479, -1.0674130, -1.7060198, 1.7077050

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 466

## Relational analysis of IS_B1_A1_B2_A1_A1

### Relational analysis result of IS_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2268305, upper bound: 1.2271457
time: 4.30 seconds

## Relational analysis of IS_B1_A1_B2_A1_A2

### Relational analysis result of IS_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2268305, upper bound: 1.2328266
time: 4.46 seconds

## BFS IS instance: IS_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.1115007, -4.4356828, -7.1727805, -4.3910670, -2.3962181, 2.3493357
1: -7.3082476, -5.0106921, -7.2758951, -5.0605955, -2.2476521, 2.1826551
2: -6.1114635, -3.9932475, -6.1161089, -4.0253973, -1.7689443, 1.7867297
3: -6.1707859, -3.5283008, -6.1501617, -3.5627110, -2.5448804, 2.4691157
4: -6.4877615, -4.0830927, -6.5173273, -4.0493546, -2.4384069, 2.4342346
5: -6.5349960, -4.3130980, -6.5304084, -4.3028469, -2.0504723, 2.0856371
6: -11.4671421, -8.6813564, -11.4894371, -8.6807251, -2.6941218, 2.7197323
7: 2.7663136, 4.8331795, 2.7653708, 4.8337879, -1.9933934, 1.9936914
8: -4.4456615, -2.0537205, -4.4195957, -2.0637841, -2.0762644, 2.0768540
9: -2.8090382, -1.0672218, -2.7990479, -1.0674130, -1.7416252, 1.7318262

Time for backsubstitution: 14.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_B1_A1_B2_A2_B1

### Relational analysis result of IS_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2211398, upper bound: 1.2328273
time: 4.49 seconds

## Relational analysis of IS_B1_A1_B2_A2_B2

### Relational analysis result of IS_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2268303, upper bound: 1.2328270
time: 4.52 seconds

## BFS IS instance: IS_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -7.1695762, -4.4006906, -7.1004934, -4.4165487, -2.4002466, 2.3514371
1: -7.2646260, -5.0636544, -7.2239132, -5.0750589, -2.1895671, 2.1602588
2: -6.1019726, -4.0274525, -6.0544538, -4.0355463, -1.7659473, 1.7176540
3: -6.1375046, -3.5663834, -6.0800228, -3.5790098, -2.4153013, 2.3694124
4: -6.5140057, -4.0620666, -6.4652872, -4.0890179, -2.4249878, 2.4032207
5: -6.5274053, -4.3061290, -6.5098391, -4.3232937, -2.0661194, 2.0904207
6: -11.4823437, -8.6925764, -11.4581842, -8.7297812, -2.6815567, 2.6966891
7: 2.7682042, 4.8123765, 2.7883921, 4.7697535, -1.9287512, 1.9461627
8: -4.4079251, -2.0663404, -4.3772392, -2.0776043, -2.0622287, 2.0228207
9: -2.7833128, -1.0702506, -2.7541101, -1.0822996, -1.7010132, 1.6838595

Time for backsubstitution: 14.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 466

## Relational analysis of IS_B1_A2_B1_A1_A1

### Relational analysis result of IS_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2292435, upper bound: 1.2292567
time: 4.77 seconds

## Relational analysis of IS_B1_A2_B1_A1_A2

### Relational analysis result of IS_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2292435, upper bound: 1.2349246
time: 4.60 seconds

## BFS IS instance: IS_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.1951680, -4.3798075, -7.1004934, -4.4165487, -2.4205718, 2.3711753
1: -7.3397517, -4.9709368, -7.2239132, -5.0750589, -2.2556129, 2.2212963
2: -6.1376915, -3.9774675, -6.0544538, -4.0355463, -1.7907286, 1.7516383
3: -6.2281485, -3.4859986, -6.0800228, -3.5790098, -2.5070152, 2.4518454
4: -6.5446711, -4.0268178, -6.4652872, -4.0890179, -2.4556532, 2.4384694
5: -6.5760355, -4.2806587, -6.5098391, -4.3232937, -2.1171610, 2.1157179
6: -11.5171814, -8.6603355, -11.4581842, -8.7297812, -2.7151070, 2.7289748
7: 2.7217102, 4.8426762, 2.7883921, 4.7697535, -1.9754374, 1.9733341
8: -4.4574499, -2.0335340, -4.3772392, -2.0776043, -2.0841269, 2.0549769
9: -2.8224404, -1.0449395, -2.7541101, -1.0822996, -1.7401408, 1.7091706

Time for backsubstitution: 14.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 484
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 466

## Relational analysis of IS_B1_A2_B1_A2_A1

### Relational analysis result of IS_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2292435, upper bound: 1.2292590
time: 4.89 seconds

## Relational analysis of IS_B1_A2_B1_A2_A2

### Relational analysis result of IS_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2292435, upper bound: 1.2349269
time: 4.83 seconds

## BFS IS instance: IS_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -7.1739597, -4.3910313, -7.1302667, -4.3909240, -2.4290981, 2.3913598
1: -7.2767477, -5.0605578, -7.2549381, -5.0657640, -2.2109838, 2.1943803
2: -6.1170478, -4.0253420, -6.0877571, -4.0038705, -1.7998297, 1.7519331
3: -6.1509418, -3.5626533, -6.1228428, -3.5679901, -2.4482369, 2.4213803
4: -6.5182781, -4.0492411, -6.4875002, -4.0545506, -2.4637275, 2.4382591
5: -6.5306139, -4.3025007, -6.5252781, -4.3112626, -2.0927827, 2.1265159
6: -11.4897671, -8.6805687, -11.4998035, -8.7039375, -2.7144489, 2.7498679
7: 2.7651467, 4.8338847, 2.7559752, 4.8221083, -1.9823112, 1.9961381
8: -4.4200945, -2.0637302, -4.4081435, -2.0556488, -2.0986562, 2.0578294
9: -2.7991116, -1.0671510, -2.7911742, -1.0629299, -1.7361817, 1.7240232

Time for backsubstitution: 14.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 484
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5746

## Relational analysis of IS_B1_A2_B2_A1_B1

### Relational analysis result of IS_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2307376, upper bound: 1.2349191
time: 4.93 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2

### Relational analysis result of IS_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2349204, upper bound: 1.2349197
time: 4.38 seconds

## BFS IS instance: IS_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.1996574, -4.3701382, -7.1302667, -4.3909240, -2.4494309, 2.4111104
1: -7.3517747, -4.9677100, -7.2549381, -5.0657640, -2.2779932, 2.2533779
2: -6.1527476, -3.9752960, -6.0877571, -4.0038705, -1.8245485, 1.7861624
3: -6.2411776, -3.4822698, -6.1228428, -3.5679901, -2.5397029, 2.5037906
4: -6.5488410, -4.0139098, -6.4875002, -4.0545506, -2.4942904, 2.4735904
5: -6.5792484, -4.2770443, -6.5252781, -4.3112626, -2.1438611, 2.1516354
6: -11.5247231, -8.6483107, -11.4998035, -8.7039375, -2.7480240, 2.7821589
7: 2.7186995, 4.8643475, 2.7559752, 4.8221083, -2.0289364, 2.0170867
8: -4.4698825, -2.0309606, -4.4081435, -2.0556488, -2.1133590, 2.0899663
9: -2.8383694, -1.0418425, -2.7911742, -1.0629299, -1.7754395, 1.7493317

Time for backsubstitution: 14.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 484
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5746

## Relational analysis of IS_B1_A2_B2_A2_B1

### Relational analysis result of IS_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2307376, upper bound: 1.2349174
time: 4.58 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2

### Relational analysis result of IS_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2349204, upper bound: 1.2349185
time: 4.89 seconds

## BFS IS instance: IS_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -7.0920687, -4.4598622, -7.1033707, -4.4534731, -2.3447518, 2.3147626
1: -7.2790647, -5.0956106, -7.2855148, -5.0161743, -2.2320077, 2.1510007
2: -6.0891161, -4.0375595, -6.0811687, -3.9967079, -1.7494307, 1.7166278
3: -6.1289272, -3.6047218, -6.1459827, -3.5349965, -2.5033412, 2.3851295
4: -6.4754457, -4.1222868, -6.4813428, -4.1080570, -2.3673887, 2.3590560
5: -6.4932461, -4.3261518, -6.5296783, -4.3185248, -1.9818788, 2.0649180
6: -11.4373436, -8.7211494, -11.4522171, -8.7045059, -2.6441116, 2.6362615
7: 2.7796090, 4.7876596, 2.7714953, 4.7902269, -1.9334743, 1.9294748
8: -4.3892498, -2.0639391, -4.4218078, -2.0588770, -2.0208807, 2.0526326
9: -2.7647610, -1.0742966, -2.7790065, -1.0728186, -1.6846404, 1.7047099

Time for backsubstitution: 14.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 6178
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 466

## Relational analysis of IS_B2_A1_B1_B1_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2261218, upper bound: 1.2261212
time: 4.97 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2

### Relational analysis result of IS_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2261218, upper bound: 1.2318124
time: 4.92 seconds

## BFS IS instance: IS_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -7.0964451, -4.4503498, -7.1331038, -4.4276876, -2.3740282, 2.3444023
1: -7.2911320, -5.0926938, -7.3164535, -5.0066042, -2.2547998, 2.1813285
2: -6.1041689, -4.0354805, -6.1144357, -3.9650075, -1.7834032, 1.7505846
3: -6.1421390, -3.6009622, -6.1885905, -3.5238309, -2.5378251, 2.4372842
4: -6.4796677, -4.1095495, -6.5035110, -4.0732484, -2.4064193, 2.3939614
5: -6.4963489, -4.3226357, -6.5446815, -4.3061771, -2.0046873, 2.1007395
6: -11.4446507, -8.7091589, -11.4938602, -8.6785336, -2.6767092, 2.6896935
7: 2.7766743, 4.8092351, 2.7390676, 4.8433981, -1.9867623, 1.9868898
8: -4.4015989, -2.0613546, -4.4533453, -2.0370193, -2.0617962, 2.0885527
9: -2.7802811, -1.0712249, -2.8163342, -1.0534602, -1.7237439, 1.7451093

Time for backsubstitution: 14.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 6178
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_B2_A1_B1_B2_A1

### Relational analysis result of IS_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2318061, upper bound: 1.2275139
time: 4.22 seconds

## Relational analysis of IS_B2_A1_B1_B2_A2

### Relational analysis result of IS_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2318061, upper bound: 1.2318055
time: 4.66 seconds

## BFS IS instance: IS_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -7.0920687, -4.4598622, -7.1892939, -4.3891010, -2.4041719, 2.3628867
1: -7.2790647, -5.0956106, -7.3268423, -4.9741406, -2.2706871, 2.1949062
2: -6.0891161, -4.0375595, -6.1215224, -3.9796584, -1.7662716, 1.7578583
3: -6.1289272, -3.6047218, -6.2142162, -3.4889579, -2.5365753, 2.4522853
4: -6.4754457, -4.1222868, -6.5400467, -4.0399199, -2.4355259, 2.4177599
5: -6.4932461, -4.3261518, -6.5724878, -4.2839446, -2.0170145, 2.1095474
6: -11.4373436, -8.7211494, -11.5089779, -8.6726723, -2.6805544, 2.7027397
7: 2.7796090, 4.7876596, 2.7250414, 4.8213758, -1.9599857, 1.9829452
8: -4.3892498, -2.0639391, -4.4456110, -2.0363913, -2.0494099, 2.0797553
9: -2.7647610, -1.0742966, -2.8069649, -1.0485929, -1.7125072, 1.7326683

Time for backsubstitution: 14.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 6178
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 466

## Relational analysis of IS_B2_A1_B2_B1_A1

### Relational analysis result of IS_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2261218, upper bound: 1.2320885
time: 4.56 seconds

## Relational analysis of IS_B2_A1_B2_B1_A2

### Relational analysis result of IS_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2261218, upper bound: 1.2377823
time: 4.77 seconds

## BFS IS instance: IS_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -7.0964451, -4.4503498, -7.2189589, -4.3632689, -2.4333928, 2.3925202
1: -7.2911320, -5.0926938, -7.3577089, -4.9645276, -2.2935021, 2.2252564
2: -6.1041689, -4.0354805, -6.1547356, -3.9479373, -1.8002491, 1.7918568
3: -6.1421390, -3.6009622, -6.2565093, -3.4777970, -2.5710125, 2.5043640
4: -6.4796677, -4.1095495, -6.5622144, -4.0050769, -2.4745908, 2.4526649
5: -6.4963489, -4.3226357, -6.5877337, -4.2714663, -2.0399327, 2.1420186
6: -11.4446507, -8.7091589, -11.5506220, -8.6467037, -2.7131491, 2.7561536
7: 2.7766743, 4.8092351, 2.6925609, 4.8743682, -2.0131466, 2.0297821
8: -4.4015989, -2.0613546, -4.4770923, -2.0144968, -2.0903611, 2.1155844
9: -2.7802811, -1.0712249, -2.8443017, -1.0292265, -1.7510545, 1.7730768

Time for backsubstitution: 14.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 6178
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_B2_A1_B2_B2_A1

### Relational analysis result of IS_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2318060, upper bound: 1.2335371
time: 4.48 seconds

## Relational analysis of IS_B2_A1_B2_B2_A2

### Relational analysis result of IS_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2318061, upper bound: 1.2377757
time: 4.94 seconds

## BFS IS instance: IS_B2_A2_B1_A1

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

Time for backsubstitution: 14.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 6178
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 484
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_B2_A2_B1_A1_A1

### Relational analysis result of IS_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2341860, upper bound: 1.2299476
time: 4.58 seconds

## Relational analysis of IS_B2_A2_B1_A1_A2

### Relational analysis result of IS_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2341859, upper bound: 1.2342014
time: 4.50 seconds

## BFS IS instance: IS_B2_A2_B1_A2

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

Time for backsubstitution: 14.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_B2_A2_B1_A2_A1

### Relational analysis result of IS_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2341860, upper bound: 1.2356587
time: 4.50 seconds

## Relational analysis of IS_B2_A2_B1_A2_A2

### Relational analysis result of IS_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2341859, upper bound: 1.2398870
time: 4.30 seconds

## BFS IS instance: IS_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -7.1704140, -4.3928890, -7.1520109, -4.3708506, -2.4571996, 2.4404948
1: -7.2775474, -5.0866261, -7.3053703, -4.9784336, -2.2536583, 2.2187443
2: -6.1195068, -4.0467558, -6.1192226, -3.9663048, -1.8008790, 1.7536309
3: -6.1744347, -3.5654058, -6.2064576, -3.4890230, -2.5270853, 2.4805162
4: -6.4864979, -4.0795722, -6.4954567, -4.0247288, -2.4617691, 2.4158845
5: -6.5125408, -4.3016348, -6.5631490, -4.2878885, -2.1055613, 2.1564307
6: -11.4658966, -8.6952496, -11.5196266, -8.6744890, -2.7252860, 2.7566853
7: 2.7682805, 4.8226042, 2.7150457, 4.8488312, -2.0048223, 2.0176897
8: -4.3959689, -2.0805330, -4.4524131, -2.0416479, -2.0665898, 2.0806375
9: -2.7594488, -1.1040366, -2.8243532, -1.0641783, -1.6952704, 1.7203166

Time for backsubstitution: 14.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6178
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 484
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6178

## Relational analysis of IS_B2_A2_B2_A1_A1

### Relational analysis result of IS_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2349176, upper bound: 1.2356608
time: 4.62 seconds

## Relational analysis of IS_B2_A2_B2_A1_A2

### Relational analysis result of IS_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2349176, upper bound: 1.2307352
time: 5.01 seconds

## BFS IS instance: IS_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.1831465, -4.3881083, -7.1559677, -4.3696280, -2.4793577, 2.4490962
1: -7.3332462, -5.0522299, -7.3301721, -4.9731503, -2.2791312, 2.2777443
2: -6.1454639, -4.0189643, -6.1234550, -3.9539602, -1.8361247, 1.7728074
3: -6.2103209, -3.5549552, -6.2141085, -3.4868996, -2.5622191, 2.5083699
4: -6.5391598, -4.0432515, -6.5191445, -4.0193782, -2.5197816, 2.4758930
5: -6.5367889, -4.2877588, -6.5737562, -4.2846518, -2.1249266, 2.1827278
6: -11.5003672, -8.6774454, -11.5341454, -8.6715851, -2.7542238, 2.7928519
7: 2.7308130, 4.8404198, 2.7091281, 4.8535304, -2.0461092, 2.0395997
8: -4.4256630, -2.0390692, -4.4589620, -2.0227451, -2.1170254, 2.1143484
9: -2.8062062, -1.0470613, -2.8312168, -1.0376345, -1.7685717, 1.7841555

Time for backsubstitution: 14.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 5746

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5805

## Relational analysis of IS_B2_A2_B2_A2_B1

### Relational analysis result of IS_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2298609, upper bound: 1.2355272
time: 5.12 seconds

## Relational analysis of IS_B2_A2_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2398842, upper bound: 1.2398848
time: 5.01 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 24.50 seconds
IS_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 24.50
Output dim: 7, lower bound: -1.2268306, upper bound: 1.2211372
IS_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 24.50
Output dim: 7, lower bound: -1.2268306, upper bound: 1.2268277
IS_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 24.50
Output dim: 7, lower bound: -1.2211400, upper bound: 1.2268277
IS_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 24.50
Output dim: 7, lower bound: -1.2268304, upper bound: 1.2268273
IS_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 24.50
Output dim: 7, lower bound: -1.2268305, upper bound: 1.2271457
IS_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 24.50
Output dim: 7, lower bound: -1.2268305, upper bound: 1.2328266
IS_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 24.50
Output dim: 7, lower bound: -1.2211398, upper bound: 1.2328273
IS_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 24.50
Output dim: 7, lower bound: -1.2268303, upper bound: 1.2328270
IS_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 24.50
Output dim: 7, lower bound: -1.2292435, upper bound: 1.2292567
IS_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 24.50
Output dim: 7, lower bound: -1.2292435, upper bound: 1.2349246
IS_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 24.50
Output dim: 7, lower bound: -1.2292435, upper bound: 1.2292590
IS_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 24.50
Output dim: 7, lower bound: -1.2292435, upper bound: 1.2349269
IS_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 24.50
Output dim: 7, lower bound: -1.2307376, upper bound: 1.2349191
IS_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 24.50
Output dim: 7, lower bound: -1.2349204, upper bound: 1.2349197
IS_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 24.50
Output dim: 7, lower bound: -1.2307376, upper bound: 1.2349174
IS_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 24.50
Output dim: 7, lower bound: -1.2349204, upper bound: 1.2349185
IS_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.50
Output dim: 7, lower bound: -1.2261218, upper bound: 1.2261212
IS_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.50
Output dim: 7, lower bound: -1.2261218, upper bound: 1.2318124
IS_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.50
Output dim: 7, lower bound: -1.2318061, upper bound: 1.2275139
IS_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.50
Output dim: 7, lower bound: -1.2318061, upper bound: 1.2318055
IS_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.50
Output dim: 7, lower bound: -1.2261218, upper bound: 1.2320885
IS_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.50
Output dim: 7, lower bound: -1.2261218, upper bound: 1.2377823
IS_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.50
Output dim: 7, lower bound: -1.2318060, upper bound: 1.2335371
IS_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.50
Output dim: 7, lower bound: -1.2318061, upper bound: 1.2377757
IS_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 24.50
Output dim: 7, lower bound: -1.2341860, upper bound: 1.2299476
IS_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 24.50
Output dim: 7, lower bound: -1.2341859, upper bound: 1.2342014
IS_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 24.50
Output dim: 7, lower bound: -1.2341860, upper bound: 1.2356587
IS_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 24.50
Output dim: 7, lower bound: -1.2341859, upper bound: 1.2398870
IS_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 24.50
Output dim: 7, lower bound: -1.2349176, upper bound: 1.2356608
IS_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 24.50
Output dim: 7, lower bound: -1.2349176, upper bound: 1.2307352
IS_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 24.50
Output dim: 7, lower bound: -1.2298609, upper bound: 1.2355272
IS_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 24.50
Output dim: 7, lower bound: -1.2398842, upper bound: 1.2398848

## BFS IS instance: IS_B1_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -7.0785074, -4.4720697, -7.0830669, -4.4624624, -2.2387762, 2.2140532
1: -7.2102599, -5.1066265, -7.2224002, -5.1037202, -1.8958497, 2.0960655
2: -6.0455213, -4.0458918, -6.0607362, -4.0436912, -1.6033032, 1.6945560
3: -6.0556769, -3.6161091, -6.0692978, -3.6124370, -2.3586202, 2.3772225
4: -6.4499207, -4.1408119, -6.4545097, -4.1279960, -2.3219247, 2.3136978
5: -6.4842033, -4.3443871, -6.4874263, -4.3408937, -1.9511967, 1.9496284
6: -11.4190016, -8.7362728, -11.4268293, -8.7241507, -2.5909247, 2.4942188
7: 2.8168187, 4.7595048, 2.8136716, 4.7811975, -1.8856978, 1.8679881
8: -4.3715687, -2.0913925, -4.3838568, -2.0885715, -1.9790540, 1.9905045
9: -2.7423232, -1.0978405, -2.7578826, -1.0944183, -1.6479049, 1.6600420

Time for backsubstitution: 14.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_B1_A1_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2211400, upper bound: 1.2211400
time: 4.14 seconds

## Relational analysis of IS_B1_A1_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2211400, upper bound: 1.2211397
time: 5.21 seconds

## BFS IS instance: IS_B1_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -7.1080518, -4.4468966, -7.0873661, -4.4529672, -2.2791886, 2.2630749
1: -7.2412229, -5.0976734, -7.2344947, -5.1007643, -2.1194220, 2.1164153
2: -6.0788193, -4.0143332, -6.0757937, -4.0416369, -1.7154210, 1.7412336
3: -6.0983977, -3.6049092, -6.0826664, -3.6086740, -2.4154410, 2.4107633
4: -6.4720488, -4.1066399, -6.4587402, -4.1152768, -2.3567719, 2.3521004
5: -6.4986053, -4.3324761, -6.4905047, -4.3373165, -1.9827478, 1.9717655
6: -11.4604044, -8.7104778, -11.4341774, -8.7121696, -2.6444883, 2.6181307
7: 2.7847095, 4.8119922, 2.8107345, 4.8027415, -1.9421296, 1.9192057
8: -4.4025812, -2.0695806, -4.3960166, -2.0859804, -2.0147867, 2.0312378
9: -2.7786160, -1.0785308, -2.7734232, -1.0913451, -1.6872709, 1.6948924

Time for backsubstitution: 14.14 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.0076169967651367
rel_dist={7: [-1.2399111761101596, 1.2399105900004943]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6178
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 484
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6178

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1637599, upper bound: 1.1600636
time: 6.47 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1637600, upper bound: 1.1637592
time: 6.53 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 13.18 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 13.18
Output dim: 7, lower bound: -1.1637599, upper bound: 1.1600636
IS_B2, status: Status.UNKNOWN, split count: 1, time: 13.18
Output dim: 7, lower bound: -1.1637600, upper bound: 1.1637592

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -7.1132598, -4.3958220, -7.1096139, -4.3970814, -2.3218584, 2.3178821
1: -7.2710686, -5.0654883, -7.2482762, -5.0689411, -2.1841297, 2.1650803
2: -6.0964041, -4.0285683, -6.0847487, -4.0312190, -1.7247624, 1.7159967
3: -6.1314068, -3.5683851, -6.1070867, -3.5716767, -2.3664961, 2.3458574
4: -6.4826851, -4.0608730, -6.4741940, -4.0632663, -2.4194188, 2.4133210
5: -6.5192018, -4.3104753, -6.5166945, -4.3161192, -2.0387106, 2.0450480
6: -11.4777546, -8.7043009, -11.4735241, -8.7056084, -2.6552715, 2.6503944
7: 2.7680235, 4.8151183, 2.7820542, 4.8129635, -1.9479532, 1.9370029
8: -4.4033613, -2.0619678, -4.4016609, -2.0721102, -1.9978480, 2.0084169
9: -2.7886424, -1.0674250, -2.7858586, -1.0757484, -1.7128940, 1.7184336

Time for backsubstitution: 14.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 484
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 484

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1565761, upper bound: 1.1574176
time: 4.70 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1637554, upper bound: 1.1600597
time: 6.96 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -7.1188102, -4.3942261, -7.1353788, -4.3758936, -2.3612013, 2.3808951
1: -7.3048563, -5.0606909, -7.3234887, -4.9762707, -2.2494712, 2.2175386
2: -6.1131282, -4.0248547, -6.1204948, -3.9811788, -1.7752647, 1.7466815
3: -6.1664319, -3.5639722, -6.1979294, -3.4905908, -2.4841251, 2.4295602
4: -6.4951172, -4.0573540, -6.5056715, -4.0280871, -2.4670300, 2.4483175
5: -6.5228028, -4.3014259, -6.5651598, -4.2898593, -2.0761547, 2.1083436
6: -11.4839725, -8.7024984, -11.5080643, -8.6733360, -2.6970301, 2.6917558
7: 2.7477880, 4.8194661, 2.7353404, 4.8440175, -2.0006218, 1.9847412
8: -4.4071083, -2.0474300, -4.4522853, -2.0392408, -2.0265846, 2.0668297
9: -2.7929373, -1.0556004, -2.8258042, -1.0504507, -1.7424866, 1.7702038

Time for backsubstitution: 14.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 484
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 6178

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 484

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1565761, upper bound: 1.1611157
time: 4.54 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1637554, upper bound: 1.1637553
time: 6.09 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 25.08 seconds
IS_B1_A1, status: Status.VERIFIED, split count: 2, time: 25.08
Output dim: 7, lower bound: -1.1565761, upper bound: 1.1574176
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 25.08
Output dim: 7, lower bound: -1.1637554, upper bound: 1.1600597
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 25.08
Output dim: 7, lower bound: -1.1565761, upper bound: 1.1611157
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 25.08
Output dim: 7, lower bound: -1.1637554, upper bound: 1.1637553

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -7.1776047, -4.3897362, -7.1096067, -4.3970957, -2.3656344, 2.3099828
1: -7.2995253, -5.0570726, -7.2482700, -5.0689578, -2.2055511, 2.1720276
2: -6.1287184, -4.0227075, -6.0847459, -4.0312247, -1.7573547, 1.7130792
3: -6.1752882, -3.5593626, -6.1070762, -3.5717006, -2.4093013, 2.3566415
4: -6.5267634, -4.0468073, -6.4741883, -4.0632920, -2.4634714, 2.4273810
5: -6.5331535, -4.2968321, -6.5166807, -4.3161302, -2.0353854, 2.0620232
6: -11.4941273, -8.6792517, -11.4735050, -8.7056103, -2.6670732, 2.6717105
7: 2.7510953, 4.8360710, 2.7820685, 4.8129597, -1.9572821, 1.9470892
8: -4.4218292, -2.0535917, -4.4016562, -2.0721149, -2.0267720, 2.0094993
9: -2.8019161, -1.0588461, -2.7858534, -1.0757546, -1.7261615, 1.7270073

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 6178
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 484

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1594453, upper bound: 1.1599793
time: 4.70 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1637500, upper bound: 1.1600546
time: 4.44 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -7.0964484, -4.4503407, -7.1224771, -4.4100661, -2.3266978, 2.2731407
1: -7.2911396, -5.0926914, -7.3155799, -4.9959993, -2.2192299, 2.1357551
2: -6.1041722, -4.0354795, -6.1154475, -3.9877024, -1.7400665, 1.7150435
3: -6.1421442, -3.6009614, -6.1828365, -3.5123637, -2.4881115, 2.3774450
4: -6.4796696, -4.1095409, -6.4968967, -4.0598164, -2.4050989, 2.3873558
5: -6.4963503, -4.3226337, -6.5481510, -4.3027196, -1.9559257, 2.0500455
6: -11.4446545, -8.7091522, -11.4843311, -8.6773548, -2.6289368, 2.6301336
7: 2.7766745, 4.8092527, 2.7529509, 4.8380775, -1.9574387, 1.9413185
8: -4.4016075, -2.0613551, -4.4490910, -2.0476456, -1.9931760, 2.0320079
9: -2.7802906, -1.0712237, -2.8170691, -1.0599616, -1.6827464, 1.7458453

Time for backsubstitution: 14.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 484
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 6178
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1522971, upper bound: 1.1611024
time: 4.19 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1565714, upper bound: 1.1611113
time: 4.70 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -7.1831522, -4.3880954, -7.1353726, -4.3759098, -2.4029145, 2.3728125
1: -7.3332777, -5.0522261, -7.3234835, -4.9762883, -2.2633555, 2.2245405
2: -6.1454706, -4.0189538, -6.1204944, -3.9811842, -1.7945025, 1.7438042
3: -6.2103310, -3.5549531, -6.1979203, -3.4906142, -2.5078282, 2.4402885
4: -6.5391822, -4.0432396, -6.5056667, -4.0281119, -2.5110703, 2.4624271
5: -6.5367985, -4.2877555, -6.5651464, -4.2898703, -2.0729103, 2.1178508
6: -11.5003824, -8.6774368, -11.5080471, -8.6733398, -2.7089787, 2.7130690
7: 2.7308083, 4.8404388, 2.7353547, 4.8440142, -2.0099783, 1.9944510
8: -4.4256773, -2.0390630, -4.4522839, -2.0392451, -2.0555334, 2.0679493
9: -2.8062205, -1.0470481, -2.8257987, -1.0504560, -1.7557645, 1.7787507

Time for backsubstitution: 14.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 484

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1594454, upper bound: 1.1636825
time: 4.16 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1637499, upper bound: 1.1637506
time: 6.15 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.86 seconds
IS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 24.86
Output dim: 7, lower bound: -1.1594453, upper bound: 1.1599793
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 24.86
Output dim: 7, lower bound: -1.1637500, upper bound: 1.1600546
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 24.86
Output dim: 7, lower bound: -1.1522971, upper bound: 1.1611024
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 24.86
Output dim: 7, lower bound: -1.1565714, upper bound: 1.1611113
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 24.86
Output dim: 7, lower bound: -1.1594454, upper bound: 1.1636825
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 24.86
Output dim: 7, lower bound: -1.1637499, upper bound: 1.1637506

## BFS IS instance: IS_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.1722651, -4.4013782, -7.1004920, -4.4165516, -2.3399198, 2.2890644
1: -7.2849445, -5.0607667, -7.2239141, -5.0750628, -2.1785240, 2.1404443
2: -6.1105757, -4.0252700, -6.0544534, -4.0355463, -1.7271252, 1.6766512
3: -6.1591687, -3.5638340, -6.0800219, -3.5790145, -2.3769927, 2.3130567
4: -6.5215902, -4.0622516, -6.4652872, -4.0890207, -2.4325695, 2.4030356
5: -6.5292702, -4.3011613, -6.5098367, -4.3232946, -2.0065989, 2.0351675
6: -11.4850836, -8.6937141, -11.4581814, -8.7297821, -2.6329389, 2.6423202
7: 2.7547917, 4.8101692, 2.7883945, 4.7697535, -1.9059575, 1.9101634
8: -4.4071660, -2.0567546, -4.3772392, -2.0776062, -2.0012727, 1.9754732
9: -2.7828991, -1.0626127, -2.7541091, -1.0823008, -1.7005984, 1.6914965

Time for backsubstitution: 14.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 484

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6178

## Relational analysis of IS_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1557498, upper bound: 1.1599810
time: 3.99 seconds

## Relational analysis of IS_B1_A2_B1_A2

### Relational analysis result of IS_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1557498, upper bound: 1.1599795
time: 4.63 seconds

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.1776013, -4.3897457, -7.1302657, -4.3909249, -2.3695540, 2.3310714
1: -7.2995143, -5.0570745, -7.2549372, -5.0657673, -2.2033274, 2.1721468
2: -6.1287098, -4.0227084, -6.0877566, -4.0038705, -1.7639096, 1.7109492
3: -6.1752777, -3.5593667, -6.1228414, -3.5679948, -2.4133878, 2.3660285
4: -6.5267606, -4.0468202, -6.4874988, -4.0545559, -2.4722047, 2.4406786
5: -6.5331516, -4.2968340, -6.5252767, -4.3112640, -2.0347714, 2.0727084
6: -11.4941216, -8.6792583, -11.4997997, -8.7039366, -2.6670103, 2.6980991
7: 2.7510960, 4.8360500, 2.7559779, 4.8221064, -1.9597969, 1.9639187
8: -4.4218192, -2.0535927, -4.4081450, -2.0556507, -2.0387239, 2.0112734
9: -2.8019042, -1.0588492, -2.7911725, -1.0629303, -1.7389739, 1.7323233

Time for backsubstitution: 14.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6178
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 484

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6178

## Relational analysis of IS_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1600559, upper bound: 1.1600544
time: 4.32 seconds

## Relational analysis of IS_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1600561, upper bound: 1.1600543
time: 4.49 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -7.0911522, -4.4617996, -7.1132431, -4.4295793, -2.3009958, 2.2493858
1: -7.2766156, -5.0962005, -7.2914248, -5.0024018, -2.1922693, 2.1049945
2: -6.0860543, -4.0379972, -6.0851517, -3.9921956, -1.7099910, 1.6786370
3: -6.1262259, -3.6054688, -6.1564159, -3.5197480, -2.4544230, 2.3349237
4: -6.4745512, -4.1248722, -6.4882202, -4.0858345, -2.3639936, 2.3633480
5: -6.4926033, -4.3268452, -6.5413480, -4.3098540, -1.9323359, 2.0226874
6: -11.4358139, -8.7235880, -11.4687147, -8.7015591, -2.5949230, 2.6008835
7: 2.7802281, 4.7832866, 2.7591758, 4.7944989, -1.9060841, 1.9045422
8: -4.3867764, -2.0644903, -4.4241838, -2.0530539, -1.9677148, 1.9976060
9: -2.7616210, -1.0749582, -2.7850485, -1.0665034, -1.6539073, 1.7100903

Time for backsubstitution: 14.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 484
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 484

## Relational analysis of IS_B2_A1_B1_B1

### Relational analysis result of IS_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1522971, upper bound: 1.1565696
time: 4.38 seconds

## Relational analysis of IS_B2_A1_B1_B2

### Relational analysis result of IS_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1522971, upper bound: 1.1611024
time: 4.42 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.0964441, -4.4503517, -7.1430759, -4.4038095, -2.3309336, 2.2808256
1: -7.2911277, -5.0926933, -7.3223524, -4.9928584, -2.2173870, 2.1359363
2: -6.1041646, -4.0354805, -6.1184149, -3.9604862, -1.7468491, 1.7126260
3: -6.1421337, -3.6009645, -6.1989527, -3.5086565, -2.4920235, 2.3880575
4: -6.4796653, -4.1095514, -6.5103803, -4.0510874, -2.4057326, 2.4008288
5: -6.4963479, -4.3226371, -6.5565519, -4.2975101, -1.9563870, 2.0564637
6: -11.4446487, -8.7091608, -11.5104065, -8.6755934, -2.6287193, 2.6568880
7: 2.7766769, 4.8092327, 2.7267313, 4.8476124, -1.9594567, 1.9638200
8: -4.4015961, -2.0613565, -4.4556727, -2.0311708, -2.0113149, 2.0342860
9: -2.7802789, -1.0712264, -2.8224630, -1.0471430, -1.6962101, 1.7512367

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 484
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 466

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 484

## Relational analysis of IS_B2_A1_B2_B1

### Relational analysis result of IS_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1565714, upper bound: 1.1565699
time: 4.79 seconds

## Relational analysis of IS_B2_A1_B2_B2

### Relational analysis result of IS_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1565714, upper bound: 1.1611113
time: 5.05 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.1777554, -4.3997488, -7.1260428, -4.3953948, -2.3771806, 2.3494105
1: -7.3187213, -5.0559053, -7.2993288, -4.9826579, -2.2362325, 2.1930697
2: -6.1273317, -4.0215416, -6.0901966, -3.9856868, -1.7643869, 1.7073696
3: -6.1943231, -3.5594201, -6.1716981, -3.4979277, -2.4747853, 2.3978546
4: -6.5340157, -4.0586934, -6.4969854, -4.0540309, -2.4799848, 2.4382920
5: -6.5329008, -4.2920437, -6.5582886, -4.2969966, -2.0439188, 2.0898237
6: -11.4912539, -8.6919079, -11.4923725, -8.6975431, -2.6748734, 2.6832628
7: 2.7344990, 4.8145185, 2.7415996, 4.8004618, -1.9584582, 1.9577157
8: -4.4109116, -2.0422211, -4.4274845, -2.0446577, -2.0299373, 2.0333605
9: -2.7871904, -1.0508130, -2.7937741, -1.0569978, -1.7301927, 1.7429612

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 6178
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 484

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 466

## Relational analysis of IS_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1594454, upper bound: 1.1594453
time: 4.11 seconds

## Relational analysis of IS_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1594454, upper bound: 1.1636824
time: 4.14 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.1831489, -4.3881073, -7.1559649, -4.3696532, -2.4071994, 2.3805122
1: -7.3332658, -5.0522280, -7.3301778, -4.9731512, -2.2611871, 2.2247810
2: -6.1454616, -4.0189552, -6.1234565, -3.9539649, -1.8012681, 1.7416399
3: -6.2103195, -3.5549545, -6.2141075, -3.4869413, -2.5120273, 2.4508851
4: -6.5391793, -4.0432515, -6.5191488, -4.0193825, -2.5197968, 2.4758973
5: -6.5367966, -4.2877584, -6.5737543, -4.2846642, -2.0723884, 2.1236019
6: -11.5003767, -8.6774454, -11.5341396, -8.6715851, -2.7090101, 2.7393932
7: 2.7308123, 4.8404183, 2.7091305, 4.8534975, -2.0128653, 2.0112779
8: -4.4256649, -2.0390635, -4.4589062, -2.0227418, -2.0614767, 2.0701003
9: -2.8062086, -1.0470507, -2.8311863, -1.0376325, -1.7685761, 1.7841356

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 6178
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 484

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_B2_A2_B2_A1

### Relational analysis result of IS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1637444, upper bound: 1.1603317
time: 4.51 seconds

## Relational analysis of IS_B2_A2_B2_A2

### Relational analysis result of IS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1637443, upper bound: 1.1637453
time: 4.35 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 23.41 seconds
IS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 23.41
Output dim: 7, lower bound: -1.1557498, upper bound: 1.1599810
IS_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 23.41
Output dim: 7, lower bound: -1.1557498, upper bound: 1.1599795
IS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 23.41
Output dim: 7, lower bound: -1.1600559, upper bound: 1.1600544
IS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 23.41
Output dim: 7, lower bound: -1.1600561, upper bound: 1.1600543
IS_B2_A1_B1_B1, status: Status.VERIFIED, split count: 4, time: 23.41
Output dim: 7, lower bound: -1.1522971, upper bound: 1.1565696
IS_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 23.41
Output dim: 7, lower bound: -1.1522971, upper bound: 1.1611024
IS_B2_A1_B2_B1, status: Status.VERIFIED, split count: 4, time: 23.41
Output dim: 7, lower bound: -1.1565714, upper bound: 1.1565699
IS_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 23.41
Output dim: 7, lower bound: -1.1565714, upper bound: 1.1611113
IS_B2_A2_B1_A1, status: Status.VERIFIED, split count: 4, time: 23.41
Output dim: 7, lower bound: -1.1594454, upper bound: 1.1594453
IS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 23.41
Output dim: 7, lower bound: -1.1594454, upper bound: 1.1636824
IS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 23.41
Output dim: 7, lower bound: -1.1637444, upper bound: 1.1603317
IS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 23.41
Output dim: 7, lower bound: -1.1637443, upper bound: 1.1637453

## BFS IS instance: IS_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -7.1686621, -4.4026551, -7.1004920, -4.4165516, -2.3297660, 2.2796111
1: -7.2621675, -5.0642614, -7.2239141, -5.0750628, -2.1625485, 2.1369672
2: -6.0989103, -4.0278969, -6.0544534, -4.0355463, -1.7182007, 1.6739695
3: -6.1347575, -3.5671258, -6.0800219, -3.5790145, -2.3530016, 2.3098056
4: -6.5131030, -4.0646663, -6.4652872, -4.0890207, -2.4240823, 2.4006209
5: -6.5267448, -4.3068571, -6.5098367, -4.3232946, -2.0041490, 2.0264130
6: -11.4807940, -8.6950207, -11.4581814, -8.7297821, -2.6263642, 2.6408973
7: 2.7688451, 4.8080177, 2.7883945, 4.7697535, -1.8927948, 1.9079547
8: -4.4054508, -2.0668936, -4.3772392, -2.0776062, -1.9994121, 1.9630201
9: -2.7801180, -1.0709156, -2.7541091, -1.0823008, -1.6975112, 1.6831936

Time for backsubstitution: 14.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 484

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 466

## Relational analysis of IS_B1_A2_B1_A1_A1

### Relational analysis result of IS_B1_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1557498, upper bound: 1.1557661
time: 4.09 seconds

## Relational analysis of IS_B1_A2_B1_A1_A2

### Relational analysis result of IS_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1557498, upper bound: 1.1599811
time: 4.13 seconds

## BFS IS instance: IS_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.1942091, -4.3818083, -7.1004920, -4.4165516, -2.3500674, 2.2994041
1: -7.3372846, -4.9715872, -7.2239141, -5.0750628, -2.1974654, 2.1685572
2: -6.1346316, -3.9779513, -6.0544534, -4.0355463, -1.7412605, 1.7068985
3: -6.2254667, -3.4867146, -6.0800219, -3.5790145, -2.4447336, 2.3908107
4: -6.5433569, -4.0294476, -6.4652872, -4.0890207, -2.4543362, 2.4358397
5: -6.5753670, -4.2816014, -6.5098367, -4.3232946, -2.0551777, 2.0515306
6: -11.5155993, -8.6627846, -11.4581814, -8.7297821, -2.6599412, 2.6731653
7: 2.7223897, 4.8382797, 2.7883945, 4.7697535, -1.9394457, 1.9343612
8: -4.4549408, -2.0340819, -4.3772392, -2.0776062, -2.0201278, 1.9951618
9: -2.8192260, -1.0456166, -2.7541091, -1.0823008, -1.7363398, 1.7084925

Time for backsubstitution: 14.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 484

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 466

## Relational analysis of IS_B1_A2_B1_A2_A1

### Relational analysis result of IS_B1_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1557498, upper bound: 1.1557648
time: 4.39 seconds

## Relational analysis of IS_B1_A2_B1_A2_A2

### Relational analysis result of IS_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1557498, upper bound: 1.1599795
time: 4.44 seconds

## BFS IS instance: IS_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -7.1739597, -4.3910341, -7.1302657, -4.3909249, -2.3593607, 2.3216271
1: -7.2767453, -5.0605574, -7.2549372, -5.0657673, -2.1873577, 2.1686590
2: -6.1170464, -4.0253429, -6.0877566, -4.0038705, -1.7549732, 1.7082407
3: -6.1509409, -3.5626550, -6.1228414, -3.5679948, -2.3895073, 2.3627782
4: -6.5182781, -4.0492444, -6.4874988, -4.0545559, -2.4637222, 2.4382544
5: -6.5306129, -4.3025007, -6.5252767, -4.3112640, -2.0323009, 2.0640130
6: -11.4897652, -8.6805706, -11.4997997, -8.7039366, -2.6604671, 2.6966705
7: 2.7651470, 4.8338819, 2.7559779, 4.8221064, -1.9466248, 1.9615586
8: -4.4200940, -2.0637312, -4.4081450, -2.0556507, -2.0372195, 1.9988141
9: -2.7991097, -1.0671506, -2.7911725, -1.0629303, -1.7361794, 1.7240219

Time for backsubstitution: 14.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 484

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5746

## Relational analysis of IS_B1_A2_B2_A1_B1

### Relational analysis result of IS_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1566373, upper bound: 1.1600490
time: 4.70 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2

### Relational analysis result of IS_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1600504, upper bound: 1.1600488
time: 6.10 seconds

## BFS IS instance: IS_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.1996403, -4.3701682, -7.1302657, -4.3909249, -2.3796792, 2.3414407
1: -7.3517547, -4.9677134, -7.2549372, -5.0657673, -2.2221906, 2.2011356
2: -6.1527472, -3.9753232, -6.0877566, -4.0038705, -1.7779758, 1.7414691
3: -6.2411661, -3.4822431, -6.1228414, -3.5679948, -2.4809513, 2.4437249
4: -6.5484152, -4.0139184, -6.4874988, -4.0545559, -2.4938593, 2.4735804
5: -6.5792475, -4.2772551, -6.5252767, -4.3112640, -2.0833774, 2.0889213
6: -11.5247192, -8.6483126, -11.4997997, -8.7039366, -2.6940508, 2.7289486
7: 2.7187438, 4.8643641, 2.7559779, 4.8221064, -1.9932070, 1.9823415
8: -4.4699135, -2.0309606, -4.4081450, -2.0556507, -2.0517144, 2.0309358
9: -2.8383889, -1.0418553, -2.7911725, -1.0629303, -1.7720928, 1.7493172

Time for backsubstitution: 14.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 484

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5746

## Relational analysis of IS_B1_A2_B2_A2_B1

### Relational analysis result of IS_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1566373, upper bound: 1.1600494
time: 4.52 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2

### Relational analysis result of IS_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1600503, upper bound: 1.1600488
time: 4.90 seconds

## BFS IS instance: IS_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -7.0911522, -4.4617996, -7.1875210, -4.3891964, -2.3276482, 2.2860184
1: -7.2766156, -5.0962005, -7.3255529, -4.9742041, -2.2148397, 2.1414592
2: -6.0860543, -4.0379972, -6.1201100, -3.9797525, -1.7187734, 1.7104471
3: -6.1262259, -3.6054688, -6.2132015, -3.4890842, -2.4726157, 2.3899689
4: -6.4745512, -4.1248722, -6.5386200, -4.0401068, -2.3961325, 2.4137478
5: -6.4926033, -4.3268452, -6.5721598, -4.2844524, -1.9541993, 2.0452385
6: -11.4358139, -8.7235880, -11.5084515, -8.6729250, -2.6232176, 2.6459465
7: 2.7802281, 4.7832866, 2.7253838, 4.8211708, -1.9239585, 1.9436653
8: -4.3867764, -2.0644903, -4.4447899, -2.0364761, -1.9849105, 2.0180199
9: -2.7616210, -1.0749582, -2.8068171, -1.0489899, -1.6716084, 1.7318588

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 6178
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 466

## Relational analysis of IS_B2_A1_B1_B2_A1

### Relational analysis result of IS_B2_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1522971, upper bound: 1.1568603
time: 4.41 seconds

## Relational analysis of IS_B2_A1_B1_B2_A2

### Relational analysis result of IS_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1522971, upper bound: 1.1611024
time: 4.46 seconds

## BFS IS instance: IS_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -7.0964441, -4.4503517, -7.2171926, -4.3633676, -2.3575435, 2.3174596
1: -7.2911277, -5.0926933, -7.3564181, -4.9645948, -2.2399704, 2.1724257
2: -6.1041646, -4.0354805, -6.1533203, -3.9480333, -1.7556372, 1.7444773
3: -6.1421337, -3.6009645, -6.2554965, -3.4779308, -2.5102458, 2.4430559
4: -6.4796653, -4.1095514, -6.5607886, -4.0052691, -2.4387517, 2.4512372
5: -6.4963479, -4.3226371, -6.5874124, -4.2719798, -1.9783533, 2.0790436
6: -11.4446487, -8.7091608, -11.5500975, -8.6469555, -2.6570053, 2.7019553
7: 2.7766769, 4.8092327, 2.6929038, 4.8741589, -1.9772234, 1.9923542
8: -4.4015961, -2.0613565, -4.4762692, -2.0145826, -2.0281053, 2.0545723
9: -2.7802789, -1.0712264, -2.8441491, -1.0296247, -1.7139187, 1.7729228

Time for backsubstitution: 14.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 466

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_B2_A1_B2_B2_A1

### Relational analysis result of IS_B2_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1565659, upper bound: 1.1577113
time: 4.94 seconds

## Relational analysis of IS_B2_A1_B2_B2_A2

### Relational analysis result of IS_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1565659, upper bound: 1.1611059
time: 4.44 seconds

## BFS IS instance: IS_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.2037191, -4.3817487, -7.1260428, -4.3953948, -2.3899550, 2.3574600
1: -7.3401504, -5.0489416, -7.2993288, -4.9826579, -2.2506998, 2.1974368
2: -6.1484261, -3.9915962, -6.0901966, -3.9856868, -1.7706020, 1.7333133
3: -6.2259359, -3.5511444, -6.1716981, -3.4979277, -2.5032258, 2.4100459
4: -6.5526695, -4.0343275, -6.4969854, -4.0540309, -2.4986386, 2.4626579
5: -6.5455341, -4.2827621, -6.5582886, -4.2969966, -2.0585616, 2.1006129
6: -11.5267048, -8.6757164, -11.4923725, -8.6975431, -2.7098913, 2.7005887
7: 2.7045596, 4.8498659, 2.7415996, 4.8004618, -1.9771659, 1.9848599
8: -4.4324198, -2.0225391, -4.4274845, -2.0446577, -2.0403414, 2.0407436
9: -2.8115807, -1.0342282, -2.7937741, -1.0569978, -1.7545829, 1.7595459

Time for backsubstitution: 14.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 484

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_B2_A2_B1_A2_A1

### Relational analysis result of IS_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1594396, upper bound: 1.1602470
time: 4.48 seconds

## Relational analysis of IS_B2_A2_B1_A2_A2

### Relational analysis result of IS_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1594398, upper bound: 1.1636771
time: 4.52 seconds

## BFS IS instance: IS_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -7.1704130, -4.3928919, -7.1512442, -4.3710880, -2.3870983, 2.3659132
1: -7.2775440, -5.0866270, -7.3007364, -4.9794874, -2.2014685, 2.1622338
2: -6.1195035, -4.0467563, -6.1183820, -3.9686239, -1.7542965, 1.7079842
3: -6.1744261, -3.5654070, -6.2049866, -3.4894705, -2.4662704, 2.4184308
4: -6.4864964, -4.0795736, -6.4910393, -4.0257978, -2.4606986, 2.4114656
5: -6.5125408, -4.3016362, -6.5611606, -4.2885294, -2.0433431, 2.0916324
6: -11.4658966, -8.6952496, -11.5169220, -8.6750507, -2.6703711, 2.7018480
7: 2.7682827, 4.8226004, 2.7161984, 4.8478837, -1.9682691, 1.9811850
8: -4.3959675, -2.0805349, -4.4510479, -2.0451860, -2.0001254, 2.0190620
9: -2.7594464, -1.1040378, -2.8229477, -1.0691594, -1.6902870, 1.7189100

Time for backsubstitution: 14.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 6178
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 484

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5805

## Relational analysis of IS_B2_A2_B2_A1_B1

### Relational analysis result of IS_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1547117, upper bound: 1.1555726
time: 4.25 seconds

## Relational analysis of IS_B2_A2_B2_A1_B2

### Relational analysis result of IS_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1637413, upper bound: 1.1603293
time: 4.49 seconds

## BFS IS instance: IS_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.1831455, -4.3881073, -7.1559649, -4.3696551, -2.4090552, 2.3746433
1: -7.3332438, -5.0522318, -7.3301678, -4.9731541, -2.2253799, 2.2226181
2: -6.1454592, -4.0189648, -6.1234546, -3.9539690, -1.7912562, 1.7271702
3: -6.2103138, -3.5549569, -6.2141056, -3.4869423, -2.5008097, 2.4468620
4: -6.5391569, -4.0432534, -6.5191393, -4.0193834, -2.5020361, 2.4758859
5: -6.5367875, -4.2877612, -6.5737505, -4.2846646, -2.0630150, 2.1195221
6: -11.5003672, -8.6774473, -11.5341339, -8.6715860, -2.6987381, 2.7392583
7: 2.7308145, 4.8404155, 2.7091317, 4.8534966, -2.0088520, 2.0032341
8: -4.4256611, -2.0390730, -4.4589043, -2.0227461, -2.0531549, 2.0526993
9: -2.8062043, -1.0470624, -2.8311837, -1.0376390, -1.7685653, 1.7796479

Time for backsubstitution: 14.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 6178
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 5746

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5805

## Relational analysis of IS_B2_A2_B2_A2_B1

### Relational analysis result of IS_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1547119, upper bound: 1.1589788
time: 7.03 seconds

## Relational analysis of IS_B2_A2_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1637413, upper bound: 1.1637426
time: 4.46 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 25.92 seconds
IS_B1_A2_B1_A1_A1, status: Status.VERIFIED, split count: 5, time: 25.92
Output dim: 7, lower bound: -1.1557498, upper bound: 1.1557661
IS_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 7, lower bound: -1.1557498, upper bound: 1.1599811
IS_B1_A2_B1_A2_A1, status: Status.VERIFIED, split count: 5, time: 25.92
Output dim: 7, lower bound: -1.1557498, upper bound: 1.1557648
IS_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 7, lower bound: -1.1557498, upper bound: 1.1599795
IS_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 7, lower bound: -1.1566373, upper bound: 1.1600490
IS_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 7, lower bound: -1.1600504, upper bound: 1.1600488
IS_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 7, lower bound: -1.1566373, upper bound: 1.1600494
IS_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 7, lower bound: -1.1600503, upper bound: 1.1600488
IS_B2_A1_B1_B2_A1, status: Status.VERIFIED, split count: 5, time: 25.92
Output dim: 7, lower bound: -1.1522971, upper bound: 1.1568603
IS_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 7, lower bound: -1.1522971, upper bound: 1.1611024
IS_B2_A1_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 25.92
Output dim: 7, lower bound: -1.1565659, upper bound: 1.1577113
IS_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 7, lower bound: -1.1565659, upper bound: 1.1611059
IS_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 7, lower bound: -1.1594396, upper bound: 1.1602470
IS_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 7, lower bound: -1.1594398, upper bound: 1.1636771
IS_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 25.92
Output dim: 7, lower bound: -1.1547117, upper bound: 1.1555726
IS_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 7, lower bound: -1.1637413, upper bound: 1.1603293
IS_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 25.92
Output dim: 7, lower bound: -1.1547119, upper bound: 1.1589788
IS_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 7, lower bound: -1.1637413, upper bound: 1.1637426

## BFS IS instance: IS_B1_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -7.1944447, -4.3847656, -7.1004920, -4.4165516, -2.3424428, 2.2969666
1: -7.2834435, -5.0573325, -7.2239141, -5.0750628, -2.1768541, 2.1414084
2: -6.1200218, -3.9980054, -6.0544534, -4.0355463, -1.7244029, 1.6997343
3: -6.1664863, -3.5588291, -6.0800219, -3.5790145, -2.3848648, 2.3219221
4: -6.5315881, -4.0403180, -6.4652872, -4.0890207, -2.4425673, 2.4249692
5: -6.5392761, -4.2976375, -6.5098367, -4.3232946, -2.0187807, 2.0379233
6: -11.5160484, -8.6788864, -11.4581814, -8.7297821, -2.6613221, 2.6581521
7: 2.7390420, 4.8430414, 2.7883945, 4.7697535, -1.9234264, 1.9348361
8: -4.4266005, -2.0472507, -4.3772392, -2.0776062, -2.0157318, 1.9810362
9: -2.8044434, -1.0543329, -2.7541091, -1.0823008, -1.7219868, 1.6997763

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 484

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5746

## Relational analysis of IS_B1_A2_B1_A1_A2_B1

### Relational analysis result of IS_B1_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1523155, upper bound: 1.1599760
time: 4.47 seconds

## Relational analysis of IS_B1_A2_B1_A1_A2_B2

### Relational analysis result of IS_B1_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1557442, upper bound: 1.1599764
time: 4.41 seconds

## BFS IS instance: IS_B1_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -7.2200460, -4.3638263, -7.1004920, -4.4165516, -2.3629405, 2.3171902
1: -7.3584623, -4.9645224, -7.2239141, -5.0750628, -2.2122526, 2.1738749
2: -6.1556773, -3.9481032, -6.0544534, -4.0355463, -1.7476444, 1.7173307
3: -6.2571154, -3.4784498, -6.0800219, -3.5790145, -2.4770722, 2.4011664
4: -6.5618796, -4.0050106, -6.4652872, -4.0890207, -2.4728589, 2.4602766
5: -6.5879145, -4.2719722, -6.5098367, -4.3232946, -2.0698562, 2.0631599
6: -11.5507946, -8.6465540, -11.4581814, -8.7297821, -2.6948800, 2.6869631
7: 2.6925116, 4.8738089, 2.7883945, 4.7697535, -1.9615009, 1.9557469
8: -4.4764662, -2.0144486, -4.3772392, -2.0776062, -2.0305743, 2.0117521
9: -2.8437579, -1.0290372, -2.7541091, -1.0823008, -1.7507083, 1.7250719

Time for backsubstitution: 14.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 484

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5746

## Relational analysis of IS_B1_A2_B1_A2_A2_B1

### Relational analysis result of IS_B1_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1560068, upper bound: 1.1599737
time: 4.52 seconds

## Relational analysis of IS_B1_A2_B1_A2_A2_B2

### Relational analysis result of IS_B1_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1594390, upper bound: 1.1599730
time: 4.70 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -7.1693230, -4.3925495, -7.1176558, -4.3955088, -2.3449395, 2.3013430
1: -7.2471304, -5.0666094, -7.1992106, -5.0999427, -2.1142476, 2.1088791
2: -6.1119337, -4.0399971, -6.0620136, -4.0315456, -1.7215035, 1.6667480
3: -6.1418810, -3.5652146, -6.0866356, -3.5784848, -2.3563170, 2.3165398
4: -6.4902773, -4.0553179, -6.4345884, -4.0908275, -2.3994498, 2.3792706
5: -6.5180607, -4.3062830, -6.5011196, -4.3251281, -2.0023644, 2.0351207
6: -11.4727345, -8.6838417, -11.4656906, -8.7216549, -2.6236801, 2.6584530
7: 2.7722926, 4.8282309, 2.7934084, 4.8043604, -1.9166212, 1.9168541
8: -4.4123487, -2.0861931, -4.3783607, -2.0971422, -1.9864373, 1.9458277
9: -2.7909980, -1.0987713, -2.7445490, -1.1197567, -1.6712413, 1.6457777

Time for backsubstitution: 14.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 484

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5805

## Relational analysis of IS_B1_A2_B2_A1_B1_B1

### Relational analysis result of IS_B1_A2_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1476091, upper bound: 1.1552583
time: 6.18 seconds

## Relational analysis of IS_B1_A2_B2_A1_B1_B2

### Relational analysis result of IS_B1_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1566343, upper bound: 1.1600486
time: 5.26 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.1739597, -4.3910346, -7.1302619, -4.3909264, -2.3535237, 2.3259201
1: -7.2767348, -5.0605593, -7.2549138, -5.0657701, -2.1709621, 2.1363254
2: -6.1170449, -4.0253477, -6.0877562, -4.0038800, -1.7375073, 1.7082350
3: -6.1509376, -3.5626554, -6.1228361, -3.5679967, -2.3853974, 2.3584955
4: -6.5182686, -4.0492458, -6.4874768, -4.0545588, -2.4637098, 2.4382310
5: -6.5306096, -4.3025017, -6.5252681, -4.3112669, -2.0322964, 2.0546389
6: -11.4897604, -8.6805706, -11.4997892, -8.7039394, -2.6602659, 2.6861210
7: 2.7651486, 4.8338795, 2.7559817, 4.8221030, -1.9449005, 1.9514139
8: -4.4200935, -2.0637341, -4.4081411, -2.0556588, -2.0198631, 1.9988072
9: -2.7991080, -1.0671568, -2.7911687, -1.0629426, -1.7235479, 1.7240119

Time for backsubstitution: 14.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 484
type: A, layer: 1, pos: 5746

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5805

## Relational analysis of IS_B1_A2_B2_A1_B2_B1

### Relational analysis result of IS_B1_A2_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1509920, upper bound: 1.1552585
time: 4.80 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2_B2

### Relational analysis result of IS_B1_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1600473, upper bound: 1.1600490
time: 4.22 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.1949196, -4.3716521, -7.1176558, -4.3955088, -2.3652103, 2.3211012
1: -7.3222160, -4.9739213, -7.1992106, -5.0999427, -2.1492574, 2.1414714
2: -6.1475449, -3.9900331, -6.0620136, -4.0315456, -1.7444479, 1.6947865
3: -6.2323003, -3.4847860, -6.0866356, -3.5784848, -2.4480581, 2.3976972
4: -6.5204029, -4.0202136, -6.4345884, -4.0908275, -2.4295754, 2.4143748
5: -6.5666389, -4.2810798, -6.5011196, -4.3251281, -2.0533869, 2.0600853
6: -11.5074196, -8.6517353, -11.4656906, -8.7216549, -2.6568451, 2.6904001
7: 2.7257557, 4.8586988, 2.7934084, 4.8043604, -1.9633956, 1.9375293
8: -4.4620647, -2.0533957, -4.3783607, -2.0971422, -2.0010161, 1.9779894
9: -2.8301170, -1.0734553, -2.7445490, -1.1197567, -1.7047288, 1.6710937

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 484

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5805

## Relational analysis of IS_B1_A2_B2_A2_B1_B1

### Relational analysis result of IS_B1_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1513160, upper bound: 1.1552560
time: 4.51 seconds

## Relational analysis of IS_B1_A2_B2_A2_B1_B2

### Relational analysis result of IS_B1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1603277, upper bound: 1.1600462
time: 5.68 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.1996388, -4.3701677, -7.1302619, -4.3909264, -2.3738337, 2.3454418
1: -7.3517437, -4.9677148, -7.2549138, -5.0657701, -2.2057955, 2.1653647
2: -6.1527452, -3.9753277, -6.0877562, -4.0038800, -1.7604852, 1.7315676
3: -6.2411642, -3.4822440, -6.1228361, -3.5679967, -2.4768791, 2.4324768
4: -6.5484056, -4.0139208, -6.4874768, -4.0545588, -2.4938469, 2.4716992
5: -6.5792427, -4.2772560, -6.5252681, -4.3112669, -2.0833728, 2.0795479
6: -11.5247154, -8.6483135, -11.4997892, -8.7039394, -2.6938486, 2.7133734
7: 2.7187462, 4.8643637, 2.7559817, 4.8221030, -1.9914827, 1.9721956
8: -4.4699101, -2.0309668, -4.4081411, -2.0556588, -2.0343618, 2.0309293
9: -2.8383863, -1.0418612, -2.7911687, -1.0629426, -1.7502022, 1.7493075

Time for backsubstitution: 14.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 484
type: A, layer: 1, pos: 5746

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5805

## Relational analysis of IS_B1_A2_B2_A2_B2_B1

### Relational analysis result of IS_B1_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1547114, upper bound: 1.1552566
time: 4.61 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2_B2

### Relational analysis result of IS_B1_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1637407, upper bound: 1.1600468
time: 4.45 seconds

## BFS IS instance: IS_B2_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -7.1172099, -4.4442263, -7.1875210, -4.3891964, -2.3404698, 2.2947109
1: -7.2980108, -5.0895948, -7.3255529, -4.9742041, -2.2292449, 2.1452618
2: -6.1071701, -4.0081677, -6.1201100, -3.9797525, -1.7249389, 1.7325594
3: -6.1579189, -3.5972173, -6.2132015, -3.4890842, -2.5003362, 2.4021473
4: -6.4931488, -4.1009493, -6.5386200, -4.0401068, -2.4064503, 2.4376707
5: -6.5044765, -4.3176551, -6.5721598, -4.2844524, -1.9671855, 2.0558455
6: -11.4710159, -8.7074261, -11.5084515, -8.6729250, -2.6580915, 2.6632214
7: 2.7505269, 4.8187757, 2.7253838, 4.8211708, -1.9359562, 1.9658930
8: -4.4083490, -2.0449190, -4.4447899, -2.0364761, -2.0046024, 2.0253611
9: -2.7855060, -1.0584118, -2.8068171, -1.0489899, -1.6949701, 1.7484052

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_B2_A1_B1_B2_A2_A1

### Relational analysis result of IS_B2_A1_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1522915, upper bound: 1.1576656
time: 4.74 seconds

## Relational analysis of IS_B2_A1_B1_B2_A2_A2

### Relational analysis result of IS_B2_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1522914, upper bound: 1.1610971
time: 4.46 seconds

## BFS IS instance: IS_B2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -7.0964403, -4.4503527, -7.2171926, -4.3633685, -2.3609598, 2.3110609
1: -7.2911053, -5.0926967, -7.3564086, -4.9645953, -2.2018695, 2.1580822
2: -6.1041632, -4.0354900, -6.1533203, -3.9480383, -1.7456703, 1.7296615
3: -6.1421275, -3.6009657, -6.2554936, -3.4779310, -2.5000672, 2.4390268
4: -6.4796438, -4.1095552, -6.5607772, -4.0052705, -2.3945689, 2.4512219
5: -6.4963403, -4.3226395, -6.5874095, -4.2719808, -1.9700885, 2.0749340
6: -11.4446392, -8.7091618, -11.5500946, -8.6469564, -2.6461353, 2.6984439
7: 2.7766805, 4.8092289, 2.6929071, 4.8741570, -1.9672642, 1.9843445
8: -4.4015932, -2.0613661, -4.4762673, -2.0145845, -2.0197835, 2.0372438
9: -2.7802751, -1.0712389, -2.8441477, -1.0296293, -1.7094595, 1.7729088

Time for backsubstitution: 14.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 468
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 5746

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5805

## Relational analysis of IS_B2_A1_B2_B2_A2_B1

### Relational analysis result of IS_B2_A1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1476430, upper bound: 1.1564590
time: 4.52 seconds

## Relational analysis of IS_B2_A1_B2_B2_A2_B2

### Relational analysis result of IS_B2_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1565628, upper bound: 1.1611030
time: 4.66 seconds

## BFS IS instance: IS_B2_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -7.1910467, -4.3863831, -7.1212964, -4.3968849, -2.3698587, 2.3427618
1: -7.2844639, -5.0832605, -7.2695885, -4.9888611, -2.1908774, 2.1347535
2: -6.1226578, -4.0192785, -6.0850334, -4.0003443, -1.7237325, 1.6997476
3: -6.1897106, -3.5615730, -6.1626143, -3.5004139, -2.4573717, 2.3776219
4: -6.4997516, -4.0707555, -6.4690018, -4.0603809, -2.4393706, 2.3982463
5: -6.5212994, -4.2966032, -6.5456729, -4.3008113, -2.0295377, 2.0685287
6: -11.4923229, -8.6935959, -11.4751053, -8.7009726, -2.6706767, 2.6627698
7: 2.7420490, 4.8321018, 2.7487459, 4.7947516, -1.9327917, 1.9551497
8: -4.4025965, -2.0640135, -4.4196959, -2.0670862, -1.9788303, 1.9899499
9: -2.7647805, -1.0910977, -2.7855062, -1.0885727, -1.6724896, 1.6944085

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 484

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5805

## Relational analysis of IS_B2_A2_B1_A2_A1_B1

### Relational analysis result of IS_B2_A2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1504206, upper bound: 1.1554982
time: 4.50 seconds

## Relational analysis of IS_B2_A2_B1_A2_A1_B2

### Relational analysis result of IS_B2_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1594367, upper bound: 1.1602450
time: 4.64 seconds

## BFS IS instance: IS_B2_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -7.2037172, -4.3817511, -7.1260419, -4.3953972, -2.3919353, 2.3516004
1: -7.3401279, -5.0489440, -7.2993193, -4.9826593, -2.2149100, 2.1953340
2: -6.1484251, -3.9916062, -6.0901966, -3.9856911, -1.7606816, 1.7159185
3: -6.2259302, -3.5511460, -6.1716957, -3.4979277, -2.4920402, 2.4059801
4: -6.5526481, -4.0343304, -6.4969745, -4.0540328, -2.4696183, 2.4626441
5: -6.5455256, -4.2827644, -6.5582857, -4.2969975, -2.0491867, 2.0965474
6: -11.5266943, -8.6757183, -11.4923677, -8.6975441, -2.6989851, 2.7004001
7: 2.7045641, 4.8498645, 2.7416015, 4.8004618, -1.9669266, 1.9768937
8: -4.4324174, -2.0225501, -4.4274836, -2.0446630, -2.0319538, 2.0233653
9: -2.8115764, -1.0342400, -2.7937729, -1.0570036, -1.7476797, 1.7461011

Time for backsubstitution: 14.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 468
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 5746

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5805

## Relational analysis of IS_B2_A2_B1_A2_A2_B1

### Relational analysis result of IS_B2_A2_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1504208, upper bound: 1.1589163
time: 4.47 seconds

## Relational analysis of IS_B2_A2_B1_A2_A2_B2

### Relational analysis result of IS_B2_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1594367, upper bound: 1.1636739
time: 4.36 seconds

## BFS IS instance: IS_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.1704082, -4.3928957, -7.1512356, -4.3711004, -2.3698325, 2.3525944
1: -7.2775431, -5.0866280, -7.3007288, -4.9794917, -2.1978800, 2.1555893
2: -6.1195025, -4.0467606, -6.1183777, -3.9686358, -1.7245617, 1.7079806
3: -6.1744213, -3.5654078, -6.2049723, -3.4894726, -2.4594240, 2.4033525
4: -6.4864960, -4.0795794, -6.4910355, -4.0258107, -2.4606853, 2.4114561
5: -6.5125351, -4.3016367, -6.5611525, -4.2885313, -2.0433364, 2.0777881
6: -11.4658937, -8.6952553, -11.5169163, -8.6750660, -2.6550522, 2.7018361
7: 2.7682853, 4.8225956, 2.7162056, 4.8478756, -1.9521055, 1.9768469
8: -4.3959637, -2.0805378, -4.4510365, -2.0451889, -1.9919877, 1.9933863
9: -2.7594426, -1.1040382, -2.8229408, -1.0691627, -1.6902798, 1.7189026

Time for backsubstitution: 14.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 6178
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 468
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 79
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 468
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 484

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4629

## Relational analysis of IS_B2_A2_B2_A1_B2_B1

### Relational analysis result of IS_B2_A2_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1544830, upper bound: 1.1549380
time: 4.63 seconds

## Relational analysis of IS_B2_A2_B2_A1_B2_B2

### Relational analysis result of IS_B2_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1637371, upper bound: 1.1603260
time: 4.37 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 23.42 seconds
IS_B1_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 23.42
Output dim: 7, lower bound: -1.1523155, upper bound: 1.1599760
IS_B1_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 23.42
Output dim: 7, lower bound: -1.1557442, upper bound: 1.1599764
IS_B1_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 23.42
Output dim: 7, lower bound: -1.1560068, upper bound: 1.1599737
IS_B1_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 23.42
Output dim: 7, lower bound: -1.1594390, upper bound: 1.1599730
IS_B1_A2_B2_A1_B1_B1, status: Status.VERIFIED, split count: 6, time: 23.42
Output dim: 7, lower bound: -1.1476091, upper bound: 1.1552583
IS_B1_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 23.42
Output dim: 7, lower bound: -1.1566343, upper bound: 1.1600486
IS_B1_A2_B2_A1_B2_B1, status: Status.VERIFIED, split count: 6, time: 23.42
Output dim: 7, lower bound: -1.1509920, upper bound: 1.1552585
IS_B1_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 23.42
Output dim: 7, lower bound: -1.1600473, upper bound: 1.1600490
IS_B1_A2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 6, time: 23.42
Output dim: 7, lower bound: -1.1513160, upper bound: 1.1552560
IS_B1_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 23.42
Output dim: 7, lower bound: -1.1603277, upper bound: 1.1600462
IS_B1_A2_B2_A2_B2_B1, status: Status.VERIFIED, split count: 6, time: 23.42
Output dim: 7, lower bound: -1.1547114, upper bound: 1.1552566
IS_B1_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 23.42
Output dim: 7, lower bound: -1.1637407, upper bound: 1.1600468
IS_B2_A1_B1_B2_A2_A1, status: Status.VERIFIED, split count: 6, time: 23.42
Output dim: 7, lower bound: -1.1522915, upper bound: 1.1576656
IS_B2_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 23.42
Output dim: 7, lower bound: -1.1522914, upper bound: 1.1610971
IS_B2_A1_B2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 23.42
Output dim: 7, lower bound: -1.1476430, upper bound: 1.1564590
IS_B2_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 23.42
Output dim: 7, lower bound: -1.1565628, upper bound: 1.1611030
IS_B2_A2_B1_A2_A1_B1, status: Status.VERIFIED, split count: 6, time: 23.42
Output dim: 7, lower bound: -1.1504206, upper bound: 1.1554982
IS_B2_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 23.42
Output dim: 7, lower bound: -1.1594367, upper bound: 1.1602450
IS_B2_A2_B1_A2_A2_B1, status: Status.VERIFIED, split count: 6, time: 23.42
Output dim: 7, lower bound: -1.1504208, upper bound: 1.1589163
IS_B2_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 23.42
Output dim: 7, lower bound: -1.1594367, upper bound: 1.1636739
IS_B2_A2_B2_A1_B2_B1, status: Status.VERIFIED, split count: 6, time: 23.42
Output dim: 7, lower bound: -1.1544830, upper bound: 1.1549380
IS_B2_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 23.42
Output dim: 7, lower bound: -1.1637371, upper bound: 1.1603260
IS_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 23.42
Output dim: 7, lower bound: -1.1637413, upper bound: 1.1637426
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=1.974029779434204
rel_dist={7: [-1.163764129436149, 1.1637638135586528]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2422.58 seconds
