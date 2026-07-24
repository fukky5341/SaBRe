## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.02634580263
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-15.8640556, -11.6210804, -15.8640556, -11.6210804, -3.7144022, 3.7144022)
1: (-7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716)
2: (-8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809)
3: (-5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.6043229, 2.6043229)
4: (-7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.6161270, 2.6161273)
5: (-6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.6302273, 2.6302273)
6: (-14.4134359, -10.9648418, -14.4134359, -10.9648418, -3.2066069, 3.2066069)
7: (2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.4982886, 2.4982884)
8: (-1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.3115258, 2.3115258)
9: (-8.8183250, -5.7160292, -8.8183250, -5.7160292, -3.0185328, 3.0185328)

## BASE Result
execution time: IAR + LP analysis = 14.12 + 32.44 = 46.56 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3553.44 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.045863628387451
rel_dist={7: [-1.3815082971736095, 1.3815080831477027]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=1.8196511268615723
rel_dist={7: [-1.029880698625119, 1.029879313881131]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=1.6688427925109863
rel_dist={7: [-0.7506851770062117, 0.7506857888818104]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=1.7442469596862793
rel_dist={7: [-0.897989859838467, 0.8979910480313595]}

## Binary Search Result
Binary search time: 198.32 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual) starts
Time budget: 3355.12 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6192
type: A, layer: 1, pos: 6192
type: A, layer: 1, pos: 6156
type: B, layer: 1, pos: 6156
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6192

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4583236, upper bound: 1.4777224
time: 4.39 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4841231, upper bound: 1.4841236
time: 3.93 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8.50 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 8.50
Output dim: 7, lower bound: -1.4583236, upper bound: 1.4777224
IS_B2, status: Status.UNKNOWN, split count: 1, time: 8.50
Output dim: 7, lower bound: -1.4841231, upper bound: 1.4841236

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -15.8392849, -11.6234875, -15.7687855, -11.6553326, -2.9726334, 2.9248013
1: -7.1877794, -4.3867526, -7.1591153, -4.4076223, -2.7801571, 2.7723627
2: -8.7379065, -6.1465340, -8.7206879, -6.1691089, -2.5687976, 2.5741539
3: -5.0084758, -2.4245584, -4.9593029, -2.4537749, -2.4590654, 2.4454134
4: -7.9676423, -5.2738123, -7.9517303, -5.2917261, -2.1686158, 2.1669877
5: -6.3194108, -3.7140961, -6.2641287, -3.7503946, -2.4186459, 2.3779609
6: -14.4092875, -10.9741488, -14.3885374, -11.0027924, -2.6031780, 2.6212471
7: 2.2585292, 4.8222275, 2.2882204, 4.7757864, -2.0556374, 2.0816388
8: -1.3055716, 0.9755721, -1.2247856, 0.9384165, -2.1109242, 2.0776079
9: -8.8161774, -5.7351828, -8.7875423, -5.7901473, -2.4225168, 2.4491284

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6192
type: A, layer: 1, pos: 6156
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6192

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4583236, upper bound: 1.4583233
time: 4.61 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4583236, upper bound: 1.4777225
time: 4.13 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -15.8640556, -11.6210804, -15.8640451, -11.6210775, -3.0224762, 2.9801788
1: -7.1961079, -4.3803363, -7.1961031, -4.3803387, -2.8157692, 2.8157668
2: -8.7408409, -6.1397600, -8.7408390, -6.1397657, -2.6010752, 2.6010790
3: -5.0245595, -2.4202366, -5.0245490, -2.4202378, -2.5323892, 2.5157423
4: -7.9703813, -5.2681599, -7.9703798, -5.2681632, -2.1964865, 2.2019093
5: -6.3388715, -3.7086442, -6.3388667, -3.7086473, -2.5019703, 2.4796739
6: -14.4134359, -10.9648418, -14.4134312, -10.9648504, -2.6285658, 2.6426473
7: 2.2540903, 4.8381100, 2.2540932, 4.8381004, -2.1051321, 2.1212654
8: -1.3332825, 0.9782434, -1.3332644, 0.9782419, -2.1878629, 2.1700807
9: -8.8183250, -5.7160292, -8.8183231, -5.7160378, -2.4725432, 2.4985061

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6156
type: A, layer: 1, pos: 6156
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6192
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6135

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6156

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4841030, upper bound: 1.4685759
time: 4.12 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4841030, upper bound: 1.4841034
time: 3.90 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 22.59 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 22.59
Output dim: 7, lower bound: -1.4583236, upper bound: 1.4583233
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 22.59
Output dim: 7, lower bound: -1.4583236, upper bound: 1.4777225
IS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 22.59
Output dim: 7, lower bound: -1.4841030, upper bound: 1.4685759
IS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 22.59
Output dim: 7, lower bound: -1.4841030, upper bound: 1.4841034

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -15.7687855, -11.6553326, -15.7687855, -11.6553326, -2.9012766, 2.9012761
1: -7.1591153, -4.4076223, -7.1591153, -4.4076223, -2.7514930, 2.7514930
2: -8.7206879, -6.1691089, -8.7206879, -6.1691089, -2.5515790, 2.5515790
3: -4.9593029, -2.4537749, -4.9593029, -2.4537749, -2.4127522, 2.4127519
4: -7.9517303, -5.2917261, -7.9517303, -5.2917261, -2.1495900, 2.1495900
5: -6.2641287, -3.7503946, -6.2641287, -3.7503946, -2.3463941, 2.3463943
6: -14.3885374, -11.0027924, -14.3885374, -11.0027924, -2.5922670, 2.5922661
7: 2.2882204, 4.7757864, 2.2882204, 4.7757864, -2.0347400, 2.0347395
8: -1.2247856, 0.9384165, -1.2247856, 0.9384165, -2.0435867, 2.0435870
9: -8.7875423, -5.7901473, -8.7875423, -5.7901473, -2.3939881, 2.3939884

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6156
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6156

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4426295, upper bound: 1.4583076
time: 4.83 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4583031, upper bound: 1.4583073
time: 4.14 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -15.8640451, -11.6210775, -15.7687855, -11.6553326, -2.9789171, 2.9260716
1: -7.1961031, -4.3803387, -7.1591153, -4.4076223, -2.7884808, 2.7787766
2: -8.7408390, -6.1397657, -8.7206879, -6.1691089, -2.5717301, 2.5809221
3: -5.0245490, -2.4202378, -4.9593029, -2.4537749, -2.4695363, 2.4462068
4: -7.9703798, -5.2681632, -7.9517303, -5.2917261, -2.1701384, 2.1748021
5: -6.3388667, -3.7086473, -6.2641287, -3.7503946, -2.4377604, 2.3791244
6: -14.4134312, -10.9648504, -14.3885374, -11.0027924, -2.6043921, 2.6305156
7: 2.2540932, 4.8381004, 2.2882204, 4.7757864, -2.0581393, 2.0895212
8: -1.3332644, 0.9782419, -1.2247856, 0.9384165, -2.1133976, 2.0795605
9: -8.8183231, -5.7160378, -8.7875423, -5.7901473, -2.4242902, 2.4527586

Time for backsubstitution: 14.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6156
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6156

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4426295, upper bound: 1.4777025
time: 5.00 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4583031, upper bound: 1.4777022
time: 4.34 seconds

## BFS IS instance: IS_B2_B1

### Backsubstitution after applying IS history:
0: -15.8619432, -11.6303759, -15.8424368, -11.6793060, -2.9583979, 2.9492555
1: -7.1948786, -4.3822117, -7.1887221, -4.3961544, -2.7987242, 2.8065104
2: -8.7316389, -6.1408548, -8.6829739, -6.1531687, -2.5784702, 2.5421190
3: -5.0222411, -2.4216046, -5.0092278, -2.4303646, -2.5186977, 2.5006833
4: -7.9627166, -5.2706766, -7.9208684, -5.2885032, -2.1727519, 2.1515172
5: -6.3364148, -3.7178655, -6.3123989, -3.7658811, -2.4386559, 2.4466786
6: -14.4118204, -10.9732723, -14.3979731, -11.0174685, -2.5745296, 2.6198571
7: 2.2632289, 4.8369155, 2.3113546, 4.8247833, -2.0830951, 2.0641267
8: -1.3313003, 0.9773989, -1.3185802, 0.9726424, -2.1801758, 2.1539295
9: -8.8149891, -5.7168012, -8.8003502, -5.7219272, -2.4616919, 2.4756825

Time for backsubstitution: 14.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 6192
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6135

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of IS_B2_B1_B1

### Relational analysis result of IS_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4840913, upper bound: 1.4656577
time: 4.05 seconds

## Relational analysis of IS_B2_B1_B2

### Relational analysis result of IS_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4840913, upper bound: 1.4685640
time: 3.85 seconds

## BFS IS instance: IS_B2_B2

### Backsubstitution after applying IS history:
0: -15.8640556, -11.6210804, -15.8640432, -11.6210861, -2.9997602, 2.9801774
1: -7.1961079, -4.3803363, -7.1961031, -4.3803391, -2.8157687, 2.8157668
2: -8.7408409, -6.1397600, -8.7408390, -6.1397667, -2.6010742, 2.6010790
3: -5.0245595, -2.4202366, -5.0245476, -2.4202385, -2.5323877, 2.5165443
4: -7.9703813, -5.2681599, -7.9703741, -5.2681651, -2.1964855, 2.1794057
5: -6.3388715, -3.7086442, -6.3388648, -3.7086492, -2.4806614, 2.4796729
6: -14.4134359, -10.9648418, -14.4134331, -10.9648571, -2.6285615, 2.6621218
7: 2.2540903, 4.8381100, 2.2540984, 4.8381004, -2.1051314, 2.0920794
8: -1.3332825, 0.9782434, -1.3332634, 0.9782405, -2.1866693, 2.1700795
9: -8.8183250, -5.7160292, -8.8183212, -5.7160392, -2.4725437, 2.5043428

Time for backsubstitution: 14.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6192
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6156
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of IS_B2_B2_B1

### Relational analysis result of IS_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4840913, upper bound: 1.4811922
time: 4.06 seconds

## Relational analysis of IS_B2_B2_B2

### Relational analysis result of IS_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4840912, upper bound: 1.4840928
time: 3.74 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 22.35 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 22.35
Output dim: 7, lower bound: -1.4426295, upper bound: 1.4583076
IS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 22.35
Output dim: 7, lower bound: -1.4583031, upper bound: 1.4583073
IS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 22.35
Output dim: 7, lower bound: -1.4426295, upper bound: 1.4777025
IS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 22.35
Output dim: 7, lower bound: -1.4583031, upper bound: 1.4777022
IS_B2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 22.35
Output dim: 7, lower bound: -1.4840913, upper bound: 1.4656577
IS_B2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 22.35
Output dim: 7, lower bound: -1.4840913, upper bound: 1.4685640
IS_B2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 22.35
Output dim: 7, lower bound: -1.4840913, upper bound: 1.4811922
IS_B2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 22.35
Output dim: 7, lower bound: -1.4840912, upper bound: 1.4840928

## BFS IS instance: IS_B1_A1_A1

### Backsubstitution after applying IS history:
0: -15.7471771, -11.7135725, -15.7666759, -11.6646318, -2.8706307, 2.8372130
1: -7.1517663, -4.4234872, -7.1579418, -4.4095078, -2.7422585, 2.7344546
2: -8.6628361, -6.1823931, -8.7114906, -6.1701326, -2.4927034, 2.5290976
3: -4.9438305, -2.4636350, -4.9569860, -2.4550014, -2.3976541, 2.3992839
4: -7.9020762, -5.3117576, -7.9440341, -5.2939811, -2.0993600, 2.1261935
5: -6.2378035, -3.8075852, -6.2616954, -3.7595906, -2.3134995, 2.2831302
6: -14.3730278, -11.0555315, -14.3869114, -11.0112276, -2.5694752, 2.5381305
7: 2.3454614, 4.7627077, 2.2973518, 4.7747555, -1.9778919, 2.0126951
8: -1.2102590, 0.9327888, -1.2229581, 0.9375672, -2.0276728, 2.0359483
9: -8.7705936, -5.7961316, -8.7848368, -5.7909350, -2.3721881, 2.3838511

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_B1_A1_A1_A1

### Relational analysis result of IS_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4396648, upper bound: 1.4582955
time: 4.64 seconds

## Relational analysis of IS_B1_A1_A1_A2

### Relational analysis result of IS_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4426217, upper bound: 1.4582951
time: 4.86 seconds

## BFS IS instance: IS_B1_A1_A2

### Backsubstitution after applying IS history:
0: -15.7687864, -11.6553373, -15.7687855, -11.6553326, -2.9012737, 2.8785594
1: -7.1591148, -4.4076223, -7.1591153, -4.4076223, -2.7514925, 2.7514930
2: -8.7206879, -6.1691084, -8.7206879, -6.1691089, -2.5515790, 2.5515795
3: -4.9593015, -2.4537759, -4.9593029, -2.4537749, -2.4135542, 2.4127510
4: -7.9517260, -5.2917271, -7.9517303, -5.2917261, -2.1270857, 2.1495891
5: -6.2641282, -3.7503965, -6.2641287, -3.7503946, -2.3463931, 2.3250854
6: -14.3885374, -11.0028000, -14.3885374, -11.0027924, -2.6117396, 2.5922632
7: 2.2882261, 4.7757874, 2.2882204, 4.7757864, -2.0055544, 2.0347397
8: -1.2247832, 0.9384151, -1.2247856, 0.9384165, -2.0435863, 2.0425243
9: -8.7875395, -5.7901487, -8.7875423, -5.7901473, -2.3998265, 2.3939879

Time for backsubstitution: 14.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_B1_A1_A2_A1

### Relational analysis result of IS_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4553194, upper bound: 1.4582955
time: 4.84 seconds

## Relational analysis of IS_B1_A1_A2_A2

### Relational analysis result of IS_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4582952, upper bound: 1.4582954
time: 4.29 seconds

## BFS IS instance: IS_B1_A2_A1

### Backsubstitution after applying IS history:
0: -15.8424368, -11.6793060, -15.7666759, -11.6646318, -2.9208164, 2.8619919
1: -7.1887221, -4.3961544, -7.1579418, -4.4095078, -2.7792144, 2.7617874
2: -8.6829739, -6.1531687, -8.7114906, -6.1701326, -2.5128412, 2.5583220
3: -5.0092278, -2.4303646, -4.9569860, -2.4550014, -2.4543784, 2.4324892
4: -7.9208684, -5.2885032, -7.9440341, -5.2939811, -2.1200094, 2.1485009
5: -6.3123989, -3.7658811, -6.2616954, -3.7595906, -2.3831894, 2.3158293
6: -14.3979731, -11.0174685, -14.3869114, -11.0112276, -2.5815892, 2.5764949
7: 2.3113546, 4.8247833, 2.2973518, 4.7747555, -2.0011568, 2.0433438
8: -1.3185802, 0.9726424, -1.2229581, 0.9375672, -2.0963597, 2.0720041
9: -8.8003502, -5.7219272, -8.7848368, -5.7909350, -2.4013386, 2.4426863

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of IS_B1_A2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4426177, upper bound: 1.4747143
time: 4.80 seconds

## Relational analysis of IS_B1_A2_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4426176, upper bound: 1.4776888
time: 4.45 seconds

## BFS IS instance: IS_B1_A2_A2

### Backsubstitution after applying IS history:
0: -15.8640432, -11.6210861, -15.7687855, -11.6553326, -2.9727454, 2.9033544
1: -7.1961031, -4.3803391, -7.1591153, -4.4076223, -2.7884808, 2.7787762
2: -8.7408390, -6.1397667, -8.7206879, -6.1691089, -2.5717301, 2.5809212
3: -5.0245476, -2.4202385, -4.9593029, -2.4537749, -2.4700077, 2.4462059
4: -7.9703741, -5.2681651, -7.9517303, -5.2917261, -2.1476336, 2.1748009
5: -6.3388648, -3.7086492, -6.2641287, -3.7503946, -2.4322414, 2.3578157
6: -14.4134331, -10.9648571, -14.3885374, -11.0027924, -2.6238661, 2.6305115
7: 2.2540984, 4.8381004, 2.2882204, 4.7757864, -2.0289531, 2.0840487
8: -1.3332634, 0.9782405, -1.2247856, 0.9384165, -2.1124043, 2.0783665
9: -8.8183212, -5.7160392, -8.7875423, -5.7901473, -2.4301276, 2.4527590

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6156
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of IS_B1_A2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4582911, upper bound: 1.4747140
time: 4.85 seconds

## Relational analysis of IS_B1_A2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4582911, upper bound: 1.4776894
time: 4.48 seconds

## BFS IS instance: IS_B2_B1_B1

### Backsubstitution after applying IS history:
0: -15.8614044, -11.6315165, -15.8401070, -11.6838961, -2.9464359, 2.9440694
1: -7.1892219, -4.3825088, -7.1648588, -4.3974371, -2.7917848, 2.7823501
2: -8.7311220, -6.1426806, -8.6807852, -6.1608610, -2.5702610, 2.5381045
3: -5.0216112, -2.4231505, -5.0064750, -2.4368744, -2.5118408, 2.4964051
4: -7.9618807, -5.2719831, -7.9171815, -5.2940030, -2.1616096, 2.1453209
5: -6.3360338, -3.7182517, -6.3106971, -3.7675221, -2.4360571, 2.4444923
6: -14.4099369, -10.9734278, -14.3900700, -11.0181313, -2.5706306, 2.6069355
7: 2.2645574, 4.8364434, 2.3169608, 4.8227139, -2.0794573, 2.0579045
8: -1.3308215, 0.9770975, -1.3165359, 0.9713130, -2.1750600, 2.1505225
9: -8.8107986, -5.7169118, -8.7826796, -5.7224026, -2.4566836, 2.4566166

Time for backsubstitution: 14.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6192
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6135

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6192

## Relational analysis of IS_B2_B1_B1_A1

### Relational analysis result of IS_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4582913, upper bound: 1.4396608
time: 5.28 seconds

## Relational analysis of IS_B2_B1_B1_A2

### Relational analysis result of IS_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4582913, upper bound: 1.4656581
time: 4.28 seconds

## BFS IS instance: IS_B2_B1_B2

### Backsubstitution after applying IS history:
0: -15.8619423, -11.6303787, -15.8610678, -11.6690998, -2.9715614, 2.9635441
1: -7.1948566, -4.3822126, -7.1942043, -4.3619776, -2.8328791, 2.8119917
2: -8.7316370, -6.1408582, -8.6974258, -6.1486645, -2.5829725, 2.5565677
3: -5.0222387, -2.4216115, -5.0340114, -2.4264865, -2.5235591, 2.5257740
4: -7.9627137, -5.2706795, -7.9423451, -5.2871876, -2.1762128, 2.1736944
5: -6.3364139, -3.7178664, -6.3315430, -3.7617269, -2.4461064, 2.4683743
6: -14.4118118, -10.9732723, -14.4078188, -11.0013638, -2.5911627, 2.6306224
7: 2.2632346, 4.8369145, 2.3073292, 4.8407097, -2.0903263, 2.0671480
8: -1.3312998, 0.9773974, -1.3247399, 0.9831514, -2.1868010, 2.1638596
9: -8.8149738, -5.7168026, -8.8070440, -5.6971583, -2.4864650, 2.4783721

Time for backsubstitution: 14.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6192
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6192

## Relational analysis of IS_B2_B1_B2_A1

### Relational analysis result of IS_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4582914, upper bound: 1.4426176
time: 4.39 seconds

## Relational analysis of IS_B2_B1_B2_A2

### Relational analysis result of IS_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4582914, upper bound: 1.4426174
time: 4.73 seconds

## BFS IS instance: IS_B2_B2_B1

### Backsubstitution after applying IS history:
0: -15.8635159, -11.6222181, -15.8617249, -11.6256790, -2.9877987, 2.9749851
1: -7.1904516, -4.3806314, -7.1722288, -4.3816137, -2.8088379, 2.7915974
2: -8.7403240, -6.1415844, -8.7386494, -6.1474504, -2.5928736, 2.5970650
3: -5.0239320, -2.4217820, -5.0217924, -2.4267564, -2.5254960, 2.5122204
4: -7.9695525, -5.2694645, -7.9667277, -5.2736559, -2.1853495, 2.1732039
5: -6.3384914, -3.7090316, -6.3371916, -3.7102938, -2.4780583, 2.4774966
6: -14.4115524, -10.9649973, -14.4055328, -10.9655285, -2.6246443, 2.6492114
7: 2.2554202, 4.8376393, 2.2597084, 4.8360305, -2.1019416, 2.0858524
8: -1.3328013, 0.9779429, -1.3312263, 0.9769092, -2.1815557, 2.1666780
9: -8.8141394, -5.7161393, -8.8006802, -5.7165184, -2.4675288, 2.4852695

Time for backsubstitution: 14.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6192
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6156
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6192

## Relational analysis of IS_B2_B2_B1_A1

### Relational analysis result of IS_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4582914, upper bound: 1.4553155
time: 4.60 seconds

## Relational analysis of IS_B2_B2_B1_A2

### Relational analysis result of IS_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4582913, upper bound: 1.4553152
time: 4.61 seconds

## BFS IS instance: IS_B2_B2_B2

### Backsubstitution after applying IS history:
0: -15.8640537, -11.6210833, -15.8826771, -11.6108828, -3.0129213, 3.0017910
1: -7.1960878, -4.3803368, -7.2019954, -4.3461905, -2.8498974, 2.8216586
2: -8.7408390, -6.1397619, -8.7552652, -6.1349716, -2.6058674, 2.6155033
3: -5.0245576, -2.4202435, -5.0493255, -2.4158285, -2.5377431, 2.5414186
4: -7.9703779, -5.2681623, -7.9919052, -5.2657213, -2.2010617, 2.2016063
5: -6.3388700, -3.7086461, -6.3580275, -3.7044063, -2.4881992, 2.5013800
6: -14.4134283, -10.9648438, -14.4232950, -10.9486589, -2.6453128, 2.6770880
7: 2.2540951, 4.8381090, 2.2500930, 4.8547006, -2.1237221, 2.0951352
8: -1.3332787, 0.9782414, -1.3401041, 0.9887376, -2.1933072, 2.1805253
9: -8.8183117, -5.7160273, -8.8274460, -5.6912746, -2.4973121, 2.5101969

Time for backsubstitution: 14.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6192
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6192

## Relational analysis of IS_B2_B2_B2_A1

### Relational analysis result of IS_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4582914, upper bound: 1.4582914
time: 4.56 seconds

## Relational analysis of IS_B2_B2_B2_A2

### Relational analysis result of IS_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4582914, upper bound: 1.4840916
time: 4.70 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 23.68 seconds
IS_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 23.68
Output dim: 7, lower bound: -1.4396648, upper bound: 1.4582955
IS_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 23.68
Output dim: 7, lower bound: -1.4426217, upper bound: 1.4582951
IS_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 23.68
Output dim: 7, lower bound: -1.4553194, upper bound: 1.4582955
IS_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 23.68
Output dim: 7, lower bound: -1.4582952, upper bound: 1.4582954
IS_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.68
Output dim: 7, lower bound: -1.4426177, upper bound: 1.4747143
IS_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.68
Output dim: 7, lower bound: -1.4426176, upper bound: 1.4776888
IS_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.68
Output dim: 7, lower bound: -1.4582911, upper bound: 1.4747140
IS_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.68
Output dim: 7, lower bound: -1.4582911, upper bound: 1.4776894
IS_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 23.68
Output dim: 7, lower bound: -1.4582913, upper bound: 1.4396608
IS_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 23.68
Output dim: 7, lower bound: -1.4582913, upper bound: 1.4656581
IS_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 23.68
Output dim: 7, lower bound: -1.4582914, upper bound: 1.4426176
IS_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 23.68
Output dim: 7, lower bound: -1.4582914, upper bound: 1.4426174
IS_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 23.68
Output dim: 7, lower bound: -1.4582914, upper bound: 1.4553155
IS_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 23.68
Output dim: 7, lower bound: -1.4582913, upper bound: 1.4553152
IS_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 23.68
Output dim: 7, lower bound: -1.4582914, upper bound: 1.4582914
IS_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 23.68
Output dim: 7, lower bound: -1.4582914, upper bound: 1.4840916

## BFS IS instance: IS_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -15.7448359, -11.7181492, -15.7661333, -11.6657667, -2.8653984, 2.8253360
1: -7.1277256, -4.4247828, -7.1522436, -4.4098063, -2.7179193, 2.7274609
2: -8.6607752, -6.1901503, -8.7110023, -6.1719756, -2.4887996, 2.5208521
3: -4.9411459, -2.4700933, -4.9563637, -2.4565370, -2.3935094, 2.3924794
4: -7.8984318, -5.3172574, -7.9432058, -5.2952871, -2.0932078, 2.1150522
5: -6.2361145, -3.8090811, -6.2613168, -3.7599425, -2.3113518, 2.2806435
6: -14.3650723, -11.0561886, -14.3850193, -11.0113811, -2.5567021, 2.5342577
7: 2.3510985, 4.7606430, 2.2986898, 4.7742844, -1.9716578, 2.0090075
8: -1.2082124, 0.9315515, -1.2224782, 0.9372883, -2.0243216, 2.0309751
9: -8.7529106, -5.7965918, -8.7806454, -5.7910447, -2.3532081, 2.3788695

Time for backsubstitution: 14.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6140

## Relational analysis of IS_B1_A1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4350625, upper bound: 1.4581834
time: 4.83 seconds

## Relational analysis of IS_B1_A1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4396606, upper bound: 1.4582915
time: 4.50 seconds

## BFS IS instance: IS_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -15.7657652, -11.7032833, -15.7666759, -11.6646318, -2.8843908, 2.8503282
1: -7.1570969, -4.3893542, -7.1579204, -4.4095082, -2.7475886, 2.7685661
2: -8.6773396, -6.1779752, -8.7114878, -6.1701355, -2.5072041, 2.5335126
3: -4.9682646, -2.4596438, -4.9569850, -2.4550076, -2.4225373, 2.4042215
4: -7.9235544, -5.3104215, -7.9440308, -5.2939830, -2.1215200, 2.1296759
5: -6.2569757, -3.8033450, -6.2616940, -3.7595911, -2.3352070, 2.2906201
6: -14.3832054, -11.0394630, -14.3869066, -11.0112276, -2.5843558, 2.5546880
7: 2.3412399, 4.7784982, 2.2973585, 4.7747536, -1.9809802, 2.0198283
8: -1.2162504, 0.9432511, -1.2229562, 0.9375668, -2.0372839, 2.0495608
9: -8.7773228, -5.7714376, -8.7848215, -5.7909355, -2.3747420, 2.4084797

Time for backsubstitution: 14.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6140

## Relational analysis of IS_B1_A1_A1_A2_A1

### Relational analysis result of IS_B1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4424989, upper bound: 1.4537122
time: 4.71 seconds

## Relational analysis of IS_B1_A1_A1_A2_A2

### Relational analysis result of IS_B1_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4426177, upper bound: 1.4582912
time: 4.84 seconds

## BFS IS instance: IS_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -15.7664652, -11.6599178, -15.7682419, -11.6564684, -2.8960338, 2.8666825
1: -7.1350632, -4.4089088, -7.1534162, -4.4079208, -2.7271423, 2.7445073
2: -8.7186270, -6.1768579, -8.7202015, -6.1709490, -2.5476780, 2.5433435
3: -4.9565759, -2.4602401, -4.9586806, -2.4553075, -2.4092765, 2.4059210
4: -7.9481206, -5.2972245, -7.9509091, -5.2930317, -2.1209278, 2.1384563
5: -6.2624664, -3.7518969, -6.2637520, -3.7507455, -2.3442516, 2.3225822
6: -14.3805895, -11.0034618, -14.3866444, -11.0029469, -2.5989771, 2.5883729
7: 2.2938638, 4.7737260, 2.2895570, 4.7753177, -1.9993200, 2.0315523
8: -1.2227454, 0.9371762, -1.2243047, 0.9381375, -2.0401926, 2.0375533
9: -8.7698803, -5.7906108, -8.7833490, -5.7902551, -2.3808484, 2.3890100

Time for backsubstitution: 14.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of IS_B1_A1_A2_A1_B1

### Relational analysis result of IS_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4553182, upper bound: 1.4553201
time: 4.24 seconds

## Relational analysis of IS_B1_A1_A2_A1_B2

### Relational analysis result of IS_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4553183, upper bound: 1.4582951
time: 4.72 seconds

## BFS IS instance: IS_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -15.7873783, -11.6450577, -15.7687836, -11.6553345, -2.9225259, 2.8916740
1: -7.1647310, -4.3735185, -7.1590924, -4.4076233, -2.7571077, 2.7855740
2: -8.7351627, -6.1643953, -8.7206860, -6.1691113, -2.5660515, 2.5562906
3: -4.9836636, -2.4492617, -4.9593019, -2.4537811, -2.4382243, 2.4181404
4: -7.9732809, -5.2892504, -7.9517269, -5.2917290, -2.1492691, 2.1541972
5: -6.2833161, -3.7460704, -6.2641287, -3.7503943, -2.3681111, 2.3326495
6: -14.3987312, -10.9866419, -14.3885298, -11.0027924, -2.6267748, 2.6089392
7: 2.2840223, 4.7922583, 2.2882257, 4.7757864, -2.0086930, 2.0532286
8: -1.2314618, 0.9488649, -1.2247832, 0.9384155, -2.0537853, 2.0561478
9: -8.7966642, -5.7654538, -8.7875261, -5.7901483, -2.4054751, 2.4186654

Time for backsubstitution: 14.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6140

## Relational analysis of IS_B1_A1_A2_A2_A1

### Relational analysis result of IS_B1_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4581830, upper bound: 1.4537124
time: 4.46 seconds

## Relational analysis of IS_B1_A1_A2_A2_A2

### Relational analysis result of IS_B1_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4582913, upper bound: 1.4582910
time: 4.72 seconds

## BFS IS instance: IS_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -15.8418951, -11.6804466, -15.7643518, -11.6692095, -2.9089227, 2.8568006
1: -7.1830692, -4.3964529, -7.1338921, -4.4107928, -2.7722764, 2.7374392
2: -8.6824570, -6.1549969, -8.7094278, -6.1778827, -2.5045743, 2.5544310
3: -5.0086002, -2.4319091, -4.9542613, -2.4614668, -2.4475303, 2.4282064
4: -7.9200339, -5.2898102, -7.9404216, -5.2994776, -2.1088810, 2.1410704
5: -6.3120131, -3.7662673, -6.2600279, -3.7610884, -2.3806977, 2.3136535
6: -14.3960896, -11.0176191, -14.3789654, -11.0118885, -2.5776658, 2.5637312
7: 2.3126826, 4.8243127, 2.3029914, 4.7726955, -1.9979744, 2.0371404
8: -1.3180990, 0.9723420, -1.2209191, 0.9363284, -2.0913236, 2.0682592
9: -8.7961550, -5.7220383, -8.7671757, -5.7913976, -2.3963389, 2.4236736

Time for backsubstitution: 14.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6140

## Relational analysis of IS_B1_A2_A1_B1_B1

### Relational analysis result of IS_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4380023, upper bound: 1.4746071
time: 4.66 seconds

## Relational analysis of IS_B1_A2_A1_B1_B2

### Relational analysis result of IS_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4426133, upper bound: 1.4747103
time: 4.51 seconds

## BFS IS instance: IS_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -15.8424377, -11.6793118, -15.7852468, -11.6543455, -2.9258366, 2.8834929
1: -7.1886992, -4.3961549, -7.1632710, -4.3753986, -2.8133006, 2.7671161
2: -8.6829720, -6.1531720, -8.7259712, -6.1657157, -2.5172563, 2.5727992
3: -5.0092263, -2.4303706, -4.9813166, -2.4509797, -2.4593611, 2.4549651
4: -7.9208660, -5.2885051, -7.9655647, -5.2926455, -2.1234760, 2.1570835
5: -6.3123989, -3.7658827, -6.2808781, -3.7553525, -2.3885674, 2.3375497
6: -14.3979654, -11.0174685, -14.3971100, -10.9951420, -2.5879316, 2.5866473
7: 2.3113599, 4.8247828, 2.2931471, 4.7905426, -2.0190182, 2.0466588
8: -1.3185782, 0.9726424, -1.2289400, 0.9480176, -2.1030116, 2.0772543
9: -8.8003330, -5.7219267, -8.7915506, -5.7662430, -2.4109392, 2.4433973

Time for backsubstitution: 14.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6140

## Relational analysis of IS_B1_A2_A1_B2_B1

### Relational analysis result of IS_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4380023, upper bound: 1.4775826
time: 4.24 seconds

## Relational analysis of IS_B1_A2_A1_B2_B2

### Relational analysis result of IS_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4426134, upper bound: 1.4776866
time: 4.17 seconds

## BFS IS instance: IS_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -15.8635063, -11.6222296, -15.7664652, -11.6599112, -2.9608593, 2.8981657
1: -7.1904459, -4.3806353, -7.1350632, -4.4089079, -2.7815380, 2.7544279
2: -8.7403202, -6.1415920, -8.7186260, -6.1768579, -2.5634623, 2.5770340
3: -5.0239186, -2.4217849, -4.9565763, -2.4602399, -2.4631548, 2.4419136
4: -7.9695458, -5.2694697, -7.9481225, -5.2972217, -2.1365042, 2.1686423
5: -6.3384867, -3.7090368, -6.2624669, -3.7518940, -2.4297342, 2.3556402
6: -14.4115524, -10.9650116, -14.3805923, -11.0034552, -2.6199436, 2.6177452
7: 2.2554283, 4.8376288, 2.2938571, 4.7737265, -2.0257688, 2.0778382
8: -1.3327827, 0.9779401, -1.2227449, 0.9371772, -2.1073675, 2.0746214
9: -8.8141356, -5.7161522, -8.7698841, -5.7906089, -2.4251289, 2.4337382

Time for backsubstitution: 14.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6156
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6140

## Relational analysis of IS_B1_A2_A2_B1_A1

### Relational analysis result of IS_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4581790, upper bound: 1.4701606
time: 4.21 seconds

## Relational analysis of IS_B1_A2_A2_B1_A2

### Relational analysis result of IS_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4582872, upper bound: 1.4747101
time: 4.97 seconds

## BFS IS instance: IS_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -15.8640442, -11.6210899, -15.7873802, -11.6450481, -2.9777627, 2.9248843
1: -7.1960802, -4.3803411, -7.1647320, -4.3735170, -2.8225632, 2.7843909
2: -8.7408371, -6.1397700, -8.7351646, -6.1643968, -2.5764403, 2.5953946
3: -5.0245452, -2.4202442, -4.9836655, -2.4492612, -2.4754257, 2.4681063
4: -7.9703736, -5.2681665, -7.9732842, -5.2892489, -2.1522312, 2.1964583
5: -6.3388638, -3.7086520, -6.2833171, -3.7460666, -2.4376392, 2.3795342
6: -14.4134283, -10.9648561, -14.3987322, -10.9866371, -2.6371617, 2.6400590
7: 2.2541037, 4.8380995, 2.2840161, 4.7922587, -2.0474427, 2.0874004
8: -1.3332615, 0.9782395, -1.2314627, 0.9488640, -2.1190443, 2.0841305
9: -8.8183088, -5.7160392, -8.7966671, -5.7654543, -2.4380245, 2.4566057

Time for backsubstitution: 14.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6156
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6140

## Relational analysis of IS_B1_A2_A2_B2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4581790, upper bound: 1.4731186
time: 5.84 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4582872, upper bound: 1.4776848
time: 4.73 seconds

## BFS IS instance: IS_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -15.7661333, -11.6657667, -15.8401070, -11.6838961, -2.8500309, 2.9144673
1: -7.1522436, -4.4098063, -7.1648588, -4.3974371, -2.7548065, 2.7550526
2: -8.7110023, -6.1719756, -8.6807852, -6.1608610, -2.5501413, 2.5088096
3: -4.9563637, -2.4565370, -5.0064750, -2.4368744, -2.4256401, 2.4496453
4: -7.9432058, -5.2952871, -7.9171815, -5.2940030, -2.1373258, 2.1138155
5: -6.2613168, -3.7599425, -6.3106971, -3.7675221, -2.3131871, 2.3809991
6: -14.3850193, -11.0113811, -14.3900700, -11.0181313, -2.5726094, 2.5686700
7: 2.2986898, 4.7742844, 2.3169608, 4.8227139, -2.0396633, 1.9949360
8: -1.2224782, 0.9372883, -1.3165359, 0.9713130, -2.0668886, 2.0925813
9: -8.7806454, -5.7910447, -8.7826796, -5.7224026, -2.4355931, 2.3822715

Time for backsubstitution: 14.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6140

## Relational analysis of IS_B2_B1_B1_A1_B1

### Relational analysis result of IS_B2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4537087, upper bound: 1.4395384
time: 4.46 seconds

## Relational analysis of IS_B2_B1_B1_A1_B2

### Relational analysis result of IS_B2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4582872, upper bound: 1.4396567
time: 4.85 seconds

## BFS IS instance: IS_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -15.8613901, -11.6315155, -15.8401070, -11.6838961, -2.9041381, 2.9434018
1: -7.1892176, -4.3825111, -7.1648588, -4.3974371, -2.7917805, 2.7823477
2: -8.7311192, -6.1426883, -8.6807852, -6.1608610, -2.5702581, 2.5380969
3: -5.0216007, -2.4231524, -5.0064750, -2.4368744, -2.5118337, 2.5130434
4: -7.9618807, -5.2719865, -7.9171815, -5.2940030, -2.1670308, 2.1453180
5: -6.3360310, -3.7182541, -6.3106971, -3.7675221, -2.4360538, 2.4653344
6: -14.4099379, -10.9734364, -14.3900700, -11.0181313, -2.5706277, 2.5928175
7: 2.2645607, 4.8364334, 2.3169608, 4.8227139, -2.0744648, 2.0417690
8: -1.3308053, 0.9770975, -1.3165359, 0.9713130, -2.1572580, 2.1505208
9: -8.8107977, -5.7169228, -8.7826796, -5.7224026, -2.4566827, 2.4306538

Time for backsubstitution: 14.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6140

## Relational analysis of IS_B2_B1_B1_A2_A1

### Relational analysis result of IS_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4581792, upper bound: 1.4611143
time: 4.19 seconds

## Relational analysis of IS_B2_B1_B1_A2_A2

### Relational analysis result of IS_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4582875, upper bound: 1.4396563
time: 4.77 seconds

## BFS IS instance: IS_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -15.7666759, -11.6646318, -15.8610678, -11.6690998, -2.8751564, 2.9281704
1: -7.1579204, -4.4095082, -7.1942043, -4.3619776, -2.7959428, 2.7846961
2: -8.7114878, -6.1701355, -8.6974258, -6.1486645, -2.5628233, 2.5272903
3: -4.9569850, -2.4550076, -5.0340114, -2.4264865, -2.4373507, 2.4666328
4: -7.9440308, -5.2939830, -7.9423451, -5.2871876, -2.1462371, 2.1421902
5: -6.2616940, -3.7595911, -6.3315430, -3.7617269, -2.3233213, 2.3961358
6: -14.3869066, -11.0112276, -14.4078188, -11.0013638, -2.5814919, 2.5927165
7: 2.2973585, 4.7747536, 2.3073292, 4.8407097, -2.0504572, 2.0041790
8: -1.2229562, 0.9375668, -1.3247399, 0.9831514, -2.0786295, 2.1016462
9: -8.7848215, -5.7909355, -8.8070440, -5.6971583, -2.4446721, 2.4040298

Time for backsubstitution: 14.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6140

## Relational analysis of IS_B2_B1_B2_A1_B1

### Relational analysis result of IS_B2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4537087, upper bound: 1.4424948
time: 4.54 seconds

## Relational analysis of IS_B2_B1_B2_A1_B2

### Relational analysis result of IS_B2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4582872, upper bound: 1.4426136
time: 5.19 seconds

## BFS IS instance: IS_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -15.8619289, -11.6303797, -15.8610678, -11.6690998, -2.9292631, 2.9573867
1: -7.1948519, -4.3822155, -7.1942043, -4.3619776, -2.8328743, 2.8119888
2: -8.7316360, -6.1408644, -8.6974258, -6.1486645, -2.5829716, 2.5565615
3: -5.0222292, -2.4216139, -5.0340114, -2.4264865, -2.5235515, 2.5366380
4: -7.9627147, -5.2706833, -7.9423451, -5.2871876, -2.1795006, 2.1736925
5: -6.3364100, -3.7178686, -6.3315430, -3.7617269, -2.4461021, 2.4804749
6: -14.4118109, -10.9732809, -14.4078188, -11.0013638, -2.5911589, 2.6208334
7: 2.2632356, 4.8369036, 2.3073292, 4.8407097, -2.0853281, 2.0510132
8: -1.3312826, 0.9773965, -1.3247399, 0.9831514, -2.1745768, 2.1638587
9: -8.8149729, -5.7168117, -8.8070440, -5.6971583, -2.4789867, 2.4524102

Time for backsubstitution: 14.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6140

## Relational analysis of IS_B2_B1_B2_A2_B1

### Relational analysis result of IS_B2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4537086, upper bound: 1.4424947
time: 5.15 seconds

## Relational analysis of IS_B2_B1_B2_A2_B2

### Relational analysis result of IS_B2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4582872, upper bound: 1.4685604
time: 4.82 seconds

## BFS IS instance: IS_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -15.7682419, -11.6564684, -15.8617249, -11.6256790, -2.8913946, 2.9664245
1: -7.1534162, -4.4079208, -7.1722288, -4.3816137, -2.7718024, 2.7643080
2: -8.7202015, -6.1709490, -8.7386494, -6.1474504, -2.5727510, 2.5677004
3: -4.9586806, -2.4553075, -5.0217924, -2.4267564, -2.4393215, 2.4652383
4: -7.9509091, -5.2930317, -7.9667277, -5.2736559, -2.1636643, 2.1414335
5: -6.2637520, -3.7507455, -6.3371916, -3.7102938, -2.3551679, 2.4300528
6: -14.3866444, -11.0029469, -14.4055328, -10.9655285, -2.6264958, 2.6109586
7: 2.2895570, 4.7753177, 2.2597084, 4.8360305, -2.0803540, 2.0227270
8: -1.2243047, 0.9381375, -1.3312263, 0.9769092, -2.0732536, 2.1086307
9: -8.7833490, -5.7902551, -8.8006802, -5.7165184, -2.4456587, 2.4110572

Time for backsubstitution: 14.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6156
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_B2_B2_B1_A1_A1

### Relational analysis result of IS_B2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4553145, upper bound: 1.4553151
time: 6.07 seconds

## Relational analysis of IS_B2_B2_B1_A1_A2

### Relational analysis result of IS_B2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4553145, upper bound: 1.4553155
time: 4.65 seconds

## BFS IS instance: IS_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -15.8635044, -11.6222200, -15.8617249, -11.6256790, -2.9455013, 2.9749846
1: -7.1904469, -4.3806357, -7.1722288, -4.3816137, -2.8088331, 2.7915931
2: -8.7403202, -6.1415920, -8.7386494, -6.1474504, -2.5928698, 2.5970573
3: -5.0239205, -2.4217844, -5.0217924, -2.4267564, -2.5254893, 2.5288587
4: -7.9695520, -5.2694678, -7.9667277, -5.2736559, -2.1907711, 2.1732018
5: -6.3384886, -3.7090337, -6.3371916, -3.7102938, -2.4780545, 2.4997880
6: -14.4115505, -10.9650078, -14.4055328, -10.9655285, -2.6246405, 2.6352072
7: 2.2554231, 4.8376293, 2.2597084, 4.8360305, -2.1019402, 2.0697169
8: -1.3327847, 0.9779406, -1.3312263, 0.9769092, -2.1640043, 2.1666768
9: -8.8141384, -5.7161517, -8.8006802, -5.7165184, -2.4675279, 2.4593072

Time for backsubstitution: 14.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_B2_B2_B1_A2_A1

### Relational analysis result of IS_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4553146, upper bound: 1.4811925
time: 4.35 seconds

## Relational analysis of IS_B2_B2_B1_A2_A2

### Relational analysis result of IS_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4553146, upper bound: 1.4811921
time: 7.39 seconds

## BFS IS instance: IS_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -15.7687836, -11.6553345, -15.8826771, -11.6108828, -2.9165163, 2.9804261
1: -7.1590924, -4.4076233, -7.2019954, -4.3461905, -2.8129020, 2.7943721
2: -8.7206860, -6.1691113, -8.7552652, -6.1349716, -2.5857143, 2.5861540
3: -4.9593019, -2.4537811, -5.0493255, -2.4158285, -2.4515605, 2.4820113
4: -7.9517269, -5.2917290, -7.9919052, -5.2657213, -2.1793895, 2.1698372
5: -6.2641287, -3.7503943, -6.3580275, -3.7044063, -2.3653932, 2.4452362
6: -14.3885298, -11.0027924, -14.4232950, -10.9486589, -2.6350288, 2.6388314
7: 2.2882257, 4.7757864, 2.2500930, 4.8547006, -2.0917444, 2.0320094
8: -1.2247832, 0.9384155, -1.3401041, 0.9887376, -2.0850060, 2.1182163
9: -8.7875261, -5.7901483, -8.8274460, -5.6912746, -2.4547381, 2.4359860

Time for backsubstitution: 14.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6156
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6140

## Relational analysis of IS_B2_B2_B2_A1_B1

### Relational analysis result of IS_B2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4537086, upper bound: 1.4581790
time: 4.55 seconds

## Relational analysis of IS_B2_B2_B2_A1_B2

### Relational analysis result of IS_B2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4582873, upper bound: 1.4582872
time: 4.58 seconds

## BFS IS instance: IS_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -15.8640432, -11.6210823, -15.8826771, -11.6108828, -2.9706225, 3.0017905
1: -7.1960812, -4.3803406, -7.2019954, -4.3461905, -2.8498907, 2.8216548
2: -8.7408371, -6.1397696, -8.7552652, -6.1349716, -2.6058655, 2.6154957
3: -5.0245461, -2.4202456, -5.0493255, -2.4158285, -2.5377345, 2.5520401
4: -7.9703770, -5.2681646, -7.9919052, -5.2657213, -2.2064834, 2.2016027
5: -6.3388653, -3.7086473, -6.3580275, -3.7044063, -2.4881954, 2.5236731
6: -14.4134274, -10.9648495, -14.4232950, -10.9486589, -2.6453080, 2.6633620
7: 2.2540979, 4.8380995, 2.2500930, 4.8547006, -2.1237202, 2.0789998
8: -1.3332620, 0.9782419, -1.3401041, 0.9887376, -2.1810834, 2.1805236
9: -8.8183098, -5.7160387, -8.8274460, -5.6912746, -2.4898181, 2.4842358

Time for backsubstitution: 14.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6140

## Relational analysis of IS_B2_B2_B2_A2_B1

### Relational analysis result of IS_B2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4537086, upper bound: 1.4581787
time: 7.69 seconds

## Relational analysis of IS_B2_B2_B2_A2_B2

### Relational analysis result of IS_B2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4582872, upper bound: 1.4582869
time: 4.45 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 26.44 seconds
IS_B1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 26.44
Output dim: 7, lower bound: -1.4350625, upper bound: 1.4581834
IS_B1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 26.44
Output dim: 7, lower bound: -1.4396606, upper bound: 1.4582915
IS_B1_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 26.44
Output dim: 7, lower bound: -1.4424989, upper bound: 1.4537122
IS_B1_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 26.44
Output dim: 7, lower bound: -1.4426177, upper bound: 1.4582912
IS_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 26.44
Output dim: 7, lower bound: -1.4553182, upper bound: 1.4553201
IS_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 26.44
Output dim: 7, lower bound: -1.4553183, upper bound: 1.4582951
IS_B1_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 26.44
Output dim: 7, lower bound: -1.4581830, upper bound: 1.4537124
IS_B1_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 26.44
Output dim: 7, lower bound: -1.4582913, upper bound: 1.4582910
IS_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 26.44
Output dim: 7, lower bound: -1.4380023, upper bound: 1.4746071
IS_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 26.44
Output dim: 7, lower bound: -1.4426133, upper bound: 1.4747103
IS_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 26.44
Output dim: 7, lower bound: -1.4380023, upper bound: 1.4775826
IS_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 26.44
Output dim: 7, lower bound: -1.4426134, upper bound: 1.4776866
IS_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 26.44
Output dim: 7, lower bound: -1.4581790, upper bound: 1.4701606
IS_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 26.44
Output dim: 7, lower bound: -1.4582872, upper bound: 1.4747101
IS_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 26.44
Output dim: 7, lower bound: -1.4581790, upper bound: 1.4731186
IS_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 26.44
Output dim: 7, lower bound: -1.4582872, upper bound: 1.4776848
IS_B2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 26.44
Output dim: 7, lower bound: -1.4537087, upper bound: 1.4395384
IS_B2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 26.44
Output dim: 7, lower bound: -1.4582872, upper bound: 1.4396567
IS_B2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 26.44
Output dim: 7, lower bound: -1.4581792, upper bound: 1.4611143
IS_B2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 26.44
Output dim: 7, lower bound: -1.4582875, upper bound: 1.4396563
IS_B2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 26.44
Output dim: 7, lower bound: -1.4537087, upper bound: 1.4424948
IS_B2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 26.44
Output dim: 7, lower bound: -1.4582872, upper bound: 1.4426136
IS_B2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 26.44
Output dim: 7, lower bound: -1.4537086, upper bound: 1.4424947
IS_B2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 26.44
Output dim: 7, lower bound: -1.4582872, upper bound: 1.4685604
IS_B2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 26.44
Output dim: 7, lower bound: -1.4553145, upper bound: 1.4553151
IS_B2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 26.44
Output dim: 7, lower bound: -1.4553145, upper bound: 1.4553155
IS_B2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 26.44
Output dim: 7, lower bound: -1.4553146, upper bound: 1.4811925
IS_B2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 26.44
Output dim: 7, lower bound: -1.4553146, upper bound: 1.4811921
IS_B2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 26.44
Output dim: 7, lower bound: -1.4537086, upper bound: 1.4581790
IS_B2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 26.44
Output dim: 7, lower bound: -1.4582873, upper bound: 1.4582872
IS_B2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 26.44
Output dim: 7, lower bound: -1.4537086, upper bound: 1.4581787
IS_B2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 26.44
Output dim: 7, lower bound: -1.4582872, upper bound: 1.4582869

## BFS IS instance: IS_B1_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -15.7447128, -11.7182713, -15.7645779, -11.6671963, -2.8634748, 2.8232813
1: -7.1274643, -4.4258947, -7.1490097, -4.4235020, -2.7039623, 2.7231150
2: -8.6602268, -6.1904626, -8.7044153, -6.1759005, -2.4843264, 2.5139527
3: -4.9399781, -2.4705195, -4.9420266, -2.4618199, -2.3868971, 2.3770361
4: -7.8980145, -5.3186760, -7.9379873, -5.3127270, -2.0740843, 2.1088607
5: -6.2355976, -3.8093300, -6.2548199, -3.7630548, -2.3071556, 2.2730513
6: -14.3639851, -11.0563726, -14.3716335, -11.0137949, -2.5537271, 2.5203276
7: 2.3514023, 4.7600675, 2.3024483, 4.7671547, -1.9642162, 2.0051646
8: -1.2078209, 0.9314442, -1.2173157, 0.9359264, -2.0178742, 2.0216558
9: -8.7517796, -5.7967582, -8.7668133, -5.7931266, -2.3498969, 2.3653038

Time for backsubstitution: 13.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6156
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6156

## Relational analysis of IS_B1_A1_A1_A1_B1_B1

### Relational analysis result of IS_B1_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4350624, upper bound: 1.4424988
time: 4.64 seconds

## Relational analysis of IS_B1_A1_A1_A1_B1_B2

### Relational analysis result of IS_B1_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4350625, upper bound: 1.4581834
time: 4.73 seconds

## BFS IS instance: IS_B1_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -15.7448387, -11.7181520, -15.7779331, -11.6636839, -2.8705950, 2.8409262
1: -7.1277218, -4.4247904, -7.1736875, -4.4001150, -2.7276068, 2.7488971
2: -8.6607714, -6.1901512, -8.7177181, -6.1494136, -2.5113578, 2.5275669
3: -4.9411330, -2.4700954, -4.9664993, -2.4177547, -2.4263618, 2.4004397
4: -7.8984299, -5.3172722, -8.0019970, -5.2908421, -2.0994229, 2.1453161
5: -6.2361102, -3.8090827, -6.2793055, -3.7328362, -2.3362103, 2.2986972
6: -14.3650665, -11.0561867, -14.3913441, -10.9654465, -2.5838287, 2.5394001
7: 2.3511014, 4.7606406, 2.2799315, 4.7778611, -1.9747314, 2.0213706
8: -1.2082086, 0.9315515, -1.2380209, 0.9405680, -2.0236754, 2.0546787
9: -8.7529030, -5.7965922, -8.7922220, -5.7550182, -2.3841763, 2.3977182

Time for backsubstitution: 12.64 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.121267795562744
rel_dist={7: [-1.4841364466020996, 1.4841360647073234]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6192
type: A, layer: 1, pos: 6192
type: B, layer: 1, pos: 6156
type: A, layer: 1, pos: 6156
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6192

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1392274, upper bound: 1.1486612
time: 4.71 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1530065, upper bound: 1.1530070
time: 4.97 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 9.87 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 9.87
Output dim: 7, lower bound: -1.1392274, upper bound: 1.1486612
IS_B2, status: Status.UNKNOWN, split count: 1, time: 9.87
Output dim: 7, lower bound: -1.1530065, upper bound: 1.1530070

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -15.8229198, -11.6252165, -15.7687855, -11.6553326, -2.5419102, 2.5087538
1: -7.1823936, -4.3909721, -7.1591153, -4.4076223, -2.7567616, 2.7428141
2: -8.7359295, -6.1509843, -8.7206879, -6.1691089, -2.4103079, 2.4203587
3: -4.9978518, -2.4275467, -4.9593029, -2.4537749, -2.2660737, 2.2625663
4: -7.9657664, -5.2775507, -7.9517303, -5.2917261, -1.9150314, 1.9103401
5: -6.3065300, -3.7179561, -6.2641287, -3.7503946, -2.2038503, 2.1778007
6: -14.4064779, -10.9802923, -14.3885374, -11.0027924, -2.2637844, 2.2771213
7: 2.2615671, 4.8117642, 2.2882204, 4.7757864, -1.8277831, 1.8453286
8: -1.2872524, 0.9737010, -1.2247856, 0.9384165, -1.9163160, 1.9006140
9: -8.8147469, -5.7478409, -8.7875423, -5.7901473, -2.1092038, 2.1245203

Time for backsubstitution: 14.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6156
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6192
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6156

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1301860, upper bound: 1.1486495
time: 4.79 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1392154, upper bound: 1.1486498
time: 4.82 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -15.8640537, -11.6210804, -15.8640451, -11.6210775, -2.6073208, 2.5575747
1: -7.1961083, -4.3803353, -7.1961031, -4.3803387, -2.7928877, 2.8105760
2: -8.7408409, -6.1397614, -8.7408390, -6.1397657, -2.4588299, 2.4694009
3: -5.0245576, -2.4202356, -5.0245490, -2.4202378, -2.3505244, 2.3351064
4: -7.9703808, -5.2681613, -7.9703798, -5.2681632, -1.9444685, 1.9494982
5: -6.3388705, -3.7086444, -6.3388667, -3.7086473, -2.3002243, 2.2795751
6: -14.4134331, -10.9648457, -14.4134312, -10.9648504, -2.2872505, 2.3056948
7: 2.2540908, 4.8381071, 2.2540932, 4.8381004, -1.8760800, 1.8950529
8: -1.3332782, 0.9782429, -1.3332644, 0.9782419, -2.0092335, 1.9913650
9: -8.8183250, -5.7160306, -8.8183231, -5.7160378, -2.1559525, 2.1864891

Time for backsubstitution: 14.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 6192

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6156

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1529946, upper bound: 1.1440046
time: 5.35 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1529948, upper bound: 1.1529943
time: 4.99 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.57 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 24.57
Output dim: 7, lower bound: -1.1301860, upper bound: 1.1486495
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 24.57
Output dim: 7, lower bound: -1.1392154, upper bound: 1.1486498
IS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 24.57
Output dim: 7, lower bound: -1.1529946, upper bound: 1.1440046
IS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 24.57
Output dim: 7, lower bound: -1.1529948, upper bound: 1.1529943

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -15.8013153, -11.6834469, -15.7649879, -11.6719265, -2.4766130, 2.4430332
1: -7.1749640, -4.4068003, -7.1570511, -4.4110212, -2.7441015, 2.7237101
2: -8.6780624, -6.1643634, -8.7042704, -6.1709166, -2.3507328, 2.3903742
3: -4.9824810, -2.4376297, -4.9551430, -2.4559221, -2.2499318, 2.2473049
4: -7.9161944, -5.2979136, -7.9379983, -5.2956653, -1.8636880, 1.8805940
5: -6.2801113, -3.7751884, -6.2597332, -3.7668066, -2.1538777, 2.1129224
6: -14.3909931, -11.0329885, -14.3856068, -11.0178375, -2.2324047, 2.2223229
7: 2.3188338, 4.7984467, 2.3045321, 4.7739925, -1.7701759, 1.7926507
8: -1.2725563, 0.9680920, -1.2215726, 0.9369025, -1.8986826, 1.8915484
9: -8.7968674, -5.7537770, -8.7829628, -5.7915521, -2.0856991, 2.1122150

Time for backsubstitution: 14.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6192
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of IS_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1301753, upper bound: 1.1461355
time: 4.93 seconds

## Relational analysis of IS_B1_A1_B2

### Relational analysis result of IS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1301753, upper bound: 1.1486408
time: 4.78 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -15.8229189, -11.6252232, -15.7687855, -11.6553326, -2.5368233, 2.4820457
1: -7.1823931, -4.3909712, -7.1591153, -4.4076223, -2.7531490, 2.7428145
2: -8.7359285, -6.1509843, -8.7206879, -6.1691089, -2.3750134, 2.4203582
3: -4.9978504, -2.4275477, -4.9593029, -2.4537749, -2.2663751, 2.2625649
4: -7.9657631, -5.2775507, -7.9517303, -5.2917261, -1.8885627, 1.9103394
5: -6.3065281, -3.7179592, -6.2641287, -3.7503946, -2.2038493, 2.1527426
6: -14.4064751, -10.9802980, -14.3885374, -11.0027924, -2.2812352, 2.2771173
7: 2.2615738, 4.8117628, 2.2882204, 4.7757864, -1.7934580, 1.8408210
8: -1.2872519, 0.9737005, -1.2247856, 0.9384165, -1.9153247, 1.8990402
9: -8.8147449, -5.7478409, -8.7875423, -5.7901473, -2.1144390, 2.1245193

Time for backsubstitution: 14.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6192
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 6156
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1392069, upper bound: 1.1461375
time: 4.73 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1392069, upper bound: 1.1486406
time: 4.49 seconds

## BFS IS instance: IS_B2_B1

### Backsubstitution after applying IS history:
0: -15.8602524, -11.6376753, -15.8424368, -11.6793060, -2.5416098, 2.5187356
1: -7.1939688, -4.3837185, -7.1887221, -4.3961544, -2.7737045, 2.7979178
2: -8.7244186, -6.1416664, -8.6829739, -6.1531687, -2.4287291, 2.4096971
3: -5.0204186, -2.4225872, -5.0092278, -2.4303646, -2.3352804, 2.3188329
4: -7.9566936, -5.2724600, -7.9208684, -5.2885032, -1.9148211, 1.8978367
5: -6.3344364, -3.7250943, -6.3123989, -3.7658811, -2.2353053, 2.2388217
6: -14.4105186, -10.9798689, -14.3979731, -11.0174685, -2.2324963, 2.2743506
7: 2.2704139, 4.8360844, 2.3113546, 4.8247833, -1.8395476, 1.8372016
8: -1.3298550, 0.9767361, -1.3185802, 0.9726424, -2.0000443, 1.9742792
9: -8.8128519, -5.7174082, -8.8003502, -5.7219272, -2.1426044, 2.1630938

Time for backsubstitution: 14.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 6192

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of IS_B2_B1_B1

### Relational analysis result of IS_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1529852, upper bound: 1.1416439
time: 4.65 seconds

## Relational analysis of IS_B2_B1_B2

### Relational analysis result of IS_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1529855, upper bound: 1.1439948
time: 4.86 seconds

## BFS IS instance: IS_B2_B2

### Backsubstitution after applying IS history:
0: -15.8640537, -11.6210804, -15.8640432, -11.6210861, -2.5806127, 2.5575733
1: -7.1961083, -4.3803353, -7.1961031, -4.3803391, -2.7928867, 2.8070259
2: -8.7408409, -6.1397614, -8.7408390, -6.1397667, -2.4588308, 2.4341049
3: -5.0245576, -2.4202356, -5.0245476, -2.4202385, -2.3505230, 2.3354082
4: -7.9703808, -5.2681613, -7.9703741, -5.2681651, -1.9444680, 1.9230299
5: -6.3388705, -3.7086444, -6.3388648, -3.7086492, -2.2751665, 2.2795737
6: -14.4134331, -10.9648457, -14.4134331, -10.9648571, -2.2872467, 2.3231449
7: 2.2540908, 4.8381071, 2.2540984, 4.8381004, -1.8760798, 1.8607278
8: -1.3332782, 0.9782429, -1.3332634, 0.9782405, -2.0074902, 1.9913640
9: -8.8183250, -5.7160306, -8.8183212, -5.7160392, -2.1559529, 2.1917250

Time for backsubstitution: 14.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 6192

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of IS_B2_B2_B1

### Relational analysis result of IS_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1529855, upper bound: 1.1506192
time: 4.67 seconds

## Relational analysis of IS_B2_B2_B2

### Relational analysis result of IS_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1529855, upper bound: 1.1529853
time: 4.78 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 23.73 seconds
IS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 23.73
Output dim: 7, lower bound: -1.1301753, upper bound: 1.1461355
IS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 23.73
Output dim: 7, lower bound: -1.1301753, upper bound: 1.1486408
IS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 23.73
Output dim: 7, lower bound: -1.1392069, upper bound: 1.1461375
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 23.73
Output dim: 7, lower bound: -1.1392069, upper bound: 1.1486406
IS_B2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 23.73
Output dim: 7, lower bound: -1.1529852, upper bound: 1.1416439
IS_B2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 23.73
Output dim: 7, lower bound: -1.1529855, upper bound: 1.1439948
IS_B2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 23.73
Output dim: 7, lower bound: -1.1529855, upper bound: 1.1506192
IS_B2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 23.73
Output dim: 7, lower bound: -1.1529855, upper bound: 1.1529853

## BFS IS instance: IS_B1_A1_B1

### Backsubstitution after applying IS history:
0: -15.8001900, -11.6857662, -15.7626648, -11.6765051, -2.4640727, 2.4350252
1: -7.1633043, -4.4074192, -7.1330023, -4.4123082, -2.7308741, 2.6988511
2: -8.6770258, -6.1681318, -8.7022095, -6.1786652, -2.3392339, 2.3831935
3: -4.9811821, -2.4408019, -4.9524169, -2.4623861, -2.2424707, 2.2414680
4: -7.9144578, -5.3006015, -7.9343815, -5.3011599, -1.8516397, 1.8703606
5: -6.2793078, -3.7759535, -6.2580624, -3.7683074, -2.1509776, 2.1102135
6: -14.3871126, -11.0333023, -14.3776608, -11.0184975, -2.2253437, 2.2093327
7: 2.3215756, 4.7974682, 2.3101697, 4.7719326, -1.7655418, 1.7859910
8: -1.2715693, 0.9674811, -1.2195308, 0.9356651, -1.8930583, 1.8870232
9: -8.7882481, -5.7540059, -8.7652988, -5.7920151, -2.0759931, 2.0931091

Time for backsubstitution: 14.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 6192
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6140

## Relational analysis of IS_B1_A1_B1_B1

### Relational analysis result of IS_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1267431, upper bound: 1.1455334
time: 4.91 seconds

## Relational analysis of IS_B1_A1_B1_B2

### Relational analysis result of IS_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1301712, upper bound: 1.1461304
time: 4.61 seconds

## BFS IS instance: IS_B1_A1_B2

### Backsubstitution after applying IS history:
0: -15.8013115, -11.6834526, -15.7835636, -11.6616459, -2.4793224, 2.4645309
1: -7.1749239, -4.4068022, -7.1623793, -4.3769064, -2.7767963, 2.7243285
2: -8.6780586, -6.1643710, -8.7187548, -6.1664991, -2.3554096, 2.4012403
3: -4.9824777, -2.4376407, -4.9794765, -2.4519053, -2.2543855, 2.2720308
4: -7.9161901, -5.2979169, -7.9595032, -5.2943282, -1.8644514, 1.8891745
5: -6.2801113, -3.7751908, -6.2789149, -3.7625706, -2.1592121, 2.1346397
6: -14.3909826, -11.0329914, -14.3957987, -11.0017519, -2.2419260, 2.2344103
7: 2.3188410, 4.7984443, 2.3003240, 4.7897825, -1.7880540, 1.7955685
8: -1.2725534, 0.9680901, -1.2275558, 0.9473562, -1.9046574, 1.9011939
9: -8.7968397, -5.7537756, -8.7896767, -5.7668581, -2.1090961, 2.1086335

Time for backsubstitution: 13.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 6192
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6135

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6140

## Relational analysis of IS_B1_A1_B2_B1

### Relational analysis result of IS_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1267431, upper bound: 1.1480203
time: 5.17 seconds

## Relational analysis of IS_B1_A1_B2_B2

### Relational analysis result of IS_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1301712, upper bound: 1.1486360
time: 4.68 seconds

## BFS IS instance: IS_B1_A2_B1

### Backsubstitution after applying IS history:
0: -15.8218050, -11.6275425, -15.7664652, -11.6599112, -2.5242977, 2.4740410
1: -7.1707287, -4.3915887, -7.1350632, -4.4089079, -2.7399311, 2.7179608
2: -8.7348928, -6.1547484, -8.7186260, -6.1768579, -2.3635206, 2.4131837
3: -4.9965487, -2.4307227, -4.9565763, -2.4602399, -2.2588468, 2.2567091
4: -7.9640474, -5.2802358, -7.9481225, -5.2972217, -1.8765121, 1.9016037
5: -6.3057380, -3.7187264, -6.2624669, -3.7518940, -2.2009826, 2.1500373
6: -14.4025993, -10.9806175, -14.3805923, -11.0034552, -2.2741752, 2.2641194
7: 2.2643147, 4.8107853, 2.2938571, 4.7737265, -1.7888122, 1.8341670
8: -1.2862654, 0.9730887, -1.2227449, 0.9371772, -1.9097004, 1.8945181
9: -8.8061371, -5.7480712, -8.7698841, -5.7906089, -2.1047292, 2.1053958

Time for backsubstitution: 14.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6192
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 6156
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6192

## Relational analysis of IS_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1392069, upper bound: 1.1370007
time: 4.58 seconds

## Relational analysis of IS_B1_A2_B1_A2

### Relational analysis result of IS_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1392069, upper bound: 1.1461375
time: 4.74 seconds

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: -15.8229160, -11.6252270, -15.7873802, -11.6450481, -2.5395308, 2.5035725
1: -7.1823540, -4.3909750, -7.1647320, -4.3735170, -2.7855983, 2.7437901
2: -8.7359247, -6.1509910, -8.7351646, -6.1643968, -2.3799582, 2.4358368
3: -4.9978485, -2.4275599, -4.9836655, -2.4492612, -2.2712703, 2.2872305
4: -7.9657583, -5.2775555, -7.9732842, -5.2892489, -1.8904552, 1.9325175
5: -6.3065257, -3.7179620, -6.2833171, -3.7460666, -2.2113304, 2.1744602
6: -14.4064617, -10.9802999, -14.3987322, -10.9866371, -2.2958946, 2.2885151
7: 2.2615814, 4.8117619, 2.2840161, 4.7922587, -1.8119428, 1.8437929
8: -1.2872496, 0.9736986, -1.2314627, 0.9488640, -1.9212866, 1.9092360
9: -8.8147144, -5.7478409, -8.7966671, -5.7654543, -2.1361251, 2.1239707

Time for backsubstitution: 14.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6192
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 6156
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6192

## Relational analysis of IS_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1392069, upper bound: 1.1392094
time: 4.62 seconds

## Relational analysis of IS_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1392069, upper bound: 1.1486426
time: 4.85 seconds

## BFS IS instance: IS_B2_B1_B1

### Backsubstitution after applying IS history:
0: -15.8591337, -11.6399899, -15.8401070, -11.6838961, -2.5290174, 2.5094323
1: -7.1823416, -4.3843336, -7.1648588, -4.3974371, -2.7604733, 2.7730923
2: -8.7233524, -6.1454172, -8.6807852, -6.1608610, -2.4171658, 2.4024000
3: -5.0191078, -2.4257631, -5.0064750, -2.4368744, -2.3277268, 2.3129945
4: -7.9549541, -5.2751470, -7.9171815, -5.2940030, -1.9027524, 1.8890674
5: -6.3336406, -3.7258918, -6.3106971, -3.7675221, -2.2322965, 2.2360828
6: -14.4066563, -10.9801884, -14.3900700, -11.0181313, -2.2254481, 2.2611966
7: 2.2731476, 4.8351030, 2.3169608, 4.8227139, -1.8343849, 1.8305349
8: -1.3288689, 0.9761100, -1.3165359, 0.9713130, -1.9943378, 1.9697483
9: -8.8042450, -5.7176399, -8.7826796, -5.7224026, -2.1328750, 2.1438904

Time for backsubstitution: 14.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 6192

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6140

## Relational analysis of IS_B2_B1_B1_A1

### Relational analysis result of IS_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1524005, upper bound: 1.1381989
time: 4.80 seconds

## Relational analysis of IS_B2_B1_B1_A2

### Relational analysis result of IS_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1529798, upper bound: 1.1416365
time: 4.85 seconds

## BFS IS instance: IS_B2_B1_B2

### Backsubstitution after applying IS history:
0: -15.8602467, -11.6376791, -15.8610678, -11.6690998, -2.5524759, 2.5264375
1: -7.1939301, -4.3837204, -7.1942043, -4.3619776, -2.8049746, 2.7981868
2: -8.7244139, -6.1416731, -8.6974258, -6.1486645, -2.4337025, 2.4252386
3: -5.0204148, -2.4225984, -5.0340114, -2.4264865, -2.3396463, 2.3439188
4: -7.9566894, -5.2724657, -7.9423451, -5.2871876, -1.9118891, 1.9200091
5: -6.3344350, -3.7250969, -6.3315430, -3.7617269, -2.2427130, 2.2540727
6: -14.4105072, -10.9798717, -14.4078188, -11.0013638, -2.2491207, 2.2801607
7: 2.2704215, 4.8360820, 2.3073292, 4.8407097, -1.8467789, 1.8398433
8: -1.3298526, 0.9767342, -1.3247399, 0.9831514, -2.0059881, 1.9842069
9: -8.8128262, -5.7174091, -8.8070440, -5.6971583, -2.1634479, 2.1613870

Time for backsubstitution: 14.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 6192

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6140

## Relational analysis of IS_B2_B1_B2_A1

### Relational analysis result of IS_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1524005, upper bound: 1.1405542
time: 4.96 seconds

## Relational analysis of IS_B2_B1_B2_A2

### Relational analysis result of IS_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1529798, upper bound: 1.1439887
time: 5.17 seconds

## BFS IS instance: IS_B2_B2_B1

### Backsubstitution after applying IS history:
0: -15.8629360, -11.6233940, -15.8617249, -11.6256790, -2.5680218, 2.5495543
1: -7.1844778, -4.3809490, -7.1722288, -4.3816137, -2.7796612, 2.7822065
2: -8.7397757, -6.1435108, -8.7386494, -6.1474504, -2.4472795, 2.4268103
3: -5.0232491, -2.4234133, -5.0217924, -2.4267564, -2.3429332, 2.3295164
4: -7.9686561, -5.2708454, -7.9667277, -5.2736559, -1.9324036, 1.9142547
5: -6.3380785, -3.7094429, -6.3371916, -3.7102938, -2.2721543, 2.2768440
6: -14.4095745, -10.9651680, -14.4055328, -10.9655285, -2.2801824, 2.3099985
7: 2.2568254, 4.8371253, 2.2597084, 4.8360305, -1.8714342, 1.8540537
8: -1.3322897, 0.9776154, -1.3312263, 0.9769092, -2.0017872, 1.9868388
9: -8.8097248, -5.7162638, -8.8006802, -5.7165184, -2.1462131, 2.1725125

Time for backsubstitution: 14.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 6192

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6140

## Relational analysis of IS_B2_B2_B1_A1

### Relational analysis result of IS_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1524005, upper bound: 1.1471906
time: 4.91 seconds

## Relational analysis of IS_B2_B2_B1_A2

### Relational analysis result of IS_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1529801, upper bound: 1.1506133
time: 6.83 seconds

## BFS IS instance: IS_B2_B2_B2

### Backsubstitution after applying IS history:
0: -15.8640518, -11.6210823, -15.8826771, -11.6108828, -2.5914783, 2.5791850
1: -7.1960673, -4.3803377, -7.2019954, -4.3461905, -2.8247280, 2.8076553
2: -8.7408371, -6.1397681, -8.7552652, -6.1349716, -2.4640632, 2.4496431
3: -5.0245538, -2.4202487, -5.0493255, -2.4158285, -2.3553824, 2.3602777
4: -7.9703751, -5.2681637, -7.9919052, -5.2657213, -1.9463415, 1.9452248
5: -6.3388681, -3.7086468, -6.3580275, -3.7044063, -2.2826600, 2.3012793
6: -14.4134235, -10.9648438, -14.4232950, -10.9486589, -2.3039899, 2.3342297
7: 2.2541003, 4.8381052, 2.2500930, 4.8547006, -1.8946652, 1.8634043
8: -1.3332734, 0.9782414, -1.3401041, 0.9887376, -2.0134480, 2.0018067
9: -8.8182983, -5.7160301, -8.8274460, -5.6912746, -2.1767249, 2.1931827

Time for backsubstitution: 14.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6192

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6140

## Relational analysis of IS_B2_B2_B2_A1

### Relational analysis result of IS_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1524005, upper bound: 1.1495508
time: 4.39 seconds

## Relational analysis of IS_B2_B2_B2_A2

### Relational analysis result of IS_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1529801, upper bound: 1.1529796
time: 5.00 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 23.67 seconds
IS_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 23.67
Output dim: 7, lower bound: -1.1267431, upper bound: 1.1455334
IS_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 23.67
Output dim: 7, lower bound: -1.1301712, upper bound: 1.1461304
IS_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 23.67
Output dim: 7, lower bound: -1.1267431, upper bound: 1.1480203
IS_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 23.67
Output dim: 7, lower bound: -1.1301712, upper bound: 1.1486360
IS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 23.67
Output dim: 7, lower bound: -1.1392069, upper bound: 1.1370007
IS_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 23.67
Output dim: 7, lower bound: -1.1392069, upper bound: 1.1461375
IS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 23.67
Output dim: 7, lower bound: -1.1392069, upper bound: 1.1392094
IS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 23.67
Output dim: 7, lower bound: -1.1392069, upper bound: 1.1486426
IS_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 23.67
Output dim: 7, lower bound: -1.1524005, upper bound: 1.1381989
IS_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 23.67
Output dim: 7, lower bound: -1.1529798, upper bound: 1.1416365
IS_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 23.67
Output dim: 7, lower bound: -1.1524005, upper bound: 1.1405542
IS_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 23.67
Output dim: 7, lower bound: -1.1529798, upper bound: 1.1439887
IS_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 23.67
Output dim: 7, lower bound: -1.1524005, upper bound: 1.1471906
IS_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 23.67
Output dim: 7, lower bound: -1.1529801, upper bound: 1.1506133
IS_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 23.67
Output dim: 7, lower bound: -1.1524005, upper bound: 1.1495508
IS_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 23.67
Output dim: 7, lower bound: -1.1529801, upper bound: 1.1529796

## BFS IS instance: IS_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -15.7995682, -11.6863613, -15.7611084, -11.6779423, -2.4615769, 2.4323707
1: -7.1620111, -4.4128437, -7.1297584, -4.4259958, -2.7165365, 2.6909041
2: -8.6743355, -6.1696653, -8.6956215, -6.1825972, -2.3274326, 2.3699527
3: -4.9754629, -2.4428926, -4.9380851, -2.4676795, -2.2310658, 2.2243118
4: -7.9124045, -5.3075256, -7.9291286, -5.3185987, -1.8310556, 1.8567195
5: -6.2768040, -3.7771587, -6.2515898, -3.7714152, -2.1442420, 2.1015146
6: -14.3818283, -11.0342474, -14.3642654, -11.0209017, -2.2180347, 2.1948240
7: 2.3230495, 4.7946491, 2.3139415, 4.7648144, -1.7571878, 1.7796204
8: -1.2696452, 0.9669461, -1.2144151, 0.9343014, -1.8835430, 1.8759508
9: -8.7827339, -5.7548189, -8.7514620, -5.7940903, -2.0684571, 2.0788517

Time for backsubstitution: 14.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6192
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6192

## Relational analysis of IS_B1_A1_B1_B1_A1

### Relational analysis result of IS_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1267431, upper bound: 1.1363796
time: 4.68 seconds

## Relational analysis of IS_B1_A1_B1_B1_A2

### Relational analysis result of IS_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1267431, upper bound: 1.1455334
time: 4.88 seconds

## BFS IS instance: IS_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -15.8001862, -11.6857681, -15.7744370, -11.6744242, -2.4671612, 2.4506712
1: -7.1633005, -4.4074326, -7.1544352, -4.4026175, -2.7406759, 2.7261438
2: -8.6770210, -6.1681356, -8.7089367, -6.1561294, -2.3601303, 2.3913295
3: -4.9811616, -2.4408073, -4.9625878, -2.4236097, -2.2630730, 2.2461658
4: -7.9144545, -5.3006268, -7.9932256, -5.2967143, -1.8513250, 1.8920302
5: -6.2793026, -3.7759557, -6.2760057, -3.7411876, -2.1638284, 2.1274300
6: -14.3871050, -11.0333061, -14.3839455, -10.9726009, -2.2487705, 2.2118134
7: 2.3215771, 4.7974634, 2.2914181, 4.7755203, -1.7684996, 1.7983356
8: -1.2715640, 0.9674807, -1.2350621, 0.9389472, -1.8923008, 1.9091084
9: -8.7882318, -5.7540054, -8.7768974, -5.7559576, -2.1017613, 2.1065478

Time for backsubstitution: 14.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6192
type: B, layer: 1, pos: 6156
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6192

## Relational analysis of IS_B1_A1_B1_B2_A1

### Relational analysis result of IS_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1301712, upper bound: 1.1369987
time: 4.82 seconds

## Relational analysis of IS_B1_A1_B1_B2_A2

### Relational analysis result of IS_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1301712, upper bound: 1.1461304
time: 4.56 seconds

## BFS IS instance: IS_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -15.8006887, -11.6840458, -15.7820148, -11.6630726, -2.4768295, 2.4618645
1: -7.1736317, -4.4122272, -7.1592054, -4.3905950, -2.7623320, 2.7164226
2: -8.6753654, -6.1659036, -8.7121038, -6.1704140, -2.3435888, 2.3875952
3: -4.9767599, -2.4397304, -4.9650817, -2.4571698, -2.2430205, 2.2547793
4: -7.9141340, -5.3048420, -7.9542365, -5.3117657, -1.8438625, 1.8755441
5: -6.2776055, -3.7763982, -6.2724333, -3.7656617, -2.1524434, 2.1259460
6: -14.3857012, -11.0339365, -14.3824224, -11.0041714, -2.2336659, 2.2199225
7: 2.3203130, 4.7956228, 2.3040385, 4.7826328, -1.7796788, 1.7892346
8: -1.2706261, 0.9675541, -1.2224269, 0.9459944, -1.8951378, 1.8900537
9: -8.7913246, -5.7545934, -8.7758408, -5.7689438, -2.1004872, 2.0944071

Time for backsubstitution: 14.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6192
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6192

## Relational analysis of IS_B1_A1_B2_B1_A1

### Relational analysis result of IS_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1267431, upper bound: 1.1386226
time: 4.79 seconds

## Relational analysis of IS_B1_A1_B2_B1_A2

### Relational analysis result of IS_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1267431, upper bound: 1.1480203
time: 5.15 seconds

## BFS IS instance: IS_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -15.8013077, -11.6834536, -15.7953310, -11.6595526, -2.4824142, 2.4801474
1: -7.1749201, -4.4068155, -7.1837893, -4.3671837, -2.7865829, 2.7516060
2: -8.6780539, -6.1643734, -8.7255230, -6.1439772, -2.3762436, 2.4044142
3: -4.9824586, -2.4376462, -4.9896407, -2.4131746, -2.2751350, 2.2765975
4: -7.9161863, -5.2979431, -8.0182276, -5.2899089, -1.8641334, 1.9108844
5: -6.2801046, -3.7751935, -6.2967916, -3.7355061, -2.1720614, 2.1518798
6: -14.3909750, -11.0329933, -14.4021454, -10.9558706, -2.2605689, 2.2369270
7: 2.3188457, 4.7984381, 2.2816024, 4.7933674, -1.7910092, 1.8080194
8: -1.2725492, 0.9680901, -1.2430346, 0.9506497, -1.9038923, 1.9231362
9: -8.7968235, -5.7537804, -8.8012371, -5.7308455, -2.1159420, 2.1219835

Time for backsubstitution: 14.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6192
type: B, layer: 1, pos: 6156
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6192

## Relational analysis of IS_B1_A1_B2_B2_A1

### Relational analysis result of IS_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1301712, upper bound: 1.1392057
time: 4.69 seconds

## Relational analysis of IS_B1_A1_B2_B2_A2

### Relational analysis result of IS_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1301712, upper bound: 1.1486360
time: 4.79 seconds

## BFS IS instance: IS_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -15.7676659, -11.6576509, -15.7664652, -11.6599112, -2.4744630, 2.4522371
1: -7.1473989, -4.4082422, -7.1350632, -4.4089079, -2.7096348, 2.7016425
2: -8.7196846, -6.1728897, -8.7186260, -6.1768579, -2.3478470, 2.3874450
3: -4.9580069, -2.4569290, -4.9565763, -2.4602399, -2.2255993, 2.2269843
4: -7.9500208, -5.2944136, -7.9481225, -5.2972217, -1.8595924, 1.8893757
5: -6.2633419, -3.7511239, -6.2624669, -3.7518940, -2.1455059, 2.1206894
6: -14.3846521, -11.0031147, -14.3805923, -11.0034552, -2.2646790, 2.2412500
7: 2.2909732, 4.7748079, 2.2938571, 4.7737265, -1.7701578, 1.8024452
8: -1.2237952, 0.9378328, -1.2227449, 0.9371772, -1.8631196, 1.8625879
9: -8.7789278, -5.7903709, -8.7698841, -5.7906089, -2.0774007, 2.0627310

Time for backsubstitution: 14.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6140

## Relational analysis of IS_B1_A2_B1_A1_B1

### Relational analysis result of IS_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1358135, upper bound: 1.1363825
time: 4.61 seconds

## Relational analysis of IS_B1_A2_B1_A1_B2

### Relational analysis result of IS_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1392028, upper bound: 1.1369967
time: 4.52 seconds

## BFS IS instance: IS_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -15.8628969, -11.6245928, -15.7664652, -11.6599112, -2.5277820, 2.4756999
1: -7.1824808, -4.3810377, -7.1350632, -4.4089079, -2.7453136, 2.7291474
2: -8.7395487, -6.1436410, -8.7186260, -6.1768579, -2.3681288, 2.4287291
3: -5.0231495, -2.4243305, -4.9565763, -2.4602399, -2.2716689, 2.2594824
4: -7.9677896, -5.2708755, -7.9481225, -5.2972217, -1.8790507, 1.9143572
5: -6.3379207, -3.7115312, -6.2624669, -3.7518940, -2.2108164, 2.1527176
6: -14.4093399, -10.9652739, -14.3805923, -11.0034552, -2.2775445, 2.2717748
7: 2.2579603, 4.8371086, 2.2938571, 4.7737265, -1.7925916, 1.8377299
8: -1.3322468, 0.9768586, -1.2227449, 0.9371772, -1.9137392, 1.8933647
9: -8.8095570, -5.7163506, -8.7698841, -5.7906089, -2.1072831, 2.1087735

Time for backsubstitution: 14.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 6156
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6140

## Relational analysis of IS_B1_A2_B1_A2_A1

### Relational analysis result of IS_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1386200, upper bound: 1.1426846
time: 4.87 seconds

## Relational analysis of IS_B1_A2_B1_A2_A2

### Relational analysis result of IS_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1392031, upper bound: 1.1461325
time: 4.75 seconds

## BFS IS instance: IS_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -15.7687807, -11.6553431, -15.7873802, -11.6450481, -2.4978123, 2.4815273
1: -7.1590748, -4.4076247, -7.1647320, -4.3735170, -2.7554626, 2.7274513
2: -8.7206831, -6.1691160, -8.7351646, -6.1643968, -2.3642502, 2.4100862
3: -4.9592991, -2.4537854, -4.9836655, -2.4492612, -2.2380147, 2.2574821
4: -7.9517217, -5.2917314, -7.9732842, -5.2892489, -1.8735428, 1.9202881
5: -6.2641263, -3.7503994, -6.2833171, -3.7460666, -2.1559224, 2.1450760
6: -14.3885269, -11.0028000, -14.3987322, -10.9866371, -2.2863116, 2.2666011
7: 2.2882352, 4.7757854, 2.2840161, 4.7922587, -1.7932863, 1.8118875
8: -1.2247813, 0.9384141, -1.2314627, 0.9488640, -1.8816099, 1.8772717
9: -8.7875128, -5.7901483, -8.7966671, -5.7654543, -2.1087942, 2.0830960

Time for backsubstitution: 14.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 6156
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6135

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6140

## Relational analysis of IS_B1_A2_B2_A1_B1

### Relational analysis result of IS_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1358135, upper bound: 1.1386229
time: 4.97 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2

### Relational analysis result of IS_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1392028, upper bound: 1.1392055
time: 4.44 seconds

## BFS IS instance: IS_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -15.8640137, -11.6222067, -15.7873802, -11.6450481, -2.5430264, 2.5010703
1: -7.1941948, -4.3804207, -7.1647320, -4.3735170, -2.7903686, 2.7549615
2: -8.7406235, -6.1398916, -8.7351646, -6.1643968, -2.3846188, 2.4513936
3: -5.0244608, -2.4211090, -4.9836655, -2.4492612, -2.2841630, 2.2843132
4: -7.9695625, -5.2681923, -7.9732842, -5.2892489, -1.8930492, 1.9366503
5: -6.3387156, -3.7106085, -6.2833171, -3.7460666, -2.2190957, 2.1772106
6: -14.4132042, -10.9649467, -14.3987322, -10.9866371, -2.2938557, 2.2904842
7: 2.2551641, 4.8380880, 2.2840161, 4.7922587, -1.8131847, 1.8473659
8: -1.3332329, 0.9775281, -1.2314627, 0.9488640, -1.9253483, 1.9041095
9: -8.8181438, -5.7161150, -8.7966671, -5.7654543, -2.1221838, 2.1273601

Time for backsubstitution: 14.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 6156
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6140

## Relational analysis of IS_B1_A2_B2_A2_B1

### Relational analysis result of IS_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1358135, upper bound: 1.1480203
time: 5.27 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2

### Relational analysis result of IS_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1392028, upper bound: 1.1486357
time: 4.56 seconds

## BFS IS instance: IS_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -15.8575563, -11.6414165, -15.8394823, -11.6844883, -2.5263534, 2.5069315
1: -7.1790380, -4.3980355, -7.1635571, -4.4028625, -2.7527704, 2.7587185
2: -8.7166882, -6.1493349, -8.6780863, -6.1623950, -2.4039354, 2.3907509
3: -5.0046797, -2.4310675, -5.0007439, -2.4389701, -2.3104877, 2.3015189
4: -7.9496746, -5.2925773, -7.9151130, -5.3009233, -1.8905921, 1.8684833
5: -6.3272209, -3.7289526, -6.3082032, -3.7687190, -2.2236085, 2.2296159
6: -14.3933125, -10.9826164, -14.3847847, -11.0190754, -2.2108755, 2.2534115
7: 2.2768526, 4.8279305, 2.3184309, 4.8198881, -1.8280311, 1.8221350
8: -1.3237476, 0.9747415, -1.3146172, 0.9707775, -1.9828262, 1.9606102
9: -8.7904110, -5.7197185, -8.7771645, -5.7232141, -2.1186299, 2.1362977

Time for backsubstitution: 14.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 6192

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6156

## Relational analysis of IS_B2_B1_B1_A1_A1

### Relational analysis result of IS_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1434094, upper bound: 1.1381968
time: 4.68 seconds

## Relational analysis of IS_B2_B1_B1_A1_A2

### Relational analysis result of IS_B2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1434094, upper bound: 1.1381998
time: 4.63 seconds

## BFS IS instance: IS_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -15.8708382, -11.6379042, -15.8401031, -11.6838980, -2.5445662, 2.5125165
1: -7.2036333, -4.3746338, -7.1648569, -4.3974504, -2.7879229, 2.7829008
2: -8.7301693, -6.1228027, -8.6807785, -6.1608639, -2.4273553, 2.4235573
3: -5.0292459, -2.3870592, -5.0064554, -2.4368796, -2.3324127, 2.3408446
4: -8.0137949, -5.2707272, -7.9171758, -5.2940278, -1.9263184, 1.8887262
5: -6.3517313, -3.6988087, -6.3106909, -3.7675238, -2.2496662, 2.2512548
6: -14.4130163, -10.9344378, -14.3900604, -11.0181341, -2.2279186, 2.2802827
7: 2.2544446, 4.8386774, 2.3169641, 4.8227100, -1.8468504, 1.8334742
8: -1.3442874, 0.9793544, -1.3165317, 0.9713120, -2.0052633, 1.9690716
9: -8.8158655, -5.6814919, -8.7826633, -5.7224045, -2.1478477, 2.1557522

Time for backsubstitution: 14.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6192

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6156

## Relational analysis of IS_B2_B1_B1_A2_A1

### Relational analysis result of IS_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1439891, upper bound: 1.1416352
time: 5.07 seconds

## Relational analysis of IS_B2_B1_B1_A2_A2

### Relational analysis result of IS_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1439891, upper bound: 1.1416382
time: 5.69 seconds

## BFS IS instance: IS_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -15.8586693, -11.6391029, -15.8604469, -11.6696892, -2.5498114, 2.5239358
1: -7.1906333, -4.3974228, -7.1929274, -4.3673997, -2.7962780, 2.7838240
2: -8.7177429, -6.1455870, -8.6947069, -6.1501918, -2.4204617, 2.4136066
3: -5.0059843, -2.4279008, -5.0282583, -2.4285717, -2.3223743, 2.3324294
4: -7.9514146, -5.2898936, -7.9402614, -5.2941074, -1.8981857, 1.8994212
5: -6.3280129, -3.7281597, -6.3290462, -3.7629185, -2.2340202, 2.2473216
6: -14.3971748, -10.9823055, -14.4025364, -11.0023098, -2.2345362, 2.2718439
7: 2.2741232, 4.8289070, 2.3087792, 4.8378725, -1.8404326, 1.8314550
8: -1.3247242, 0.9753666, -1.3228173, 0.9826140, -1.9944663, 1.9750602
9: -8.7989922, -5.7194910, -8.8015308, -5.6979747, -2.1492214, 2.1537971

Time for backsubstitution: 14.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 6192

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6156

## Relational analysis of IS_B2_B1_B2_A1_A1

### Relational analysis result of IS_B2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1434094, upper bound: 1.1405541
time: 4.64 seconds

## Relational analysis of IS_B2_B1_B2_A1_A2

### Relational analysis result of IS_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1434094, upper bound: 1.1405551
time: 7.19 seconds

## BFS IS instance: IS_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -15.8719540, -11.6355953, -15.8610659, -11.6690998, -2.5661902, 2.5295203
1: -7.2152319, -4.3740182, -7.1941991, -4.3619909, -2.8137646, 2.8079972
2: -8.7312222, -6.1190557, -8.6974220, -6.1486678, -2.4400191, 2.4464049
3: -5.0305414, -2.3838902, -5.0339894, -2.4264901, -2.3443246, 2.3596084
4: -8.0155287, -5.2680445, -7.9423399, -5.2872124, -1.9334292, 1.9196665
5: -6.3525271, -3.6980190, -6.3315372, -3.7617290, -2.2600875, 2.2669613
6: -14.4168739, -10.9341145, -14.4078074, -11.0013647, -2.2515874, 2.2987287
7: 2.2517157, 4.8396511, 2.3073330, 4.8407016, -1.8592372, 1.8427792
8: -1.3452682, 0.9799824, -1.3247337, 0.9831510, -2.0168986, 1.9835289
9: -8.8244410, -5.6812587, -8.8070288, -5.6971598, -2.1768799, 2.1712983

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6192

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6156

## Relational analysis of IS_B2_B1_B2_A2_A1

### Relational analysis result of IS_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1439891, upper bound: 1.1439889
time: 4.84 seconds

## Relational analysis of IS_B2_B1_B2_A2_A2

### Relational analysis result of IS_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1439891, upper bound: 1.1439918
time: 5.46 seconds

## BFS IS instance: IS_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -15.8613625, -11.6248083, -15.8611050, -11.6262474, -2.5653625, 2.5470233
1: -7.1811543, -4.3946671, -7.1709123, -4.3870707, -2.7719116, 2.7678103
2: -8.7331467, -6.1474447, -8.7360096, -6.1489930, -2.4341083, 2.4151998
3: -5.0089064, -2.4287319, -5.0160933, -2.4288445, -2.3257399, 2.3180690
4: -7.9633627, -5.2883282, -7.9646349, -5.2805996, -1.9202151, 1.8935978
5: -6.3316116, -3.7125072, -6.3346281, -3.7114940, -2.2634406, 2.2703760
6: -14.3962193, -10.9675970, -14.4002285, -10.9664707, -2.2655959, 2.3026524
7: 2.2605257, 4.8299036, 2.2611675, 4.8331566, -1.8655815, 1.8456128
8: -1.3270569, 0.9762473, -1.3291769, 0.9763727, -1.9901547, 1.9775426
9: -8.7957850, -5.7183380, -8.7951450, -5.7173338, -2.1318607, 2.1650090

Time for backsubstitution: 14.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 6192

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 451

## Relational analysis of IS_B2_B2_B1_A1_A1

### Relational analysis result of IS_B2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1506431, upper bound: 1.1420912
time: 4.39 seconds

## Relational analysis of IS_B2_B2_B1_A1_A2

### Relational analysis result of IS_B2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1524000, upper bound: 1.1471882
time: 5.14 seconds

## BFS IS instance: IS_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -15.8747711, -11.6213131, -15.8617249, -11.6256790, -2.5836830, 2.5543156
1: -7.2065673, -4.3712478, -7.1722236, -4.3816257, -2.8080988, 2.7920241
2: -8.7465916, -6.1200199, -8.7386427, -6.1474543, -2.4584999, 2.4488149
3: -5.0334115, -2.3832288, -5.0217719, -2.4267597, -2.3476195, 2.3584766
4: -8.0273085, -5.2633395, -7.9667234, -5.2736797, -1.9712355, 1.9170518
5: -6.3562679, -3.6821072, -6.3371868, -3.7102976, -2.2897134, 2.3018398
6: -14.4159794, -10.9191694, -14.4055233, -10.9655304, -2.2827015, 2.3345098
7: 2.2381325, 4.8425145, 2.2597122, 4.8360238, -1.8933072, 1.8587639
8: -1.3496675, 0.9808469, -1.3312211, 0.9769092, -2.0139494, 1.9861705
9: -8.8280821, -5.6801906, -8.8006687, -5.7165217, -2.1700249, 2.1823287

Time for backsubstitution: 14.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6192

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 451

## Relational analysis of IS_B2_B2_B1_A2_A1

### Relational analysis result of IS_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1512031, upper bound: 1.1454693
time: 5.53 seconds

## Relational analysis of IS_B2_B2_B1_A2_A2

### Relational analysis result of IS_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1529793, upper bound: 1.1506125
time: 5.14 seconds

## BFS IS instance: IS_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -15.8624716, -11.6224947, -15.8820572, -11.6114502, -2.5888166, 2.5766478
1: -7.1927509, -4.3940578, -7.2006340, -4.3516474, -2.8160052, 2.7932682
2: -8.7342014, -6.1436977, -8.7526026, -6.1365080, -2.4508882, 2.4380460
3: -5.0102100, -2.4255629, -5.0436001, -2.4179101, -2.3381538, 2.3487911
4: -7.9651046, -5.2856474, -7.9898176, -5.2726617, -1.9341474, 1.9245648
5: -6.3323994, -3.7117157, -6.3554602, -3.7056015, -2.2739420, 2.2948093
6: -14.4000778, -10.9672852, -14.4179974, -10.9496059, -2.2893920, 2.3259411
7: 2.2577963, 4.8308830, 2.2515354, 4.8518229, -1.8888159, 1.8549728
8: -1.3280349, 0.9768724, -1.3380494, 0.9882002, -2.0018063, 1.9925010
9: -8.8043594, -5.7181120, -8.8219166, -5.6920910, -2.1623764, 2.1856837

Time for backsubstitution: 14.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6192

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 451

## Relational analysis of IS_B2_B2_B2_A1_A1

### Relational analysis result of IS_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1506431, upper bound: 1.1444496
time: 4.62 seconds

## Relational analysis of IS_B2_B2_B2_A1_A2

### Relational analysis result of IS_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1524000, upper bound: 1.1495473
time: 4.91 seconds

## BFS IS instance: IS_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -15.8758831, -11.6189995, -15.8826742, -11.6108789, -2.6049325, 2.5839467
1: -7.2181654, -4.3706350, -7.2019916, -4.3462048, -2.8344908, 2.8174767
2: -8.7476463, -6.1162767, -8.7552614, -6.1349754, -2.4752722, 2.4716606
3: -5.0347061, -2.3800640, -5.0493035, -2.4158340, -2.3600588, 2.3770247
4: -8.0290432, -5.2606692, -7.9919004, -5.2657442, -1.9794445, 1.9480295
5: -6.3570585, -3.6813169, -6.3580213, -3.7044096, -2.3002214, 2.3245070
6: -14.4198351, -10.9188461, -14.4232874, -10.9486599, -2.3065062, 2.3531115
7: 2.2354021, 4.8434963, 2.2500958, 4.8546948, -1.9080615, 1.8681064
8: -1.3506541, 0.9814749, -1.3400979, 0.9887366, -2.0256014, 2.0011382
9: -8.8366318, -5.6799579, -8.8274288, -5.6912746, -2.1990309, 2.2010660

Time for backsubstitution: 14.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6192

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 451

## Relational analysis of IS_B2_B2_B2_A2_A1

### Relational analysis result of IS_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1512031, upper bound: 1.1478478
time: 4.91 seconds

## Relational analysis of IS_B2_B2_B2_A2_A2

### Relational analysis result of IS_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1529793, upper bound: 1.1529782
time: 5.13 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 24.36 seconds
IS_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.36
Output dim: 7, lower bound: -1.1267431, upper bound: 1.1363796
IS_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.36
Output dim: 7, lower bound: -1.1267431, upper bound: 1.1455334
IS_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.36
Output dim: 7, lower bound: -1.1301712, upper bound: 1.1369987
IS_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.36
Output dim: 7, lower bound: -1.1301712, upper bound: 1.1461304
IS_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.36
Output dim: 7, lower bound: -1.1267431, upper bound: 1.1386226
IS_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.36
Output dim: 7, lower bound: -1.1267431, upper bound: 1.1480203
IS_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.36
Output dim: 7, lower bound: -1.1301712, upper bound: 1.1392057
IS_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.36
Output dim: 7, lower bound: -1.1301712, upper bound: 1.1486360
IS_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 24.36
Output dim: 7, lower bound: -1.1358135, upper bound: 1.1363825
IS_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 24.36
Output dim: 7, lower bound: -1.1392028, upper bound: 1.1369967
IS_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 24.36
Output dim: 7, lower bound: -1.1386200, upper bound: 1.1426846
IS_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 24.36
Output dim: 7, lower bound: -1.1392031, upper bound: 1.1461325
IS_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 24.36
Output dim: 7, lower bound: -1.1358135, upper bound: 1.1386229
IS_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 24.36
Output dim: 7, lower bound: -1.1392028, upper bound: 1.1392055
IS_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 24.36
Output dim: 7, lower bound: -1.1358135, upper bound: 1.1480203
IS_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 24.36
Output dim: 7, lower bound: -1.1392028, upper bound: 1.1486357
IS_B2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 24.36
Output dim: 7, lower bound: -1.1434094, upper bound: 1.1381968
IS_B2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 24.36
Output dim: 7, lower bound: -1.1434094, upper bound: 1.1381998
IS_B2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 24.36
Output dim: 7, lower bound: -1.1439891, upper bound: 1.1416352
IS_B2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 24.36
Output dim: 7, lower bound: -1.1439891, upper bound: 1.1416382
IS_B2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 24.36
Output dim: 7, lower bound: -1.1434094, upper bound: 1.1405541
IS_B2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 24.36
Output dim: 7, lower bound: -1.1434094, upper bound: 1.1405551
IS_B2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 24.36
Output dim: 7, lower bound: -1.1439891, upper bound: 1.1439889
IS_B2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 24.36
Output dim: 7, lower bound: -1.1439891, upper bound: 1.1439918
IS_B2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 24.36
Output dim: 7, lower bound: -1.1506431, upper bound: 1.1420912
IS_B2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 24.36
Output dim: 7, lower bound: -1.1524000, upper bound: 1.1471882
IS_B2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 24.36
Output dim: 7, lower bound: -1.1512031, upper bound: 1.1454693
IS_B2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 24.36
Output dim: 7, lower bound: -1.1529793, upper bound: 1.1506125
IS_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 24.36
Output dim: 7, lower bound: -1.1506431, upper bound: 1.1444496
IS_B2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 24.36
Output dim: 7, lower bound: -1.1524000, upper bound: 1.1495473
IS_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 24.36
Output dim: 7, lower bound: -1.1512031, upper bound: 1.1478478
IS_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 24.36
Output dim: 7, lower bound: -1.1529793, upper bound: 1.1529782

## BFS IS instance: IS_B1_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -15.7454357, -11.7164783, -15.7611084, -11.6779423, -2.4335146, 2.4106407
1: -7.1387825, -4.4295325, -7.1297584, -4.4259958, -2.6862593, 2.6745844
2: -8.6591587, -6.1877155, -8.6956215, -6.1825972, -2.3118391, 2.3442945
3: -4.9368610, -2.4688721, -4.9380851, -2.4676795, -2.1976905, 2.1947961
4: -7.8983126, -5.3213716, -7.9291286, -5.3185987, -1.8140984, 1.8479133
5: -6.2344875, -3.8095291, -6.2515898, -3.7714152, -2.0983744, 2.0721939
6: -14.3638411, -11.0567799, -14.3642654, -11.0209017, -2.2085581, 2.1719189
7: 2.3496976, 4.7589197, 2.3139415, 4.7648144, -1.7386575, 1.7608557
8: -1.2073421, 0.9316726, -1.2144151, 0.9343014, -1.8371758, 1.8440416
9: -8.7564554, -5.7971740, -8.7514620, -5.7940903, -2.0422440, 2.0361791

Time for backsubstitution: 14.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6156

## Relational analysis of IS_B1_A1_B1_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1267430, upper bound: 1.1273361
time: 4.91 seconds

## Relational analysis of IS_B1_A1_B1_B1_A1_B2

### Relational analysis result of IS_B1_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1267431, upper bound: 1.1363796
time: 4.77 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 23.97 seconds
IS_B1_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 23.97
Output dim: 7, lower bound: -1.1267430, upper bound: 1.1273361
IS_B1_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 23.97
Output dim: 7, lower bound: -1.1267431, upper bound: 1.1363796
IS_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.97
Output dim: 7, lower bound: -1.1267431, upper bound: 1.1455334
IS_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.97
Output dim: 7, lower bound: -1.1301712, upper bound: 1.1369987
IS_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.97
Output dim: 7, lower bound: -1.1301712, upper bound: 1.1461304
IS_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.97
Output dim: 7, lower bound: -1.1267431, upper bound: 1.1386226
IS_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.97
Output dim: 7, lower bound: -1.1267431, upper bound: 1.1480203
IS_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.97
Output dim: 7, lower bound: -1.1301712, upper bound: 1.1392057
IS_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.97
Output dim: 7, lower bound: -1.1301712, upper bound: 1.1486360
IS_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 23.97
Output dim: 7, lower bound: -1.1358135, upper bound: 1.1363825
IS_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 23.97
Output dim: 7, lower bound: -1.1392028, upper bound: 1.1369967
IS_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 23.97
Output dim: 7, lower bound: -1.1386200, upper bound: 1.1426846
IS_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 23.97
Output dim: 7, lower bound: -1.1392031, upper bound: 1.1461325
IS_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 23.97
Output dim: 7, lower bound: -1.1358135, upper bound: 1.1386229
IS_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 23.97
Output dim: 7, lower bound: -1.1392028, upper bound: 1.1392055
IS_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 23.97
Output dim: 7, lower bound: -1.1358135, upper bound: 1.1480203
IS_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 23.97
Output dim: 7, lower bound: -1.1392028, upper bound: 1.1486357
IS_B2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 23.97
Output dim: 7, lower bound: -1.1434094, upper bound: 1.1381968
IS_B2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 23.97
Output dim: 7, lower bound: -1.1434094, upper bound: 1.1381998
IS_B2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 23.97
Output dim: 7, lower bound: -1.1439891, upper bound: 1.1416352
IS_B2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 23.97
Output dim: 7, lower bound: -1.1439891, upper bound: 1.1416382
IS_B2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 23.97
Output dim: 7, lower bound: -1.1434094, upper bound: 1.1405541
IS_B2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 23.97
Output dim: 7, lower bound: -1.1434094, upper bound: 1.1405551
IS_B2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 23.97
Output dim: 7, lower bound: -1.1439891, upper bound: 1.1439889
IS_B2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 23.97
Output dim: 7, lower bound: -1.1439891, upper bound: 1.1439918
IS_B2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 23.97
Output dim: 7, lower bound: -1.1506431, upper bound: 1.1420912
IS_B2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 23.97
Output dim: 7, lower bound: -1.1524000, upper bound: 1.1471882
IS_B2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 23.97
Output dim: 7, lower bound: -1.1512031, upper bound: 1.1454693
IS_B2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 23.97
Output dim: 7, lower bound: -1.1529793, upper bound: 1.1506125
IS_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 23.97
Output dim: 7, lower bound: -1.1506431, upper bound: 1.1444496
IS_B2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 23.97
Output dim: 7, lower bound: -1.1524000, upper bound: 1.1495473
IS_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 23.97
Output dim: 7, lower bound: -1.1512031, upper bound: 1.1478478
IS_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 23.97
Output dim: 7, lower bound: -1.1529793, upper bound: 1.1529782
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=1.8950552940368652
rel_dist={7: [-1.1530177532447787, 1.1530172261927087]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6192
type: B, layer: 1, pos: 6192
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 6156
type: B, layer: 1, pos: 6156
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6192

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0263699, upper bound: 1.0197734
time: 4.71 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0298720, upper bound: 1.0298699
time: 4.63 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 9.53 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 9.53
Output dim: 7, lower bound: -1.0263699, upper bound: 1.0197734
IS_A2, status: Status.UNKNOWN, split count: 1, time: 9.53
Output dim: 7, lower bound: -1.0298720, upper bound: 1.0298699

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -15.7687855, -11.6553326, -15.8148575, -11.6261148, -2.3699102, 2.3957191
1: -7.1591153, -4.4076223, -7.1797771, -4.3930454, -2.6894512, 2.7028861
2: -8.7206879, -6.1691089, -8.7349367, -6.1531668, -2.3531961, 2.3457918
3: -4.9593029, -2.4537749, -4.9926190, -2.4290648, -2.2011051, 2.1996608
4: -7.9517303, -5.2917261, -7.9648185, -5.2793913, -1.8239689, 1.8301995
5: -6.2641287, -3.7503946, -6.3001766, -3.7199464, -2.1107025, 2.1297476
6: -14.3885374, -11.0027924, -14.4050684, -10.9833183, -2.1614451, 2.1502421
7: 2.2882204, 4.7757864, 2.2631035, 4.8066206, -1.7648630, 1.7515705
8: -1.2247856, 0.9384165, -1.2782235, 0.9727464, -1.8413682, 1.8511837
9: -8.7875423, -5.7901473, -8.8140335, -5.7540803, -2.0143089, 2.0045712

Time for backsubstitution: 14.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6156
type: A, layer: 1, pos: 6156
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6192
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0242605, upper bound: 1.0197641
time: 4.45 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0263606, upper bound: 1.0197642
time: 4.56 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -15.8640451, -11.6210775, -15.8640537, -11.6210804, -2.4167070, 2.4689350
1: -7.1961031, -4.3803387, -7.1961069, -4.3803372, -2.7579145, 2.7402887
2: -8.7408390, -6.1397657, -8.7408409, -6.1397619, -2.4048567, 2.3943253
3: -5.0245490, -2.4202378, -5.0245562, -2.4202373, -2.2745409, 2.2899013
4: -7.9703798, -5.2681632, -7.9703789, -5.2681627, -1.8653607, 1.8603468
5: -6.3388667, -3.7086473, -6.3388691, -3.7086451, -2.2124009, 2.2329750
6: -14.4134312, -10.9648504, -14.4134359, -10.9648466, -2.1933761, 2.1734786
7: 2.2540932, 4.8381004, 2.2540913, 4.8381057, -1.8196480, 1.7997286
8: -1.3332644, 0.9782419, -1.3332758, 0.9782424, -1.9317932, 1.9496901
9: -8.8183231, -5.7160378, -8.8183250, -5.7160344, -2.0824833, 2.0504224

Time for backsubstitution: 14.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6156
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0277098, upper bound: 1.0298646
time: 4.45 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0298634, upper bound: 1.0298612
time: 4.91 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 23.62 seconds
IS_A1_A1, status: Status.VERIFIED, split count: 2, time: 23.62
Output dim: 7, lower bound: -1.0242605, upper bound: 1.0197641
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 23.62
Output dim: 7, lower bound: -1.0263606, upper bound: 1.0197642
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 23.62
Output dim: 7, lower bound: -1.0277098, upper bound: 1.0298646
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 23.62
Output dim: 7, lower bound: -1.0298634, upper bound: 1.0298612

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -15.7873802, -11.6450481, -15.8148556, -11.6261177, -2.3914356, 2.4001119
1: -7.1647320, -4.3735170, -7.1797299, -4.3930464, -2.6897812, 2.7342339
2: -8.7351646, -6.1643968, -8.7349319, -6.1531754, -2.3686733, 2.3500605
3: -4.9836655, -2.4492612, -4.9926176, -2.4290786, -2.2257690, 2.2043917
4: -7.9732842, -5.2892489, -7.9648137, -5.2793970, -1.8461442, 1.8311920
5: -6.2833171, -3.7460666, -6.3001766, -3.7199488, -2.1324191, 2.1372142
6: -14.3987322, -10.9866371, -14.4050560, -10.9833183, -2.1716347, 2.1669092
7: 2.2840161, 4.7922587, 2.2631149, 4.8066187, -1.7674985, 1.7700546
8: -1.2314627, 0.9488640, -1.2782197, 0.9727440, -1.8515630, 1.8569188
9: -8.7966671, -5.7654543, -8.8140039, -5.7540808, -2.0135636, 2.0265496

Time for backsubstitution: 13.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6156
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6192
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6135

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6156

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0263521, upper bound: 1.0130392
time: 4.44 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0263521, upper bound: 1.0197555
time: 4.55 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -15.8617268, -11.6256685, -15.8626947, -11.6238804, -2.4075003, 2.4560771
1: -7.1722293, -4.3816123, -7.1819882, -4.3810835, -2.7329226, 2.7245483
2: -8.7386494, -6.1474495, -8.7395468, -6.1443114, -2.3964701, 2.3825431
3: -5.0217934, -2.4267561, -5.0229583, -2.4240932, -2.2679958, 2.2820137
4: -7.9667330, -5.2736549, -7.9682751, -5.2714214, -1.8555136, 1.8478894
5: -6.3371925, -3.7102919, -6.3379030, -3.7096145, -2.2094378, 2.2297890
6: -14.4055328, -10.9655228, -14.4087486, -10.9652376, -2.1801295, 2.1650982
7: 2.2597041, 4.8360295, 2.2574120, 4.8369074, -1.8127832, 1.7944772
8: -1.3312263, 0.9769106, -1.3320723, 0.9774756, -1.9267945, 1.9437342
9: -8.8006830, -5.7165194, -8.8078852, -5.7163172, -2.0632124, 2.0387096

Time for backsubstitution: 14.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6156
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6156

## Relational analysis of IS_A2_A1_A1

### Relational analysis result of IS_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0210376, upper bound: 1.0298561
time: 4.53 seconds

## Relational analysis of IS_A2_A1_A2

### Relational analysis result of IS_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0277013, upper bound: 1.0298561
time: 4.51 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -15.8826771, -11.6108723, -15.8640471, -11.6210861, -2.4383163, 2.4790344
1: -7.2019958, -4.3461890, -7.1960592, -4.3803387, -2.7578812, 2.7717218
2: -8.7552662, -6.1349716, -8.7408352, -6.1397696, -2.4203920, 2.3988810
3: -5.0493264, -2.4158285, -5.0245519, -2.4202509, -2.2981122, 2.2945952
4: -7.9919086, -5.2657194, -7.9703741, -5.2681651, -1.8875532, 1.8613181
5: -6.3580279, -3.7044034, -6.3388662, -3.7086477, -2.2341042, 2.2404556
6: -14.4232950, -10.9486542, -14.4134216, -10.9648485, -2.2047625, 2.1902194
7: 2.2500868, 4.8547006, 2.2541032, 4.8381042, -1.8221984, 1.8183129
8: -1.3401051, 0.9887390, -1.3332710, 0.9782405, -1.9422340, 1.9554002
9: -8.8274460, -5.6912737, -8.8182917, -5.7160325, -2.0824752, 2.0687561

Time for backsubstitution: 14.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6156
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6156

## Relational analysis of IS_A2_A2_A1

### Relational analysis result of IS_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0231908, upper bound: 1.0298561
time: 4.63 seconds

## Relational analysis of IS_A2_A2_A2

### Relational analysis result of IS_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0298548, upper bound: 1.0298528
time: 4.67 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 23.61 seconds
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 23.61
Output dim: 7, lower bound: -1.0263521, upper bound: 1.0130392
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 23.61
Output dim: 7, lower bound: -1.0263521, upper bound: 1.0197555
IS_A2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 23.61
Output dim: 7, lower bound: -1.0210376, upper bound: 1.0298561
IS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 23.61
Output dim: 7, lower bound: -1.0277013, upper bound: 1.0298561
IS_A2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 23.61
Output dim: 7, lower bound: -1.0231908, upper bound: 1.0298561
IS_A2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 23.61
Output dim: 7, lower bound: -1.0298548, upper bound: 1.0298528

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -15.7825632, -11.6659489, -15.7932472, -11.6843529, -2.3247185, 2.3307276
1: -7.1619339, -4.3778119, -7.1722894, -4.4088783, -2.6696548, 2.7207727
2: -8.7145014, -6.1668787, -8.6770649, -6.1665502, -2.3316636, 2.2898335
3: -4.9783878, -2.4523172, -4.9772367, -2.4391522, -2.2096701, 2.1872430
4: -7.9559340, -5.2950115, -7.9152284, -5.2997656, -1.7999637, 1.7783351
5: -6.2777367, -3.7668056, -6.2737656, -3.7771821, -2.0665898, 2.0826299
6: -14.3950129, -11.0056286, -14.3895674, -11.0360317, -2.1171000, 2.1239698
7: 2.3045621, 4.7895155, 2.3203788, 4.7933006, -1.7121594, 1.7116673
8: -1.2269261, 0.9469652, -1.2635202, 0.9671326, -1.8412118, 1.8389840
9: -8.7892551, -5.7672224, -8.7961407, -5.7600260, -1.9978623, 2.0032468

Time for backsubstitution: 14.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6192
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6135

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6140

## Relational analysis of IS_A1_A2_B1_A1

### Relational analysis result of IS_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0256404, upper bound: 1.0101120
time: 4.88 seconds

## Relational analysis of IS_A1_A2_B1_A2

### Relational analysis result of IS_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0263479, upper bound: 1.0130321
time: 4.72 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -15.7873802, -11.6450481, -15.8148508, -11.6261263, -2.3633976, 2.3939407
1: -7.1647320, -4.3735170, -7.1797299, -4.3930483, -2.6897793, 2.7302136
2: -8.7351646, -6.1643968, -8.7349310, -6.1531758, -2.3686719, 2.3130012
3: -4.9836655, -2.4492612, -4.9926147, -2.4290786, -2.2257681, 2.2045264
4: -7.9732842, -5.2892489, -7.9648089, -5.2793984, -1.8461437, 1.8034005
5: -6.2833171, -3.7460666, -6.3001747, -3.7199528, -2.1061115, 2.1372130
6: -14.3987322, -10.9866371, -14.4050541, -10.9833241, -2.1716313, 2.1803291
7: 2.2840161, 4.7922587, 2.2631197, 4.8066177, -1.7630904, 1.7340167
8: -1.2314627, 0.9488640, -1.2782183, 0.9727430, -1.8498020, 1.8559275
9: -8.7966671, -5.7654543, -8.8140001, -5.7540793, -2.0135636, 2.0304046

Time for backsubstitution: 14.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6192
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6135

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6140

## Relational analysis of IS_A1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0256404, upper bound: 1.0168514
time: 4.95 seconds

## Relational analysis of IS_A1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0263479, upper bound: 1.0197538
time: 4.32 seconds

## BFS IS instance: IS_A2_A1_A1

### Backsubstitution after applying IS history:
0: -15.8401070, -11.6838961, -15.8578815, -11.6447754, -2.3617823, 2.3893933
1: -7.1648588, -4.3974371, -7.1793318, -4.3853722, -2.7193432, 2.7045927
2: -8.6807852, -6.1608610, -8.7188635, -6.1466837, -2.3362703, 2.3481736
3: -5.0064750, -2.4368744, -5.0177326, -2.4269991, -2.2510862, 2.2659044
4: -7.9171815, -5.2940030, -7.9510126, -5.2767272, -1.8031626, 1.8147442
5: -6.3106971, -3.7675221, -6.3322754, -3.7303276, -2.1626031, 2.1639106
6: -14.3900700, -11.0181313, -14.4050512, -10.9841461, -2.1437378, 2.1099293
7: 2.3169608, 4.8227139, 2.2779732, 4.8344226, -1.7545633, 1.7519073
8: -1.3165359, 0.9713130, -1.3278279, 0.9755783, -1.9091492, 1.9336724
9: -8.7826796, -5.7224026, -8.8012791, -5.7180481, -2.0394826, 2.0240636

Time for backsubstitution: 14.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6156
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6140

## Relational analysis of IS_A2_A1_A1_B1

### Relational analysis result of IS_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0181674, upper bound: 1.0292070
time: 5.00 seconds

## Relational analysis of IS_A2_A1_A1_B2

### Relational analysis result of IS_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0210331, upper bound: 1.0298519
time: 4.39 seconds

## BFS IS instance: IS_A2_A1_A2

### Backsubstitution after applying IS history:
0: -15.8617249, -11.6256790, -15.8626947, -11.6238804, -2.4075003, 2.4280405
1: -7.1722288, -4.3816137, -7.1819882, -4.3810835, -2.7291384, 2.7245479
2: -8.7386494, -6.1474504, -8.7395468, -6.1443114, -2.3594093, 2.3825426
3: -5.0217924, -2.4267564, -5.0229583, -2.4240932, -2.2681313, 2.2820134
4: -7.9667277, -5.2736559, -7.9682751, -5.2714214, -1.8277230, 1.8478882
5: -6.3371916, -3.7102938, -6.3379030, -3.7096145, -2.2094378, 2.2034807
6: -14.4055328, -10.9655285, -14.4087486, -10.9652376, -2.1969051, 2.1650944
7: 2.2597084, 4.8360305, 2.2574120, 4.8369074, -1.7767477, 1.7944765
8: -1.3312263, 0.9769092, -1.3320723, 0.9774756, -1.9267936, 1.9418125
9: -8.8006802, -5.7165184, -8.8078852, -5.7163172, -2.0682473, 2.0387082

Time for backsubstitution: 14.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6140

## Relational analysis of IS_A2_A1_A2_B1

### Relational analysis result of IS_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0248423, upper bound: 1.0292068
time: 4.81 seconds

## Relational analysis of IS_A2_A1_A2_B2

### Relational analysis result of IS_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0276968, upper bound: 1.0298486
time: 4.60 seconds

## BFS IS instance: IS_A2_A2_A1

### Backsubstitution after applying IS history:
0: -15.8610678, -11.6690998, -15.8592358, -11.6419821, -2.3798590, 2.4123535
1: -7.1942043, -4.3619776, -7.1934013, -4.3846254, -2.7439413, 2.7506266
2: -8.6974258, -6.1486645, -8.7201557, -6.1421404, -2.3602004, 2.3642654
3: -5.0340114, -2.4264865, -5.0193276, -2.4231577, -2.2812917, 2.2779541
4: -7.9423451, -5.2871876, -7.9531298, -5.2734704, -1.8351750, 1.8221395
5: -6.3315430, -3.7617269, -6.3332453, -3.7293615, -2.1785383, 2.1744883
6: -14.4078188, -11.0013638, -14.4097252, -10.9837570, -2.1625273, 2.1349146
7: 2.3073292, 4.8407097, 2.2746668, 4.8356180, -1.7639351, 1.7648501
8: -1.3247399, 0.9831514, -1.3290253, 0.9763436, -1.9240789, 1.9453433
9: -8.8070440, -5.6971583, -8.8116922, -5.7177677, -2.0555706, 2.0542207

Time for backsubstitution: 14.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6156
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6140

## Relational analysis of IS_A2_A2_A1_B1

### Relational analysis result of IS_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0203234, upper bound: 1.0292069
time: 4.81 seconds

## Relational analysis of IS_A2_A2_A1_B2

### Relational analysis result of IS_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0231862, upper bound: 1.0298512
time: 4.50 seconds

## BFS IS instance: IS_A2_A2_A2

### Backsubstitution after applying IS history:
0: -15.8826771, -11.6108828, -15.8640471, -11.6210861, -2.4383144, 2.4509964
1: -7.2019954, -4.3461905, -7.1960592, -4.3803387, -2.7540894, 2.7709007
2: -8.7552652, -6.1349716, -8.7408352, -6.1397696, -2.3833318, 2.3988795
3: -5.0493255, -2.4158285, -5.0245519, -2.4202509, -2.2979133, 2.2945952
4: -7.9919052, -5.2657213, -7.9703741, -5.2681651, -1.8597636, 1.8613176
5: -6.3580275, -3.7044063, -6.3388662, -3.7086477, -2.2341046, 2.2141469
6: -14.4232950, -10.9486589, -14.4134216, -10.9648485, -2.2189884, 2.1902151
7: 2.2500930, 4.8547006, 2.2541032, 4.8381042, -1.7861600, 1.8164289
8: -1.3401041, 0.9887376, -1.3332710, 0.9782405, -1.9422336, 1.9534950
9: -8.8274460, -5.6912746, -8.8182917, -5.7160325, -2.0875106, 2.0687561

Time for backsubstitution: 14.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6140

## Relational analysis of IS_A2_A2_A2_B1

### Relational analysis result of IS_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0269908, upper bound: 1.0292075
time: 4.48 seconds

## Relational analysis of IS_A2_A2_A2_B2

### Relational analysis result of IS_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0298504, upper bound: 1.0298487
time: 4.41 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 23.15 seconds
IS_A1_A2_B1_A1, status: Status.VERIFIED, split count: 4, time: 23.15
Output dim: 7, lower bound: -1.0256404, upper bound: 1.0101120
IS_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 23.15
Output dim: 7, lower bound: -1.0263479, upper bound: 1.0130321
IS_A1_A2_B2_A1, status: Status.VERIFIED, split count: 4, time: 23.15
Output dim: 7, lower bound: -1.0256404, upper bound: 1.0168514
IS_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 23.15
Output dim: 7, lower bound: -1.0263479, upper bound: 1.0197538
IS_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.15
Output dim: 7, lower bound: -1.0181674, upper bound: 1.0292070
IS_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.15
Output dim: 7, lower bound: -1.0210331, upper bound: 1.0298519
IS_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.15
Output dim: 7, lower bound: -1.0248423, upper bound: 1.0292068
IS_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.15
Output dim: 7, lower bound: -1.0276968, upper bound: 1.0298486
IS_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.15
Output dim: 7, lower bound: -1.0203234, upper bound: 1.0292069
IS_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.15
Output dim: 7, lower bound: -1.0231862, upper bound: 1.0298512
IS_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.15
Output dim: 7, lower bound: -1.0269908, upper bound: 1.0292075
IS_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.15
Output dim: 7, lower bound: -1.0298504, upper bound: 1.0298487

## BFS IS instance: IS_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -15.7943668, -11.6638737, -15.7932434, -11.6843567, -2.3403683, 2.3336599
1: -7.1841354, -4.3681407, -7.1722856, -4.4088941, -2.6979074, 2.7304344
2: -8.7212248, -6.1435337, -8.6770592, -6.1665554, -2.3341718, 2.3115044
3: -4.9885020, -2.4122031, -4.9772115, -2.4391594, -2.2130747, 2.2108133
4: -8.0146952, -5.2875142, -7.9152236, -5.2997961, -1.8216715, 1.7790151
5: -6.2955313, -3.7394831, -6.2737589, -3.7771842, -2.0835090, 2.0956354
6: -14.4013300, -10.9596558, -14.3895578, -11.0360336, -2.1187139, 2.1426451
7: 2.2858677, 4.7949524, 2.3203835, 4.7932916, -1.7245705, 1.7164218
8: -1.2442887, 0.9502597, -1.2635140, 0.9671311, -1.8639970, 1.8382151
9: -8.8073778, -5.7311912, -8.7961235, -5.7600265, -2.0185390, 2.0100574

Time for backsubstitution: 14.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6156
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 6192
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6156

## Relational analysis of IS_A1_A2_B1_A2_A1

### Relational analysis result of IS_A1_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0196261, upper bound: 1.0130311
time: 5.09 seconds

## Relational analysis of IS_A1_A2_B1_A2_A2

### Relational analysis result of IS_A1_A2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0196261, upper bound: 1.0130323
time: 5.26 seconds

## BFS IS instance: IS_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -15.7992764, -11.6429577, -15.8148537, -11.6261282, -2.3791280, 2.3968968
1: -7.1869411, -4.3637958, -7.1797252, -4.3930635, -2.7180309, 2.7399216
2: -8.7419319, -6.1409912, -8.7349253, -6.1531792, -2.3793392, 2.3347411
3: -4.9938445, -2.4091105, -4.9925909, -2.4290862, -2.2292309, 2.2280414
4: -8.0317831, -5.2817540, -7.9648046, -5.2794247, -1.8690395, 1.8041327
5: -6.3012943, -3.7187495, -6.3001676, -3.7199557, -2.1232424, 2.1561780
6: -14.4051266, -10.9405098, -14.4050465, -10.9833317, -2.1733160, 2.1992877
7: 2.2653055, 4.7976966, 2.2631245, 4.8066106, -1.7755325, 1.7387800
8: -1.2489455, 0.9521441, -1.2782116, 0.9727430, -1.8726294, 1.8550947
9: -8.8147669, -5.7295127, -8.8139811, -5.7540841, -2.0342832, 2.0369058

Time for backsubstitution: 14.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6192
type: A, layer: 1, pos: 6156
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 451

## Relational analysis of IS_A1_A2_B2_A2_A1

### Relational analysis result of IS_A1_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0245272, upper bound: 1.0154298
time: 4.39 seconds

## Relational analysis of IS_A1_A2_B2_A2_A2

### Relational analysis result of IS_A1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0263473, upper bound: 1.0197530
time: 4.23 seconds

## BFS IS instance: IS_A2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -15.8392811, -11.6846790, -15.8563023, -11.6462030, -2.3590460, 2.3865032
1: -7.1631374, -4.4045973, -7.1760283, -4.3990688, -2.7046719, 2.6951885
2: -8.6772232, -6.1628933, -8.7121897, -6.1506052, -2.3233213, 2.3339715
3: -4.9989076, -2.4396482, -5.0033054, -2.4323065, -2.2376814, 2.2479649
4: -7.9144459, -5.3031435, -7.9457283, -5.2941556, -1.7819805, 1.7994798
5: -6.3074036, -3.7691078, -6.3258686, -3.7333872, -2.1549222, 2.1547754
6: -14.3830910, -11.0193834, -14.3917084, -10.9865694, -2.1346507, 2.0951211
7: 2.3189054, 4.8189845, 2.2816830, 4.8272572, -1.7457950, 1.7446547
8: -1.3139935, 0.9706039, -1.3227291, 0.9742107, -1.8989220, 1.9214647
9: -8.7754011, -5.7234750, -8.7874403, -5.7201223, -2.0301847, 2.0095403

Time for backsubstitution: 14.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6156
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 451

## Relational analysis of IS_A2_A1_A1_B1_B1

### Relational analysis result of IS_A2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0137706, upper bound: 1.0272637
time: 4.32 seconds

## Relational analysis of IS_A2_A1_A1_B1_B2

### Relational analysis result of IS_A2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0181666, upper bound: 1.0292064
time: 5.55 seconds

## BFS IS instance: IS_A2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -15.8401012, -11.6838999, -15.8695459, -11.6427059, -2.3647065, 2.4048986
1: -7.1648555, -4.3974533, -7.2006192, -4.3757215, -2.7290211, 2.7320356
2: -8.6807766, -6.1608648, -8.7256365, -6.1240907, -2.3574133, 2.3576558
3: -5.0064497, -2.4368811, -5.0278215, -2.3883054, -2.2762642, 2.2694449
4: -7.9171762, -5.2940316, -8.0098896, -5.2723341, -1.8006163, 1.8370894
5: -6.3106894, -3.7675259, -6.3502865, -3.7032418, -2.1754513, 2.1810646
6: -14.3900604, -11.0181351, -14.4113789, -10.9384689, -2.1633656, 2.1114988
7: 2.3169651, 4.8227072, 2.2592998, 4.8379745, -1.7574487, 1.7643516
8: -1.3165302, 0.9713120, -1.3431649, 0.9788256, -1.9084644, 1.9440086
9: -8.7826614, -5.7224040, -8.8128481, -5.6818843, -2.0504122, 2.0375867

Time for backsubstitution: 14.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6156

## Relational analysis of IS_A2_A1_A1_B2_B1

### Relational analysis result of IS_A2_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0210332, upper bound: 1.0231841
time: 6.31 seconds

## Relational analysis of IS_A2_A1_A1_B2_B2

### Relational analysis result of IS_A2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0210332, upper bound: 1.0298484
time: 6.19 seconds

## BFS IS instance: IS_A2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -15.8609037, -11.6264286, -15.8611202, -11.6252880, -2.4047184, 2.4251575
1: -7.1704865, -4.3888173, -7.1786642, -4.3948016, -2.7144346, 2.7150869
2: -8.7351665, -6.1494927, -8.7329178, -6.1482477, -2.3465133, 2.3684011
3: -5.0142694, -2.4295197, -5.0086164, -2.4294114, -2.2547669, 2.2641320
4: -7.9639578, -5.2828255, -7.9629750, -5.2889032, -1.8064656, 1.8333104
5: -6.3338070, -3.7118826, -6.3314352, -3.7126777, -2.2020402, 2.1943138
6: -14.3985271, -10.9667797, -14.3953915, -10.9676695, -2.1878109, 2.1502688
7: 2.2616377, 4.8322382, 2.2611132, 4.8296862, -1.7679300, 1.7876976
8: -1.3285151, 0.9761977, -1.3268442, 0.9761057, -1.9163604, 1.9294584
9: -8.7933712, -5.7175951, -8.7939453, -5.7183895, -2.0590243, 2.0240874

Time for backsubstitution: 14.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6156
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 451

## Relational analysis of IS_A2_A1_A2_B1_B1

### Relational analysis result of IS_A2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0204042, upper bound: 1.0272638
time: 4.44 seconds

## Relational analysis of IS_A2_A1_A2_B1_B2

### Relational analysis result of IS_A2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0248400, upper bound: 1.0292065
time: 5.28 seconds

## BFS IS instance: IS_A2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -15.8617239, -11.6256781, -15.8745270, -11.6217928, -2.4121242, 2.4436998
1: -7.1722231, -4.3816290, -7.2040777, -4.3713822, -2.7388725, 2.7529802
2: -8.7386417, -6.1474547, -8.7463636, -6.1208220, -2.3814101, 2.3931370
3: -5.0217667, -2.4267614, -5.0331225, -2.3839102, -2.2944489, 2.2856104
4: -7.9667215, -5.2736845, -8.0269222, -5.2639122, -1.8283424, 1.8842849
5: -6.3371859, -3.7102981, -6.3560915, -3.6822786, -2.2344322, 2.2207808
6: -14.4055223, -10.9655313, -14.4151554, -10.9192410, -2.2200668, 2.1667335
7: 2.2597117, 4.8360233, 2.2387199, 4.8422952, -1.7814169, 1.8153379
8: -1.3312197, 0.9769087, -1.3494515, 0.9807062, -1.9261236, 1.9534385
9: -8.8006639, -5.7165222, -8.8262482, -5.6802421, -2.0771742, 2.0611606

Time for backsubstitution: 14.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 451

## Relational analysis of IS_A2_A1_A2_B2_B1

### Relational analysis result of IS_A2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0232426, upper bound: 1.0278840
time: 4.34 seconds

## Relational analysis of IS_A2_A1_A2_B2_B2

### Relational analysis result of IS_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0276944, upper bound: 1.0298479
time: 4.98 seconds

## BFS IS instance: IS_A2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -15.8602476, -11.6698780, -15.8576574, -11.6434088, -2.3771222, 2.4094601
1: -7.1925173, -4.3691349, -7.1901054, -4.3983226, -2.7292862, 2.7402153
2: -8.6938353, -6.1506872, -8.7134724, -6.1460557, -2.3472681, 2.3500428
3: -5.0264158, -2.4292459, -5.0048966, -2.4284611, -2.2668431, 2.2599878
4: -7.9395881, -5.2963266, -7.9478569, -5.2908988, -1.8139901, 1.8060720
5: -6.3282456, -3.7633052, -6.3268361, -3.7324231, -2.1708555, 2.1653476
6: -14.4008474, -11.0026207, -14.3963900, -10.9861898, -2.1524692, 2.1200924
7: 2.3092475, 4.8369651, 2.2783675, 4.8284488, -1.7551761, 1.7576060
8: -1.3221941, 0.9824405, -1.3239179, 0.9749756, -1.9138441, 1.9331236
9: -8.7997684, -5.6982355, -8.7978544, -5.7198505, -2.0462732, 2.0397217

Time for backsubstitution: 14.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6156
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 451

## Relational analysis of IS_A2_A2_A1_B1_B1

### Relational analysis result of IS_A2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0159358, upper bound: 1.0272635
time: 4.63 seconds

## Relational analysis of IS_A2_A2_A1_B1_B2

### Relational analysis result of IS_A2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0203226, upper bound: 1.0292069
time: 4.76 seconds

## BFS IS instance: IS_A2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -15.8610649, -11.6690998, -15.8709021, -11.6399145, -2.3827817, 2.4245493
1: -7.1941996, -4.3619928, -7.2146978, -4.3749738, -2.7536240, 2.7594066
2: -8.6974211, -6.1486678, -8.7269182, -6.1195412, -2.3813543, 2.3698702
3: -5.0339856, -2.4264925, -5.0294018, -2.3844612, -2.2956316, 2.2814844
4: -7.9423389, -5.2872171, -8.0120077, -5.2690783, -1.8326278, 1.8436863
5: -6.3315363, -3.7617292, -6.3512592, -3.7022829, -2.1913874, 2.1916382
6: -14.4078083, -11.0013657, -14.4160595, -10.9380798, -2.1810091, 2.1364820
7: 2.3073335, 4.8407001, 2.2559862, 4.8391638, -1.7668126, 1.7772851
8: -1.3247318, 0.9831500, -1.3443584, 0.9795923, -1.9233956, 1.9556606
9: -8.8070259, -5.6971622, -8.8232584, -5.6816030, -2.0645390, 2.0662267

Time for backsubstitution: 14.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6156

## Relational analysis of IS_A2_A2_A1_B2_B1

### Relational analysis result of IS_A2_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0231858, upper bound: 1.0231876
time: 4.67 seconds

## Relational analysis of IS_A2_A2_A1_B2_B2

### Relational analysis result of IS_A2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0231862, upper bound: 1.0298512
time: 4.62 seconds

## BFS IS instance: IS_A2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -15.8818579, -11.6116285, -15.8624706, -11.6224928, -2.4355278, 2.4480991
1: -7.2001963, -4.3533926, -7.1927433, -4.3940587, -2.7394028, 2.7604613
2: -8.7517509, -6.1370072, -8.7341995, -6.1436996, -2.3704510, 2.3847318
3: -5.0417671, -2.4185824, -5.0102067, -2.4255662, -2.2835007, 2.2766566
4: -7.9891424, -5.2748857, -7.9651022, -5.2856493, -1.8385034, 1.8467319
5: -6.3546371, -3.7059879, -6.3323979, -3.7117157, -2.2267036, 2.2049739
6: -14.4162998, -10.9499149, -14.4000759, -10.9672861, -2.2089577, 2.1753750
7: 2.2519989, 4.8509016, 2.2577987, 4.8308802, -1.7773557, 1.8091428
8: -1.3373852, 0.9880266, -1.3280311, 0.9768715, -1.9317880, 1.9411290
9: -8.8201475, -5.6923528, -8.8043556, -5.7181149, -2.0776463, 2.0541420

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6156
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 451

## Relational analysis of IS_A2_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0225710, upper bound: 1.0272633
time: 4.43 seconds

## Relational analysis of IS_A2_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0269896, upper bound: 1.0292066
time: 5.07 seconds

## BFS IS instance: IS_A2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -15.8826752, -11.6108828, -15.8758812, -11.6190033, -2.4429388, 2.4628992
1: -7.2019911, -4.3462062, -7.2181578, -4.3706365, -2.7638283, 2.7806606
2: -8.7552595, -6.1349773, -8.7476454, -6.1162786, -2.4053469, 2.4094591
3: -5.0492988, -2.4158354, -5.0347013, -2.3800671, -2.3136029, 2.2981780
4: -7.9919004, -5.2657485, -8.0290422, -5.2606692, -1.8603935, 1.8919823
5: -6.3580198, -3.7044098, -6.3570571, -3.6813176, -2.2522087, 2.2314503
6: -14.4232864, -10.9486618, -14.4198341, -10.9188519, -2.2378693, 2.1918511
7: 2.2500973, 4.8546944, 2.2354045, 4.8434954, -1.7908192, 1.8288658
8: -1.3400974, 0.9887357, -1.3506508, 0.9814739, -1.9415631, 1.9651089
9: -8.8274260, -5.6912766, -8.8366232, -5.6799603, -2.0944901, 2.0896950

Time for backsubstitution: 14.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 451

## Relational analysis of IS_A2_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0254110, upper bound: 1.0278840
time: 4.36 seconds

## Relational analysis of IS_A2_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0298492, upper bound: 1.0298480
time: 6.57 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 25.22 seconds
IS_A1_A2_B1_A2_A1, status: Status.VERIFIED, split count: 5, time: 25.22
Output dim: 7, lower bound: -1.0196261, upper bound: 1.0130311
IS_A1_A2_B1_A2_A2, status: Status.VERIFIED, split count: 5, time: 25.22
Output dim: 7, lower bound: -1.0196261, upper bound: 1.0130323
IS_A1_A2_B2_A2_A1, status: Status.VERIFIED, split count: 5, time: 25.22
Output dim: 7, lower bound: -1.0245272, upper bound: 1.0154298
IS_A1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 25.22
Output dim: 7, lower bound: -1.0263473, upper bound: 1.0197530
IS_A2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 25.22
Output dim: 7, lower bound: -1.0137706, upper bound: 1.0272637
IS_A2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 25.22
Output dim: 7, lower bound: -1.0181666, upper bound: 1.0292064
IS_A2_A1_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 25.22
Output dim: 7, lower bound: -1.0210332, upper bound: 1.0231841
IS_A2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 25.22
Output dim: 7, lower bound: -1.0210332, upper bound: 1.0298484
IS_A2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 25.22
Output dim: 7, lower bound: -1.0204042, upper bound: 1.0272638
IS_A2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 25.22
Output dim: 7, lower bound: -1.0248400, upper bound: 1.0292065
IS_A2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 25.22
Output dim: 7, lower bound: -1.0232426, upper bound: 1.0278840
IS_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 25.22
Output dim: 7, lower bound: -1.0276944, upper bound: 1.0298479
IS_A2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 25.22
Output dim: 7, lower bound: -1.0159358, upper bound: 1.0272635
IS_A2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 25.22
Output dim: 7, lower bound: -1.0203226, upper bound: 1.0292069
IS_A2_A2_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 25.22
Output dim: 7, lower bound: -1.0231858, upper bound: 1.0231876
IS_A2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 25.22
Output dim: 7, lower bound: -1.0231862, upper bound: 1.0298512
IS_A2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 25.22
Output dim: 7, lower bound: -1.0225710, upper bound: 1.0272633
IS_A2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 25.22
Output dim: 7, lower bound: -1.0269896, upper bound: 1.0292066
IS_A2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 25.22
Output dim: 7, lower bound: -1.0254110, upper bound: 1.0278840
IS_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 25.22
Output dim: 7, lower bound: -1.0298492, upper bound: 1.0298480

## BFS IS instance: IS_A1_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -15.7992764, -11.6429596, -15.8148499, -11.6261272, -2.3715110, 2.3953156
1: -7.1869388, -4.3637967, -7.1797247, -4.3930645, -2.7018290, 2.7366071
2: -8.7419319, -6.1409922, -8.7349262, -6.1531811, -2.3793392, 2.3199005
3: -4.9938450, -2.4091136, -4.9925890, -2.4290853, -2.2292309, 2.2193377
4: -8.0317841, -5.2817564, -7.9648042, -5.2794247, -1.8688455, 1.8034685
5: -6.3012929, -3.7187490, -6.3001657, -3.7199552, -2.1175489, 2.1547940
6: -14.4051228, -10.9405098, -14.4050455, -10.9833269, -2.1492796, 2.1952677
7: 2.2653046, 4.7976971, 2.2631235, 4.8066101, -1.7727973, 1.7240481
8: -1.2489448, 0.9521446, -1.2782116, 0.9727421, -1.8721762, 1.8530824
9: -8.8147659, -5.7295151, -8.8139820, -5.7540841, -2.0306277, 2.0392568

Time for backsubstitution: 14.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6192
type: A, layer: 1, pos: 6156
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6135

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6135

## Relational analysis of IS_A1_A2_B2_A2_A2_A1

### Relational analysis result of IS_A1_A2_B2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0261289, upper bound: 1.0171997
time: 4.40 seconds

## Relational analysis of IS_A1_A2_B2_A2_A2_A2

### Relational analysis result of IS_A1_A2_B2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0263428, upper bound: 1.0197485
time: 4.36 seconds

## BFS IS instance: IS_A2_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -15.8289165, -11.6881676, -15.8324947, -11.6536169, -2.3344121, 2.3587756
1: -7.1503811, -4.4065652, -7.1466875, -4.4065075, -2.6839085, 2.6622272
2: -8.6755495, -6.1743050, -8.7053242, -6.1773500, -2.2946706, 2.3136878
3: -4.9972816, -2.4516735, -4.9967303, -2.4607821, -2.2073455, 2.2246091
4: -7.9124842, -5.3096566, -7.9404554, -5.3076000, -1.7647514, 1.7848775
5: -6.2930956, -3.7706146, -6.2929916, -3.7376406, -2.1291709, 2.1190333
6: -14.3650379, -11.0215378, -14.3502922, -10.9950027, -2.0941701, 2.0500329
7: 2.3232045, 4.8098421, 2.2937050, 4.8063402, -1.7187607, 1.7124522
8: -1.3134170, 0.9663906, -1.3203073, 0.9649343, -1.8879709, 1.9133298
9: -8.7733173, -5.7323346, -8.7803345, -5.7404699, -2.0024352, 1.9841275

Time for backsubstitution: 14.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6156
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6156

## Relational analysis of IS_A2_A1_A1_B1_B1_B1

### Relational analysis result of IS_A2_A1_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0137706, upper bound: 1.0206288
time: 4.53 seconds

## Relational analysis of IS_A2_A1_A1_B1_B1_B2

### Relational analysis result of IS_A2_A1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0137706, upper bound: 1.0272637
time: 4.55 seconds

## BFS IS instance: IS_A2_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -15.8392811, -11.6846809, -15.8562965, -11.6462049, -2.3574638, 2.3788843
1: -7.1631379, -4.4045963, -7.1760254, -4.3990688, -2.7046728, 2.6789880
2: -8.6772213, -6.1628933, -8.7121897, -6.1506047, -2.3084803, 2.3339720
3: -4.9989080, -2.4396474, -5.0033050, -2.4323077, -2.2290392, 2.2479644
4: -7.9144459, -5.3031440, -7.9457283, -5.2941561, -1.7813625, 1.7992855
5: -6.3074045, -3.7691078, -6.3258657, -3.7333870, -2.1535382, 2.1490555
6: -14.3830919, -11.0193815, -14.3917074, -10.9865685, -2.1308289, 2.0711198
7: 2.3189049, 4.8189840, 2.2816830, 4.8272538, -1.7310548, 1.7419195
8: -1.3139939, 0.9706044, -1.3227296, 0.9742112, -1.8970213, 1.9210165
9: -8.7754002, -5.7234783, -8.7874384, -5.7201252, -2.0346055, 2.0074894

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 6156
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6135

## Relational analysis of IS_A2_A1_A1_B1_B2_B1

### Relational analysis result of IS_A2_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0156319, upper bound: 1.0289635
time: 4.19 seconds

## Relational analysis of IS_A2_A1_A1_B1_B2_B2

### Relational analysis result of IS_A2_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0181619, upper bound: 1.0292022
time: 6.35 seconds

## BFS IS instance: IS_A2_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -15.8401012, -11.6838999, -15.8741560, -11.6220207, -2.3670242, 2.4076343
1: -7.1648555, -4.3974533, -7.2030730, -4.3720016, -2.7329645, 2.7356744
2: -8.6807766, -6.1608648, -8.7457647, -6.1218934, -2.3597164, 2.3593001
3: -5.0064497, -2.4368811, -5.0322952, -2.3856759, -2.2791457, 2.2732329
4: -7.9171762, -5.2940316, -8.0268307, -5.2678652, -1.8036366, 1.8391024
5: -6.3106894, -3.7675259, -6.3554583, -3.6826124, -2.1815488, 2.1852851
6: -14.3900604, -11.0181351, -14.4149837, -10.9201660, -2.1716862, 2.1120648
7: 2.3169651, 4.8227072, 2.2390766, 4.8398724, -1.7590547, 1.7666101
8: -1.3165302, 0.9713120, -1.3463631, 0.9806695, -1.9111209, 1.9461036
9: -8.7826614, -5.7224040, -8.8175297, -5.6802974, -2.0514965, 2.0426612

Time for backsubstitution: 14.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 451

## Relational analysis of IS_A2_A1_A1_B2_B2_B1

### Relational analysis result of IS_A2_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0166102, upper bound: 1.0278840
time: 4.32 seconds

## Relational analysis of IS_A2_A1_A1_B2_B2_B2

### Relational analysis result of IS_A2_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0210323, upper bound: 1.0298481
time: 4.83 seconds

## BFS IS instance: IS_A2_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -15.8505135, -11.6299152, -15.8372984, -11.6327019, -2.3856778, 2.3974228
1: -7.1577024, -4.3908443, -7.1492090, -4.4022899, -2.6936007, 2.6819344
2: -8.7334976, -6.1609244, -8.7260571, -6.1750975, -2.3177629, 2.3504105
3: -5.0126405, -2.4416003, -5.0020518, -2.4580700, -2.2242513, 2.2403646
4: -7.9620142, -5.2894607, -7.9577088, -5.3027825, -1.7888298, 1.8202527
5: -6.3194413, -3.7133982, -6.2985530, -3.7169676, -2.1820893, 2.1585984
6: -14.3804436, -10.9689598, -14.3539686, -10.9761400, -2.1505613, 2.1051662
7: 2.2659335, 4.8230300, 2.2731314, 4.8085146, -1.7406373, 1.7629585
8: -1.3278799, 0.9720397, -1.3241658, 0.9668746, -1.9053588, 1.9211926
9: -8.7910070, -5.7264514, -8.7858229, -5.7387362, -2.0309954, 1.9974759

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6156
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 451

## Relational analysis of IS_A2_A1_A2_B1_B1_A1

### Relational analysis result of IS_A2_A1_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0204042, upper bound: 1.0247848
time: 4.34 seconds

## Relational analysis of IS_A2_A1_A2_B1_B1_A2

### Relational analysis result of IS_A2_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0204042, upper bound: 1.0272639
time: 4.29 seconds

## BFS IS instance: IS_A2_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -15.8609028, -11.6264277, -15.8611155, -11.6252899, -2.4042921, 2.4175329
1: -7.1704874, -4.3888173, -7.1786609, -4.3948011, -2.7144337, 2.6988869
2: -8.7351656, -6.1494932, -8.7329178, -6.1482487, -2.3316741, 2.3683996
3: -5.0142713, -2.4295201, -5.0086164, -2.4294138, -2.2461228, 2.2641320
4: -7.9639578, -5.2828259, -7.9629755, -5.2889042, -1.8058162, 1.8332107
5: -6.3338065, -3.7118826, -6.3314342, -3.7126784, -2.2015204, 2.1885939
6: -14.3985291, -10.9667788, -14.3953915, -10.9676714, -2.1871533, 2.1262679
7: 2.2616386, 4.8322387, 2.2611132, 4.8296857, -1.7531588, 1.7872002
8: -1.3285160, 0.9761972, -1.3268433, 0.9761071, -1.9145002, 1.9290087
9: -8.7933693, -5.7175961, -8.7939453, -5.7183914, -2.0625601, 2.0220482

Time for backsubstitution: 14.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6135

## Relational analysis of IS_A2_A1_A2_B1_B2_B1

### Relational analysis result of IS_A2_A1_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0222789, upper bound: 1.0169549
time: 6.56 seconds

## Relational analysis of IS_A2_A1_A2_B1_B2_B2

### Relational analysis result of IS_A2_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0248353, upper bound: 1.0292024
time: 4.40 seconds

## BFS IS instance: IS_A2_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -15.8513346, -11.6291647, -15.8506718, -11.6291971, -2.3930807, 2.4159055
1: -7.1594367, -4.3836555, -7.1746311, -4.3788452, -2.7180719, 2.7198415
2: -8.7369719, -6.1588855, -8.7395525, -6.1476569, -2.3527489, 2.3751836
3: -5.0201378, -2.4388418, -5.0265694, -2.4125180, -2.2638469, 2.2620158
4: -7.9647765, -5.2803140, -8.0215921, -5.2777233, -1.8106637, 1.8696569
5: -6.3228374, -3.7118139, -6.3231707, -3.6865704, -2.2107263, 2.1849089
6: -14.3874407, -10.9677153, -14.3737602, -10.9277401, -2.1794403, 2.1216693
7: 2.2640109, 4.8268194, 2.2507524, 4.8211513, -1.7541244, 1.7831755
8: -1.3305836, 0.9727507, -1.3467517, 0.9714327, -1.9150758, 1.9451599
9: -8.7983027, -5.7253766, -8.8181391, -5.7008667, -2.0491686, 2.0344646

Time for backsubstitution: 14.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 451

## Relational analysis of IS_A2_A1_A2_B2_B1_A1

### Relational analysis result of IS_A2_A1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0232426, upper bound: 1.0254098
time: 4.69 seconds

## Relational analysis of IS_A2_A1_A2_B2_B1_A2

### Relational analysis result of IS_A2_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0232426, upper bound: 1.0278841
time: 4.71 seconds

## BFS IS instance: IS_A2_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -15.8617229, -11.6256790, -15.8745251, -11.6217957, -2.4116955, 2.4360752
1: -7.1722240, -4.3816290, -7.2040744, -4.3713818, -2.7388716, 2.7367802
2: -8.7386427, -6.1474562, -8.7463636, -6.1208248, -2.3665705, 2.3931370
3: -5.0217667, -2.4267614, -5.0331235, -2.3839097, -2.2857466, 2.2856107
4: -7.9667225, -5.2736845, -8.0269222, -5.2639136, -1.8276811, 1.8840914
5: -6.3371849, -3.7102973, -6.3560886, -3.6822791, -2.2339144, 2.2150595
6: -14.4055233, -10.9655313, -14.4151525, -10.9192419, -2.2160459, 2.1427324
7: 2.2597132, 4.8360229, 2.2387199, 4.8422952, -1.7666411, 1.8126025
8: -1.3312197, 0.9769087, -1.3494511, 0.9807053, -1.9242086, 1.9529834
9: -8.8006639, -5.7165232, -8.8262482, -5.6802425, -2.0794182, 2.0591321

Time for backsubstitution: 14.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6156
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6135

## Relational analysis of IS_A2_A1_A2_B2_B2_B1

### Relational analysis result of IS_A2_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0251246, upper bound: 1.0295932
time: 4.35 seconds

## Relational analysis of IS_A2_A1_A2_B2_B2_B2

### Relational analysis result of IS_A2_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0276897, upper bound: 1.0298434
time: 4.99 seconds

## BFS IS instance: IS_A2_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -15.8498764, -11.6733685, -15.8338490, -11.6508217, -2.3524947, 2.3816166
1: -7.1797528, -4.3711038, -7.1607628, -4.4057646, -2.7085133, 2.7072306
2: -8.6921673, -6.1620955, -8.7066135, -6.1728001, -2.3186307, 2.3259020
3: -5.0247831, -2.4412622, -4.9983187, -2.4569302, -2.2364712, 2.2366920
4: -7.9376378, -5.3028331, -7.9425821, -5.3043323, -1.7967601, 1.7914710
5: -6.3139238, -3.7648110, -6.2939587, -3.7366745, -2.1451144, 2.1296039
6: -14.3828087, -11.0047789, -14.3549852, -10.9946213, -2.1117971, 2.0750017
7: 2.3135538, 4.8278241, 2.2903895, 4.8075337, -1.7281342, 1.7253966
8: -1.3216109, 0.9782243, -1.3214989, 0.9656978, -1.9028888, 1.9249892
9: -8.7976799, -5.7070885, -8.7907524, -5.7401986, -2.0185184, 2.0143461

Time for backsubstitution: 14.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6156
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6156

## Relational analysis of IS_A2_A2_A1_B1_B1_B1

### Relational analysis result of IS_A2_A2_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0159358, upper bound: 1.0206286
time: 4.49 seconds

## Relational analysis of IS_A2_A2_A1_B1_B1_B2

### Relational analysis result of IS_A2_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0159358, upper bound: 1.0272635
time: 4.65 seconds

## BFS IS instance: IS_A2_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -15.8602476, -11.6698780, -15.8576584, -11.6434107, -2.3755398, 2.4018388
1: -7.1925159, -4.3691359, -7.1901016, -4.3983235, -2.7292843, 2.7239499
2: -8.6938353, -6.1506882, -8.7134724, -6.1460581, -2.3324308, 2.3494718
3: -5.0264158, -2.4292445, -5.0048966, -2.4284630, -2.2581394, 2.2599876
4: -7.9395885, -5.2963276, -7.9478531, -5.2908993, -1.8133607, 1.8058774
5: -6.3282442, -3.7633047, -6.3268337, -3.7324233, -2.1694710, 2.1596217
6: -14.4008465, -11.0026226, -14.3963909, -10.9861870, -2.1484580, 2.0960913
7: 2.3092470, 4.8369660, 2.2783694, 4.8284473, -1.7404208, 1.7548709
8: -1.3221941, 0.9824414, -1.3239179, 0.9749761, -1.9119434, 1.9326749
9: -8.7997675, -5.6982360, -8.7978544, -5.7198510, -2.0497160, 2.0360661

Time for backsubstitution: 14.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6156

## Relational analysis of IS_A2_A2_A1_B1_B2_B1

### Relational analysis result of IS_A2_A2_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0203226, upper bound: 1.0225403
time: 4.61 seconds

## Relational analysis of IS_A2_A2_A1_B1_B2_B2

### Relational analysis result of IS_A2_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0203226, upper bound: 1.0292069
time: 4.73 seconds

## BFS IS instance: IS_A2_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -15.8610649, -11.6690998, -15.8755093, -11.6192312, -2.3850994, 2.4223692
1: -7.1941996, -4.3619928, -7.2171578, -4.3712559, -2.7575645, 2.7609744
2: -8.6974211, -6.1486678, -8.7470446, -6.1173458, -2.3836536, 2.3715119
3: -5.0339856, -2.4264925, -5.0338759, -2.3818281, -2.2985184, 2.2852724
4: -7.9423389, -5.2872171, -8.0289497, -5.2646122, -1.8310413, 1.8456998
5: -6.3315363, -3.7617292, -6.3564243, -3.6816516, -2.1974845, 2.1958625
6: -14.4078083, -11.0013657, -14.4196634, -10.9197731, -2.1893315, 2.1355090
7: 2.3073335, 4.8407001, 2.2357616, 4.8410602, -1.7684238, 1.7795453
8: -1.3247318, 0.9831500, -1.3475556, 0.9814377, -1.9260511, 1.9577551
9: -8.8070259, -5.6971622, -8.8279324, -5.6800137, -2.0656233, 2.0703056

Time for backsubstitution: 14.23 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=1.8196511268615723
rel_dist={7: [-1.029880698625119, 1.029879313881131]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2423.97 seconds
