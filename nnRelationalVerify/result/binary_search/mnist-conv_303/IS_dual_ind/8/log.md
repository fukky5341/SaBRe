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
execution time: IAR + LP analysis = 13.87 + 32.26 = 46.13 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3553.87 seconds, max iter: 100)

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
Binary search time: 197.65 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual_ind) starts
Time budget: 3356.22 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6192
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6192

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4777228, upper bound: 1.4583234
time: 4.45 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4841240, upper bound: 1.4841229
time: 4.13 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8.76 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 8.76
Output dim: 7, lower bound: -1.4777228, upper bound: 1.4583234
IS_A2, status: Status.UNKNOWN, split count: 1, time: 8.76
Output dim: 7, lower bound: -1.4841240, upper bound: 1.4841229

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -15.7687855, -11.6553326, -15.8392849, -11.6234875, -2.9248013, 2.9726329
1: -7.1591153, -4.4076223, -7.1877794, -4.3867526, -2.7723627, 2.7801571
2: -8.7206879, -6.1691089, -8.7379065, -6.1465340, -2.5741539, 2.5687976
3: -4.9593029, -2.4537749, -5.0084758, -2.4245584, -2.4454131, 2.4590654
4: -7.9517303, -5.2917261, -7.9676423, -5.2738123, -2.1669874, 2.1686156
5: -6.2641287, -3.7503946, -6.3194108, -3.7140961, -2.3779607, 2.4186463
6: -14.3885374, -11.0027924, -14.4092875, -10.9741488, -2.6212473, 2.6031780
7: 2.2882204, 4.7757864, 2.2585292, 4.8222275, -2.0816388, 2.0556371
8: -1.2247856, 0.9384165, -1.3055716, 0.9755721, -2.0776081, 2.1109242
9: -8.7875423, -5.7901473, -8.8161774, -5.7351828, -2.4491282, 2.4225166

Time for backsubstitution: 14.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6192
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6192

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4583236, upper bound: 1.4583234
time: 4.55 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4583236, upper bound: 1.4583236
time: 6.00 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -15.8640451, -11.6210775, -15.8640556, -11.6210804, -2.9801784, 3.0224767
1: -7.1961031, -4.3803387, -7.1961079, -4.3803363, -2.8157668, 2.8157692
2: -8.7408390, -6.1397657, -8.7408409, -6.1397600, -2.6010790, 2.6010752
3: -5.0245490, -2.4202378, -5.0245595, -2.4202366, -2.5157423, 2.5323892
4: -7.9703798, -5.2681632, -7.9703813, -5.2681599, -2.2019091, 2.1964862
5: -6.3388667, -3.7086473, -6.3388715, -3.7086442, -2.4796739, 2.5019705
6: -14.4134312, -10.9648504, -14.4134359, -10.9648418, -2.6426477, 2.6285655
7: 2.2540932, 4.8381004, 2.2540903, 4.8381100, -2.1212654, 2.1051316
8: -1.3332644, 0.9782419, -1.3332825, 0.9782434, -2.1700807, 2.1878631
9: -8.8183231, -5.7160378, -8.8183250, -5.7160292, -2.4985056, 2.4725432

Time for backsubstitution: 14.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6192
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6192

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4583236, upper bound: 1.4777225
time: 4.28 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4583236, upper bound: 1.4841241
time: 4.29 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 22.89 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 22.89
Output dim: 7, lower bound: -1.4583236, upper bound: 1.4583234
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 22.89
Output dim: 7, lower bound: -1.4583236, upper bound: 1.4583236
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 22.89
Output dim: 7, lower bound: -1.4583236, upper bound: 1.4777225
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 22.89
Output dim: 7, lower bound: -1.4583236, upper bound: 1.4841241

## BFS IS instance: IS_A1_B1

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

Time for backsubstitution: 14.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6156

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4426337, upper bound: 1.4583035
time: 4.72 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4583073, upper bound: 1.4583032
time: 4.35 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -15.7687855, -11.6553326, -15.8640451, -11.6210775, -2.9260721, 2.9789174
1: -7.1591153, -4.4076223, -7.1961031, -4.3803387, -2.7787766, 2.7884808
2: -8.7206879, -6.1691089, -8.7408390, -6.1397657, -2.5809221, 2.5717301
3: -4.9593029, -2.4537749, -5.0245490, -2.4202378, -2.4462066, 2.4695361
4: -7.9517303, -5.2917261, -7.9703798, -5.2681632, -2.1748018, 2.1701379
5: -6.2641287, -3.7503946, -6.3388667, -3.7086473, -2.3791242, 2.4377608
6: -14.3885374, -11.0027924, -14.4134312, -10.9648504, -2.6305151, 2.6043918
7: 2.2882204, 4.7757864, 2.2540932, 4.8381004, -2.0895209, 2.0581391
8: -1.2247856, 0.9384165, -1.3332644, 0.9782419, -2.0795603, 2.1133976
9: -8.7875423, -5.7901473, -8.8183231, -5.7160378, -2.4527583, 2.4242899

Time for backsubstitution: 14.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6156

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4426338, upper bound: 1.4583035
time: 4.42 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4583073, upper bound: 1.4583051
time: 4.06 seconds

## BFS IS instance: IS_A2_B1

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

Time for backsubstitution: 14.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6156

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4426295, upper bound: 1.4777015
time: 4.92 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4583031, upper bound: 1.4777011
time: 4.45 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -15.8640451, -11.6210775, -15.8640451, -11.6210775, -2.9801779, 2.9801774
1: -7.1961031, -4.3803387, -7.1961031, -4.3803387, -2.8157644, 2.8157644
2: -8.7408390, -6.1397657, -8.7408390, -6.1397657, -2.6010733, 2.6010733
3: -5.0245490, -2.4202378, -5.0245490, -2.4202378, -2.5323815, 2.5323815
4: -7.9703798, -5.2681632, -7.9703798, -5.2681632, -2.2019067, 2.2019062
5: -6.3388667, -3.7086473, -6.3388667, -3.7086473, -2.5019665, 2.5019662
6: -14.4134312, -10.9648504, -14.4134312, -10.9648504, -2.6285620, 2.6285622
7: 2.2540932, 4.8381004, 2.2540932, 4.8381004, -2.1051297, 2.1051302
8: -1.3332644, 0.9782419, -1.3332644, 0.9782419, -2.1700802, 2.1700802
9: -8.8183231, -5.7160378, -8.8183231, -5.7160378, -2.4725413, 2.4725418

Time for backsubstitution: 14.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6156

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4426296, upper bound: 1.4841038
time: 4.60 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4583031, upper bound: 1.4841039
time: 4.06 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 23.02 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 23.02
Output dim: 7, lower bound: -1.4426337, upper bound: 1.4583035
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 23.02
Output dim: 7, lower bound: -1.4583073, upper bound: 1.4583032
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 23.02
Output dim: 7, lower bound: -1.4426338, upper bound: 1.4583035
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 23.02
Output dim: 7, lower bound: -1.4583073, upper bound: 1.4583051
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 23.02
Output dim: 7, lower bound: -1.4426295, upper bound: 1.4777015
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 23.02
Output dim: 7, lower bound: -1.4583031, upper bound: 1.4777011
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 23.02
Output dim: 7, lower bound: -1.4426296, upper bound: 1.4841038
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 23.02
Output dim: 7, lower bound: -1.4583031, upper bound: 1.4841039

## BFS IS instance: IS_A1_B1_A1

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

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4426218, upper bound: 1.4553192
time: 4.60 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4426219, upper bound: 1.4582953
time: 4.60 seconds

## BFS IS instance: IS_A1_B1_A2

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

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6156

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4583076, upper bound: 1.4426333
time: 4.30 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4583076, upper bound: 1.4583073
time: 4.34 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -15.7471771, -11.7135725, -15.8619308, -11.6303768, -2.8823707, 2.9151099
1: -7.1517663, -4.4234872, -7.1948729, -4.3822155, -2.7695508, 2.7713857
2: -8.6628361, -6.1823931, -8.7316380, -6.1408606, -2.5219755, 2.5492449
3: -4.9438305, -2.4636350, -5.0222306, -2.4216063, -2.4309745, 2.4561050
4: -7.9020762, -5.3117576, -7.9627156, -5.2706814, -2.1243081, 2.1467535
5: -6.2378035, -3.8075852, -6.3364110, -3.7178676, -2.3462105, 2.3752291
6: -14.3730278, -11.0555315, -14.4118166, -10.9732819, -2.5977817, 2.5502594
7: 2.3454614, 4.7627077, 2.2632313, 4.8369050, -2.0325785, 2.0262604
8: -1.2102590, 0.9327888, -1.3312855, 0.9773951, -2.0628572, 2.1056721
9: -8.7705936, -5.7961316, -8.8149862, -5.7168112, -2.4316587, 2.4133835

Time for backsubstitution: 14.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6156

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4620413, upper bound: 1.4426292
time: 4.77 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4620402, upper bound: 1.4583031
time: 5.18 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -15.7687864, -11.6553373, -15.8640451, -11.6210775, -2.9260712, 2.9562066
1: -7.1591148, -4.4076223, -7.1961031, -4.3803387, -2.7787762, 2.7884808
2: -8.7206879, -6.1691084, -8.7408390, -6.1397657, -2.5809221, 2.5717306
3: -4.9593015, -2.4537759, -5.0245490, -2.4202378, -2.4470086, 2.4690788
4: -7.9517260, -5.2917271, -7.9703798, -5.2681632, -2.1522989, 2.1701369
5: -6.2641282, -3.7503965, -6.3388667, -3.7086473, -2.3791232, 2.4164524
6: -14.3885374, -11.0028000, -14.4134312, -10.9648504, -2.6469440, 2.6043875
7: 2.2882261, 4.7757874, 2.2540932, 4.8381004, -2.0603430, 2.0581388
8: -1.2247832, 0.9384151, -1.3332644, 0.9782419, -2.0785656, 2.1121485
9: -8.7875395, -5.7901487, -8.8183231, -5.7160378, -2.4574227, 2.4242895

Time for backsubstitution: 14.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6156

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4777015, upper bound: 1.4426294
time: 4.36 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4777015, upper bound: 1.4583033
time: 4.17 seconds

## BFS IS instance: IS_A2_B1_A1

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

Time for backsubstitution: 14.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4426177, upper bound: 1.4747143
time: 4.76 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4426176, upper bound: 1.4776888
time: 4.43 seconds

## BFS IS instance: IS_A2_B1_A2

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

Time for backsubstitution: 14.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6156

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4583035, upper bound: 1.4620400
time: 4.47 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4583034, upper bound: 1.4777012
time: 4.19 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -15.8424368, -11.6793060, -15.8619308, -11.6303768, -2.9492545, 2.9160991
1: -7.1887221, -4.3961544, -7.1948729, -4.3822155, -2.8065066, 2.7987185
2: -8.6829739, -6.1531687, -8.7316380, -6.1408606, -2.5421133, 2.5784693
3: -5.0092278, -2.4303646, -5.0222306, -2.4216063, -2.5173206, 2.5186906
4: -7.9208684, -5.2885032, -7.9627156, -5.2706814, -2.1515141, 2.1781721
5: -6.3123989, -3.7658811, -6.3364110, -3.7178676, -2.4675472, 2.4386520
6: -14.3979731, -11.0174685, -14.4118166, -10.9732819, -2.6056595, 2.5745270
7: 2.3113546, 4.8247833, 2.2632313, 4.8369050, -2.0479918, 2.0780966
8: -1.3185802, 0.9726424, -1.3312855, 0.9773951, -2.1539283, 2.1623337
9: -8.8003502, -5.7219272, -8.8149862, -5.7168112, -2.4497194, 2.4616899

Time for backsubstitution: 13.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6156

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4554282, upper bound: 1.4685763
time: 3.60 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4554282, upper bound: 1.4841036
time: 3.83 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -15.8640432, -11.6210861, -15.8640451, -11.6210775, -2.9801760, 2.9574604
1: -7.1961031, -4.3803391, -7.1961031, -4.3803387, -2.8157644, 2.8157640
2: -8.7408390, -6.1397667, -8.7408390, -6.1397657, -2.6010733, 2.6010723
3: -5.0245476, -2.4202385, -5.0245490, -2.4202378, -2.5331836, 2.5323801
4: -7.9703741, -5.2681651, -7.9703798, -5.2681632, -2.1794028, 2.2019057
5: -6.3388648, -3.7086492, -6.3388667, -3.7086473, -2.5019665, 2.4806576
6: -14.4134331, -10.9648571, -14.4134312, -10.9648504, -2.6480360, 2.6285582
7: 2.2540984, 4.8381004, 2.2540932, 4.8381004, -2.0759439, 2.1051297
8: -1.3332634, 0.9782405, -1.3332644, 0.9782419, -2.1700788, 2.1690772
9: -8.8183212, -5.7160392, -8.8183231, -5.7160378, -2.4783792, 2.4725418

Time for backsubstitution: 14.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6156

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4709486, upper bound: 1.4685757
time: 4.24 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4709485, upper bound: 1.4841034
time: 4.24 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 22.96 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.96
Output dim: 7, lower bound: -1.4426218, upper bound: 1.4553192
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.96
Output dim: 7, lower bound: -1.4426219, upper bound: 1.4582953
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.96
Output dim: 7, lower bound: -1.4583076, upper bound: 1.4426333
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.96
Output dim: 7, lower bound: -1.4583076, upper bound: 1.4583073
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.96
Output dim: 7, lower bound: -1.4620413, upper bound: 1.4426292
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.96
Output dim: 7, lower bound: -1.4620402, upper bound: 1.4583031
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.96
Output dim: 7, lower bound: -1.4777015, upper bound: 1.4426294
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.96
Output dim: 7, lower bound: -1.4777015, upper bound: 1.4583033
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.96
Output dim: 7, lower bound: -1.4426177, upper bound: 1.4747143
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.96
Output dim: 7, lower bound: -1.4426176, upper bound: 1.4776888
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.96
Output dim: 7, lower bound: -1.4583035, upper bound: 1.4620400
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.96
Output dim: 7, lower bound: -1.4583034, upper bound: 1.4777012
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.96
Output dim: 7, lower bound: -1.4554282, upper bound: 1.4685763
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.96
Output dim: 7, lower bound: -1.4554282, upper bound: 1.4841036
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.96
Output dim: 7, lower bound: -1.4709486, upper bound: 1.4685757
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.96
Output dim: 7, lower bound: -1.4709485, upper bound: 1.4841034

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -15.7466335, -11.7147045, -15.7643518, -11.6692095, -2.8587561, 2.8319707
1: -7.1460686, -4.4237890, -7.1338921, -4.4107928, -2.7352757, 2.7101030
2: -8.6623487, -6.1842356, -8.7094278, -6.1778827, -2.4844661, 2.5251923
3: -4.9432139, -2.4651678, -4.9542613, -2.4614668, -2.3908329, 2.3950119
4: -7.9012527, -5.3130641, -7.9404216, -5.2994776, -2.0882292, 2.1200345
5: -6.2374210, -3.8079379, -6.2600279, -3.7610884, -2.3109975, 2.2809882
6: -14.3711338, -11.0556812, -14.3789654, -11.0118885, -2.5655861, 2.5253699
7: 2.3467979, 4.7622356, 2.3029914, 4.7726955, -1.9747074, 2.0064898
8: -1.2097788, 0.9325113, -1.2209191, 0.9363284, -2.0227032, 2.0325544
9: -8.7663946, -5.7962408, -8.7671757, -5.7913976, -2.3672090, 2.3648784

Time for backsubstitution: 14.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6140

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4424990, upper bound: 1.4507550
time: 4.65 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4426179, upper bound: 1.4553153
time: 4.74 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -15.7471714, -11.7135715, -15.7852468, -11.6543455, -2.8820343, 2.8584023
1: -7.1517439, -4.4234881, -7.1632710, -4.3753986, -2.7763453, 2.7397828
2: -8.6628332, -6.1823969, -8.7259712, -6.1657157, -2.4971175, 2.5435743
3: -4.9438291, -2.4636409, -4.9813166, -2.4509797, -2.4026084, 2.4239893
4: -7.9020748, -5.3117595, -7.9655647, -5.2926455, -2.1028366, 2.1434617
5: -6.2378035, -3.8075867, -6.2808781, -3.7553525, -2.3209896, 2.3048491
6: -14.3730202, -11.0555315, -14.3971100, -10.9951420, -2.5812073, 2.5531628
7: 2.3454671, 4.7627039, 2.2931471, 4.7905426, -1.9957530, 2.0160103
8: -1.2102585, 0.9327908, -1.2289400, 0.9480176, -2.0412722, 2.0456166
9: -8.7705765, -5.7961321, -8.7915506, -5.7662430, -2.3968682, 2.3864024

Time for backsubstitution: 14.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6140

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4424991, upper bound: 1.4537125
time: 4.80 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4426178, upper bound: 1.4582911
time: 4.54 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -15.7687864, -11.6553373, -15.7471771, -11.7135725, -2.8392382, 2.8781655
1: -7.1591148, -4.4076223, -7.1517663, -4.4234872, -2.7356277, 2.7441440
2: -8.7206879, -6.1691084, -8.6628361, -6.1823931, -2.5382948, 2.4937277
3: -4.9593015, -2.4537759, -4.9438305, -2.4636350, -2.4011936, 2.3991842
4: -7.9517260, -5.2917271, -7.9020762, -5.3117576, -2.1337228, 2.1009727
5: -6.2641282, -3.7503965, -6.2378035, -3.8075852, -2.2851038, 2.3233628
6: -14.3885374, -11.0028000, -14.3730278, -11.0555315, -2.5375400, 2.5803912
7: 2.2882261, 4.7757874, 2.3454614, 4.7627077, -2.0137768, 1.9787891
8: -1.2247832, 0.9384151, -1.2102590, 0.9327888, -2.0378098, 2.0288491
9: -8.7875395, -5.7901487, -8.7705936, -5.7961316, -2.3865895, 2.3735995

Time for backsubstitution: 14.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4553188, upper bound: 1.4426217
time: 4.71 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4582945, upper bound: 1.4426216
time: 4.29 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -15.7687864, -11.6553373, -15.7687864, -11.6553373, -2.8785591, 2.8785586
1: -7.1591148, -4.4076223, -7.1591148, -4.4076223, -2.7514925, 2.7514925
2: -8.7206879, -6.1691084, -8.7206879, -6.1691084, -2.5515795, 2.5515795
3: -4.9593015, -2.4537759, -4.9593015, -2.4537759, -2.4135532, 2.4135532
4: -7.9517260, -5.2917271, -7.9517260, -5.2917271, -2.1270847, 2.1270847
5: -6.2641282, -3.7503965, -6.2641282, -3.7503965, -2.3250847, 2.3250844
6: -14.3885374, -11.0028000, -14.3885374, -11.0028000, -2.6117363, 2.6117368
7: 2.2882261, 4.7757874, 2.2882261, 4.7757874, -2.0055537, 2.0055535
8: -1.2247832, 0.9384151, -1.2247832, 0.9384151, -2.0425229, 2.0425231
9: -8.7875395, -5.7901487, -8.7875395, -5.7901487, -2.3998256, 2.3998258

Time for backsubstitution: 14.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4553196, upper bound: 1.4426233
time: 4.18 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4582954, upper bound: 1.4426215
time: 4.29 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -15.7471771, -11.7135725, -15.8424368, -11.6793060, -2.8432627, 2.8967214
1: -7.1517663, -4.4234872, -7.1887221, -4.3961544, -2.7556119, 2.7652349
2: -8.6628361, -6.1823931, -8.6829739, -6.1531687, -2.5096674, 2.5005808
3: -4.9438305, -2.4636350, -5.0092278, -2.4303646, -2.4209266, 2.4444847
4: -7.9020762, -5.3117576, -7.9208684, -5.2885032, -2.1099720, 2.1057591
5: -6.2378035, -3.8075852, -6.3123989, -3.7658811, -2.2947760, 2.3549411
6: -14.3730278, -11.0555315, -14.3979731, -11.0174685, -2.5655079, 2.5392575
7: 2.3454614, 4.7627077, 2.3113546, 4.8247833, -2.0213711, 1.9901338
8: -1.2102590, 0.9327888, -1.3185802, 0.9726424, -2.0590816, 2.0925949
9: -8.7705936, -5.7961316, -8.8003502, -5.7219272, -2.4259834, 2.3957977

Time for backsubstitution: 14.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4590704, upper bound: 1.4426173
time: 4.70 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4620282, upper bound: 1.4426173
time: 4.73 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -15.7471771, -11.7135725, -15.8640432, -11.6210861, -2.8835242, 2.9108474
1: -7.1517663, -4.4234872, -7.1961031, -4.3803391, -2.7714272, 2.7726159
2: -8.6628361, -6.1823931, -8.7408390, -6.1397667, -2.5230694, 2.5584459
3: -4.9438305, -2.4636350, -5.0245476, -2.4202385, -2.4326401, 2.4578764
4: -7.9020762, -5.3117576, -7.9703741, -5.2681651, -2.1261845, 2.1523273
5: -6.2378035, -3.8075852, -6.3388648, -3.7086492, -2.3525915, 2.3715329
6: -14.3730278, -11.0555315, -14.4134331, -10.9648571, -2.6018386, 2.5496652
7: 2.3454614, 4.7627077, 2.2540984, 4.8381004, -2.0281646, 2.0273869
8: -1.2102590, 0.9327888, -1.3332634, 0.9782405, -2.0633621, 2.1066940
9: -8.7705936, -5.7961316, -8.8183212, -5.7160392, -2.4322672, 2.4168916

Time for backsubstitution: 14.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4590715, upper bound: 1.4582911
time: 4.60 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4620282, upper bound: 1.4582910
time: 4.70 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -15.7687864, -11.6553373, -15.8424368, -11.6793060, -2.8640451, 2.9219675
1: -7.1591148, -4.4076223, -7.1887221, -4.3961544, -2.7629604, 2.7810998
2: -8.7206879, -6.1691084, -8.6829739, -6.1531687, -2.5675192, 2.5138655
3: -4.9593015, -2.4537759, -5.0092278, -2.4303646, -2.4343987, 2.4554424
4: -7.9517260, -5.2917271, -7.9208684, -5.2885032, -2.1493261, 2.1216218
5: -6.2641282, -3.7503965, -6.3123989, -3.7658811, -2.3178034, 2.3858976
6: -14.3885374, -11.0028000, -14.3979731, -11.0174685, -2.5749044, 2.5887647
7: 2.2882261, 4.7757874, 2.3113546, 4.8247833, -2.0444252, 2.0020540
8: -1.2247832, 0.9384151, -1.3185802, 0.9726424, -2.0728755, 2.0968585
9: -8.7875395, -5.7901487, -8.8003502, -5.7219272, -2.4445791, 2.4027500

Time for backsubstitution: 14.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4747136, upper bound: 1.4426174
time: 4.54 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4776885, upper bound: 1.4426173
time: 4.72 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -15.7687864, -11.6553373, -15.8640432, -11.6210861, -2.9033542, 2.9529502
1: -7.1591148, -4.4076223, -7.1961031, -4.3803391, -2.7787757, 2.7884808
2: -8.7206879, -6.1691084, -8.7408390, -6.1397667, -2.5809212, 2.5717306
3: -4.9593015, -2.4537759, -5.0245476, -2.4202385, -2.4470081, 2.4699163
4: -7.9517260, -5.2917271, -7.9703741, -5.2681651, -2.1522965, 2.1476328
5: -6.2641282, -3.7503965, -6.3388648, -3.7086492, -2.3578148, 2.4136693
6: -14.3885374, -11.0028000, -14.4134331, -10.9648571, -2.6469421, 2.6238627
7: 2.2882261, 4.7757874, 2.2540984, 4.8381004, -2.0586169, 2.0289533
8: -1.2247832, 0.9384151, -1.3332634, 0.9782405, -2.0777702, 2.1115541
9: -8.7875395, -5.7901487, -8.8183212, -5.7160392, -2.4574227, 2.4301276

Time for backsubstitution: 14.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4747145, upper bound: 1.4426174
time: 4.67 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4776893, upper bound: 1.4426175
time: 4.80 seconds

## BFS IS instance: IS_A2_B1_A1_B1

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

Time for backsubstitution: 14.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6140

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4424948, upper bound: 1.4701604
time: 4.64 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4426136, upper bound: 1.4747099
time: 4.56 seconds

## BFS IS instance: IS_A2_B1_A1_B2

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

Time for backsubstitution: 14.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6140

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4424949, upper bound: 1.4731190
time: 4.71 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4426137, upper bound: 1.4776848
time: 4.52 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -15.8640432, -11.6210861, -15.7471771, -11.7135725, -2.9108477, 2.8835239
1: -7.1961031, -4.3803391, -7.1517663, -4.4234872, -2.7726159, 2.7714272
2: -8.7408390, -6.1397667, -8.6628361, -6.1823931, -2.5584459, 2.5230694
3: -5.0245476, -2.4202385, -4.9438305, -2.4636350, -2.4578762, 2.4326398
4: -7.9703741, -5.2681651, -7.9020762, -5.3117576, -2.1523271, 2.1261845
5: -6.3388648, -3.7086492, -6.2378035, -3.8075852, -2.3715327, 2.3525910
6: -14.4134331, -10.9648571, -14.3730278, -11.0555315, -2.5496655, 2.6018384
7: 2.2540984, 4.8381004, 2.3454614, 4.7627077, -2.0273871, 2.0281646
8: -1.3332634, 0.9782405, -1.2102590, 0.9327888, -2.1066942, 2.0633619
9: -8.8183212, -5.7160392, -8.7705936, -5.7961316, -2.4168916, 2.4322670

Time for backsubstitution: 14.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4553146, upper bound: 1.4620298
time: 4.24 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4582904, upper bound: 1.4620298
time: 4.25 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -15.8640432, -11.6210861, -15.7687864, -11.6553373, -2.9529505, 2.9033537
1: -7.1961031, -4.3803391, -7.1591148, -4.4076223, -2.7884808, 2.7787757
2: -8.7408390, -6.1397667, -8.7206879, -6.1691084, -2.5717306, 2.5809212
3: -5.0245476, -2.4202385, -4.9593015, -2.4537759, -2.4699163, 2.4470084
4: -7.9703741, -5.2681651, -7.9517260, -5.2917271, -2.1476326, 2.1522968
5: -6.3388648, -3.7086492, -6.2641282, -3.7503965, -2.4136693, 2.3578153
6: -14.4134331, -10.9648571, -14.3885374, -11.0028000, -2.6238623, 2.6469421
7: 2.2540984, 4.8381004, 2.2882261, 4.7757874, -2.0289531, 2.0586171
8: -1.3332634, 0.9782405, -1.2247832, 0.9384151, -2.1115541, 2.0777702
9: -8.8183212, -5.7160392, -8.7875395, -5.7901487, -2.4301276, 2.4574227

Time for backsubstitution: 14.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4553155, upper bound: 1.4620280
time: 4.57 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4582913, upper bound: 1.4620279
time: 4.32 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -15.8424368, -11.6793060, -15.8424368, -11.6793060, -2.8971725, 2.8971725
1: -7.1887221, -4.3961544, -7.1887221, -4.3961544, -2.7925677, 2.7925677
2: -8.6829739, -6.1531687, -8.6829739, -6.1531687, -2.5298052, 2.5298052
3: -5.0092278, -2.4303646, -5.0092278, -2.4303646, -2.5072742, 2.5072742
4: -7.9208684, -5.2885032, -7.9208684, -5.2885032, -2.1371784, 2.1371784
5: -6.3123989, -3.7658811, -6.3123989, -3.7658811, -2.4175367, 2.4175367
6: -14.3979731, -11.0174685, -14.3979731, -11.0174685, -2.5634298, 2.5634303
7: 2.3113546, 4.8247833, 2.3113546, 4.8247833, -2.0368018, 2.0368018
8: -1.3185802, 0.9726424, -1.3185802, 0.9726424, -2.1493597, 2.1493595
9: -8.8003502, -5.7219272, -8.8003502, -5.7219272, -2.4441023, 2.4441023

Time for backsubstitution: 14.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4525138, upper bound: 1.4685632
time: 4.14 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4554164, upper bound: 1.4685639
time: 4.10 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -15.8424368, -11.6793060, -15.8640432, -11.6210861, -2.9508364, 2.9181504
1: -7.1887221, -4.3961544, -7.1961031, -4.3803391, -2.8083830, 2.7999487
2: -8.6829739, -6.1531687, -8.7408390, -6.1397667, -2.5432072, 2.5876703
3: -5.0092278, -2.4303646, -5.0245476, -2.4202385, -2.5189872, 2.5206208
4: -7.9208684, -5.2885032, -7.9703741, -5.2681651, -2.1533914, 2.1826172
5: -6.3123989, -3.7658811, -6.3388648, -3.7086492, -2.4702165, 2.4406469
6: -14.3979731, -11.0174685, -14.4134331, -10.9648571, -2.6165614, 2.5739505
7: 2.3113546, 4.8247833, 2.2540984, 4.8381004, -2.0490448, 2.0792234
8: -1.3185802, 0.9726424, -1.3332634, 0.9782405, -2.1551099, 2.1643271
9: -8.8003502, -5.7219272, -8.8183212, -5.7160392, -2.4510007, 2.4651976

Time for backsubstitution: 14.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4525138, upper bound: 1.4840909
time: 4.31 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4554164, upper bound: 1.4840919
time: 4.08 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -15.8640432, -11.6210861, -15.8424368, -11.6793060, -2.9181514, 2.9508362
1: -7.1961031, -4.3803391, -7.1887221, -4.3961544, -2.7999487, 2.8083830
2: -8.7408390, -6.1397667, -8.6829739, -6.1531687, -2.5876703, 2.5432072
3: -5.0245476, -2.4202385, -5.0092278, -2.4303646, -2.5206208, 2.5189867
4: -7.9703741, -5.2681651, -7.9208684, -5.2885032, -2.1826169, 2.1533911
5: -6.3388648, -3.7086492, -6.3123989, -3.7658811, -2.4406471, 2.4702165
6: -14.4134331, -10.9648571, -14.3979731, -11.0174685, -2.5739517, 2.6165612
7: 2.2540984, 4.8381004, 2.3113546, 4.8247833, -2.0792232, 2.0490446
8: -1.3332634, 0.9782405, -1.3185802, 0.9726424, -2.1643276, 2.1551099
9: -8.8183212, -5.7160392, -8.8003502, -5.7219272, -2.4651976, 2.4510005

Time for backsubstitution: 14.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4680365, upper bound: 1.4685628
time: 4.64 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4709358, upper bound: 1.4685641
time: 4.46 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -15.8640432, -11.6210861, -15.8640432, -11.6210861, -2.9574585, 2.9574585
1: -7.1961031, -4.3803391, -7.1961031, -4.3803391, -2.8157640, 2.8157640
2: -8.7408390, -6.1397667, -8.7408390, -6.1397667, -2.6010723, 2.6010723
3: -5.0245476, -2.4202385, -5.0245476, -2.4202385, -2.5331831, 2.5331831
4: -7.9703741, -5.2681651, -7.9703741, -5.2681651, -2.1794014, 2.1794016
5: -6.3388648, -3.7086492, -6.3388648, -3.7086492, -2.4806576, 2.4806571
6: -14.4134331, -10.9648571, -14.4134331, -10.9648571, -2.6480322, 2.6480317
7: 2.2540984, 4.8381004, 2.2540984, 4.8381004, -2.0759439, 2.0759435
8: -1.3332634, 0.9782405, -1.3332634, 0.9782405, -2.1690760, 2.1690760
9: -8.8183212, -5.7160392, -8.8183212, -5.7160392, -2.4783802, 2.4783802

Time for backsubstitution: 14.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4680374, upper bound: 1.4685636
time: 4.40 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4709367, upper bound: 1.4685640
time: 4.05 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 22.88 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.88
Output dim: 7, lower bound: -1.4424990, upper bound: 1.4507550
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.88
Output dim: 7, lower bound: -1.4426179, upper bound: 1.4553153
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.88
Output dim: 7, lower bound: -1.4424991, upper bound: 1.4537125
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.88
Output dim: 7, lower bound: -1.4426178, upper bound: 1.4582911
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.88
Output dim: 7, lower bound: -1.4553188, upper bound: 1.4426217
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.88
Output dim: 7, lower bound: -1.4582945, upper bound: 1.4426216
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.88
Output dim: 7, lower bound: -1.4553196, upper bound: 1.4426233
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.88
Output dim: 7, lower bound: -1.4582954, upper bound: 1.4426215
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.88
Output dim: 7, lower bound: -1.4590704, upper bound: 1.4426173
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.88
Output dim: 7, lower bound: -1.4620282, upper bound: 1.4426173
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.88
Output dim: 7, lower bound: -1.4590715, upper bound: 1.4582911
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.88
Output dim: 7, lower bound: -1.4620282, upper bound: 1.4582910
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.88
Output dim: 7, lower bound: -1.4747136, upper bound: 1.4426174
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.88
Output dim: 7, lower bound: -1.4776885, upper bound: 1.4426173
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.88
Output dim: 7, lower bound: -1.4747145, upper bound: 1.4426174
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.88
Output dim: 7, lower bound: -1.4776893, upper bound: 1.4426175
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.88
Output dim: 7, lower bound: -1.4424948, upper bound: 1.4701604
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.88
Output dim: 7, lower bound: -1.4426136, upper bound: 1.4747099
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.88
Output dim: 7, lower bound: -1.4424949, upper bound: 1.4731190
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.88
Output dim: 7, lower bound: -1.4426137, upper bound: 1.4776848
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.88
Output dim: 7, lower bound: -1.4553146, upper bound: 1.4620298
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.88
Output dim: 7, lower bound: -1.4582904, upper bound: 1.4620298
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.88
Output dim: 7, lower bound: -1.4553155, upper bound: 1.4620280
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.88
Output dim: 7, lower bound: -1.4582913, upper bound: 1.4620279
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.88
Output dim: 7, lower bound: -1.4525138, upper bound: 1.4685632
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.88
Output dim: 7, lower bound: -1.4554164, upper bound: 1.4685639
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.88
Output dim: 7, lower bound: -1.4525138, upper bound: 1.4840909
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.88
Output dim: 7, lower bound: -1.4554164, upper bound: 1.4840919
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.88
Output dim: 7, lower bound: -1.4680365, upper bound: 1.4685628
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.88
Output dim: 7, lower bound: -1.4709358, upper bound: 1.4685641
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.88
Output dim: 7, lower bound: -1.4680374, upper bound: 1.4685636
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.88
Output dim: 7, lower bound: -1.4709367, upper bound: 1.4685640

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -15.7450714, -11.7161856, -15.7642279, -11.6693287, -2.8567896, 2.8300204
1: -7.1428504, -4.4374094, -7.1336298, -4.4119120, -2.7309384, 2.6962204
2: -8.6556339, -6.1881523, -8.7088900, -6.1781955, -2.4774384, 2.5207376
3: -4.9288902, -2.4704850, -4.9530916, -2.4618883, -2.3753624, 2.3883336
4: -7.8960490, -5.3305001, -7.9399986, -5.3008981, -2.0820627, 2.1009195
5: -6.2310781, -3.8110492, -6.2595000, -3.7613368, -2.3033824, 2.2767930
6: -14.3577881, -11.0581017, -14.3778763, -11.0120792, -2.5516500, 2.5223970
7: 2.3505759, 4.7551837, 2.3032947, 4.7721124, -1.9711447, 1.9991636
8: -1.2048607, 0.9311509, -1.2205074, 0.9362202, -2.0137153, 2.0261016
9: -8.7525129, -5.7983165, -8.7660484, -5.7915649, -2.3535838, 2.3616824

Time for backsubstitution: 14.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6156

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4424991, upper bound: 1.4350622
time: 4.58 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4424991, upper bound: 1.4507552
time: 4.68 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -15.7582283, -11.7126331, -15.7643547, -11.6692114, -2.8744040, 2.8371594
1: -7.1675329, -4.4140968, -7.1338906, -4.4108028, -2.7567301, 2.7197938
2: -8.6690216, -6.1618323, -8.7094250, -6.1778846, -2.4911370, 2.5475926
3: -4.9534512, -2.4264560, -4.9542484, -2.4614694, -2.3988128, 2.4276354
4: -7.9605684, -5.3086376, -7.9404182, -5.2994924, -2.1390486, 2.1262214
5: -6.2550449, -3.7808037, -6.2600241, -3.7610905, -2.3291068, 2.3058562
6: -14.3773155, -11.0100126, -14.3789616, -11.0118923, -2.5706224, 2.5643806
7: 2.3280244, 4.7658238, 2.3029923, 4.7726917, -1.9965193, 2.0099409
8: -1.2252870, 0.9358397, -1.2209151, 0.9363270, -2.0465174, 2.0319924
9: -8.7780628, -5.7600131, -8.7671661, -5.7913990, -2.3859634, 2.3950605

Time for backsubstitution: 14.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6156

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4426179, upper bound: 1.4396604
time: 4.47 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4426179, upper bound: 1.4553156
time: 4.77 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 23.60 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 23.60
Output dim: 7, lower bound: -1.4424991, upper bound: 1.4350622
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 23.60
Output dim: 7, lower bound: -1.4424991, upper bound: 1.4507552
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 23.60
Output dim: 7, lower bound: -1.4426179, upper bound: 1.4396604
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 23.60
Output dim: 7, lower bound: -1.4426179, upper bound: 1.4553156
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.60
Output dim: 7, lower bound: -1.4424991, upper bound: 1.4537125
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.60
Output dim: 7, lower bound: -1.4426178, upper bound: 1.4582911
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.60
Output dim: 7, lower bound: -1.4553188, upper bound: 1.4426217
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.60
Output dim: 7, lower bound: -1.4582945, upper bound: 1.4426216
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.60
Output dim: 7, lower bound: -1.4553196, upper bound: 1.4426233
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.60
Output dim: 7, lower bound: -1.4582954, upper bound: 1.4426215
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.60
Output dim: 7, lower bound: -1.4590704, upper bound: 1.4426173
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.60
Output dim: 7, lower bound: -1.4620282, upper bound: 1.4426173
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.60
Output dim: 7, lower bound: -1.4590715, upper bound: 1.4582911
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.60
Output dim: 7, lower bound: -1.4620282, upper bound: 1.4582910
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.60
Output dim: 7, lower bound: -1.4747136, upper bound: 1.4426174
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.60
Output dim: 7, lower bound: -1.4776885, upper bound: 1.4426173
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.60
Output dim: 7, lower bound: -1.4747145, upper bound: 1.4426174
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.60
Output dim: 7, lower bound: -1.4776893, upper bound: 1.4426175
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.60
Output dim: 7, lower bound: -1.4424948, upper bound: 1.4701604
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.60
Output dim: 7, lower bound: -1.4426136, upper bound: 1.4747099
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.60
Output dim: 7, lower bound: -1.4424949, upper bound: 1.4731190
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.60
Output dim: 7, lower bound: -1.4426137, upper bound: 1.4776848
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.60
Output dim: 7, lower bound: -1.4553146, upper bound: 1.4620298
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.60
Output dim: 7, lower bound: -1.4582904, upper bound: 1.4620298
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.60
Output dim: 7, lower bound: -1.4553155, upper bound: 1.4620280
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.60
Output dim: 7, lower bound: -1.4582913, upper bound: 1.4620279
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.60
Output dim: 7, lower bound: -1.4525138, upper bound: 1.4685632
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.60
Output dim: 7, lower bound: -1.4554164, upper bound: 1.4685639
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.60
Output dim: 7, lower bound: -1.4525138, upper bound: 1.4840909
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.60
Output dim: 7, lower bound: -1.4554164, upper bound: 1.4840919
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.60
Output dim: 7, lower bound: -1.4680365, upper bound: 1.4685628
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.60
Output dim: 7, lower bound: -1.4709358, upper bound: 1.4685641
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.60
Output dim: 7, lower bound: -1.4680374, upper bound: 1.4685636
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.60
Output dim: 7, lower bound: -1.4709367, upper bound: 1.4685640
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.121267795562744
rel_dist={7: [-1.4841364466020996, 1.4841360647073234]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6192
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6192

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1486618, upper bound: 1.1392275
time: 4.57 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1530073, upper bound: 1.1530054
time: 5.08 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 9.80 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 9.80
Output dim: 7, lower bound: -1.1486618, upper bound: 1.1392275
IS_A2, status: Status.UNKNOWN, split count: 1, time: 9.80
Output dim: 7, lower bound: -1.1530073, upper bound: 1.1530054

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -15.7687855, -11.6553326, -15.8229198, -11.6252165, -2.5087538, 2.5419106
1: -7.1591153, -4.4076223, -7.1823936, -4.3909721, -2.7428141, 2.7567620
2: -8.7206879, -6.1691089, -8.7359295, -6.1509843, -2.4203587, 2.4103084
3: -4.9593029, -2.4537749, -4.9978518, -2.4275467, -2.2625661, 2.2660737
4: -7.9517303, -5.2917261, -7.9657664, -5.2775507, -1.9103403, 1.9150310
5: -6.2641287, -3.7503946, -6.3065300, -3.7179561, -2.1778007, 2.2038507
6: -14.3885374, -11.0027924, -14.4064779, -10.9802923, -2.2771215, 2.2637842
7: 2.2882204, 4.7757864, 2.2615671, 4.8117642, -1.8453283, 1.8277826
8: -1.2247856, 0.9384165, -1.2872524, 0.9737010, -1.9006138, 1.9163163
9: -8.7875423, -5.7901473, -8.8147469, -5.7478409, -2.1245203, 2.1092036

Time for backsubstitution: 14.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6192
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6192

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1392273, upper bound: 1.1392269
time: 4.83 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1392273, upper bound: 1.1392270
time: 4.99 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -15.8640451, -11.6210775, -15.8640537, -11.6210804, -2.5575743, 2.6073208
1: -7.1961031, -4.3803387, -7.1961083, -4.3803353, -2.8105755, 2.7928877
2: -8.7408390, -6.1397657, -8.7408409, -6.1397614, -2.4694004, 2.4588304
3: -5.0245490, -2.4202378, -5.0245576, -2.4202356, -2.3351068, 2.3505244
4: -7.9703798, -5.2681632, -7.9703808, -5.2681613, -1.9494987, 1.9444687
5: -6.3388667, -3.7086473, -6.3388705, -3.7086444, -2.2795749, 2.3002243
6: -14.4134312, -10.9648504, -14.4134331, -10.9648457, -2.3056951, 2.2872505
7: 2.2540932, 4.8381004, 2.2540908, 4.8381071, -1.8950527, 1.8760798
8: -1.3332644, 0.9782419, -1.3332782, 0.9782429, -1.9913650, 2.0092332
9: -8.8183231, -5.7160378, -8.8183250, -5.7160306, -2.1864891, 2.1559525

Time for backsubstitution: 14.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6192
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6192

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1392273, upper bound: 1.1486613
time: 4.69 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1392273, upper bound: 1.1530064
time: 4.96 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 23.92 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 23.92
Output dim: 7, lower bound: -1.1392273, upper bound: 1.1392269
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 23.92
Output dim: 7, lower bound: -1.1392273, upper bound: 1.1392270
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 23.92
Output dim: 7, lower bound: -1.1392273, upper bound: 1.1486613
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 23.92
Output dim: 7, lower bound: -1.1392273, upper bound: 1.1530064

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -15.7687855, -11.6553326, -15.7687855, -11.6553326, -2.4869881, 2.4869876
1: -7.1591153, -4.4076223, -7.1591153, -4.4076223, -2.7264748, 2.7264748
2: -8.7206879, -6.1691089, -8.7206879, -6.1691089, -2.3946085, 2.3946085
3: -4.9593029, -2.4537749, -4.9593029, -2.4537749, -2.2328176, 2.2328179
4: -7.9517303, -5.2917261, -7.9517303, -5.2917261, -1.8981104, 1.8981104
5: -6.2641287, -3.7503946, -6.2641287, -3.7503946, -2.1484170, 2.1484170
6: -14.3885374, -11.0027924, -14.3885374, -11.0027924, -2.2542481, 2.2542481
7: 2.2882204, 4.7757864, 2.2882204, 4.7757864, -1.8091254, 1.8091257
8: -1.2247856, 0.9384165, -1.2247856, 0.9384165, -1.8686838, 1.8686838
9: -8.7875423, -5.7901473, -8.7875423, -5.7901473, -2.0818448, 2.0818446

Time for backsubstitution: 14.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6156

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1301869, upper bound: 1.1392155
time: 4.67 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1392185, upper bound: 1.1392158
time: 6.46 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -15.7687855, -11.6553326, -15.8640146, -11.6222239, -2.5104418, 2.5464900
1: -7.1591153, -4.4076223, -7.1941910, -4.3804197, -2.7539845, 2.7620797
2: -8.7206879, -6.1691089, -8.7406235, -6.1398859, -2.4359107, 2.4149609
3: -4.9593029, -2.4537749, -5.0244637, -2.4211154, -2.2653813, 2.2792742
4: -7.9517303, -5.2917261, -7.9695520, -5.2681890, -1.9231033, 1.9176078
5: -6.2641287, -3.7503946, -6.3387156, -3.7106481, -2.1805363, 2.2192595
6: -14.3885374, -11.0027924, -14.4132109, -10.9649429, -2.2848151, 2.2671947
7: 2.2882204, 4.7757864, 2.2551751, 4.8380909, -1.8498659, 1.8315716
8: -1.2247856, 0.9384165, -1.3332386, 0.9775133, -1.9000711, 1.9203789
9: -8.7875423, -5.7901473, -8.8181705, -5.7161155, -2.1279421, 2.1117885

Time for backsubstitution: 14.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6156

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1301869, upper bound: 1.1392161
time: 7.86 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1392186, upper bound: 1.1392156
time: 4.79 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -15.8640146, -11.6222239, -15.7687855, -11.6553326, -2.5464897, 2.5104413
1: -7.1941910, -4.3804197, -7.1591153, -4.4076223, -2.7620792, 2.7539845
2: -8.7406235, -6.1398859, -8.7206879, -6.1691089, -2.4149618, 2.4359112
3: -5.0244637, -2.4211154, -4.9593029, -2.4537749, -2.2792740, 2.2653821
4: -7.9695520, -5.2681890, -7.9517303, -5.2917261, -1.9176078, 1.9231036
5: -6.3387156, -3.7106481, -6.2641287, -3.7503946, -2.2192593, 2.1805363
6: -14.4132109, -10.9649429, -14.3885374, -11.0027924, -2.2671947, 2.2848148
7: 2.2551751, 4.8380909, 2.2882204, 4.7757864, -1.8315716, 1.8498662
8: -1.3332386, 0.9775133, -1.2247856, 0.9384165, -1.9203792, 1.9000709
9: -8.8181705, -5.7161155, -8.7875423, -5.7901473, -2.1117887, 2.1279426

Time for backsubstitution: 14.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6156

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1301838, upper bound: 1.1486487
time: 4.79 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1392154, upper bound: 1.1486488
time: 4.98 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -15.8640451, -11.6210775, -15.8640451, -11.6210775, -2.5575743, 2.5575743
1: -7.1961031, -4.3803387, -7.1961031, -4.3803387, -2.8105736, 2.8105736
2: -8.7408390, -6.1397657, -8.7408390, -6.1397657, -2.4693966, 2.4693966
3: -5.0245490, -2.4202378, -5.0245490, -2.4202378, -2.3505173, 2.3505175
4: -7.9703798, -5.2681632, -7.9703798, -5.2681632, -1.9494958, 1.9494956
5: -6.3388667, -3.7086473, -6.3388667, -3.7086473, -2.3002214, 2.3002212
6: -14.4134312, -10.9648504, -14.4134312, -10.9648504, -2.2872481, 2.2872481
7: 2.2540932, 4.8381004, 2.2540932, 4.8381004, -1.8760781, 1.8760784
8: -1.3332644, 0.9782419, -1.3332644, 0.9782419, -1.9913650, 1.9913650
9: -8.8183231, -5.7160378, -8.8183231, -5.7160378, -2.1559515, 2.1559517

Time for backsubstitution: 14.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6156

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1301838, upper bound: 1.1486496
time: 8.19 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1392153, upper bound: 1.1529951
time: 4.82 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 27.38 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 27.38
Output dim: 7, lower bound: -1.1301869, upper bound: 1.1392155
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 27.38
Output dim: 7, lower bound: -1.1392185, upper bound: 1.1392158
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 27.38
Output dim: 7, lower bound: -1.1301869, upper bound: 1.1392161
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 27.38
Output dim: 7, lower bound: -1.1392186, upper bound: 1.1392156
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 27.38
Output dim: 7, lower bound: -1.1301838, upper bound: 1.1486487
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 27.38
Output dim: 7, lower bound: -1.1392154, upper bound: 1.1486488
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 27.38
Output dim: 7, lower bound: -1.1301838, upper bound: 1.1486496
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 27.38
Output dim: 7, lower bound: -1.1392153, upper bound: 1.1529951

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -15.7471771, -11.7135725, -15.7649879, -11.6719265, -2.4485269, 2.4213114
1: -7.1517663, -4.4234872, -7.1570511, -4.4110212, -2.7139235, 2.7073689
2: -8.6628361, -6.1823931, -8.7042704, -6.1709166, -2.3351135, 2.3647032
3: -4.9438305, -2.4636350, -4.9551430, -2.4559221, -2.2165513, 2.2177668
4: -7.9020762, -5.3117576, -7.9379983, -5.2956653, -1.8467159, 1.8687909
5: -6.2378035, -3.8075852, -6.2597332, -3.7668066, -2.1077719, 2.0835669
6: -14.3730278, -11.0555315, -14.3856068, -11.0178375, -2.2228913, 2.1994035
7: 2.3454614, 4.7627077, 2.3045321, 4.7739925, -1.7516460, 1.7738788
8: -1.2102590, 0.9327888, -1.2215726, 0.9369025, -1.8518400, 1.8596039
9: -8.7705936, -5.7961316, -8.7829628, -5.7915521, -2.0594664, 2.0695200

Time for backsubstitution: 14.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1301785, upper bound: 1.1370011
time: 5.05 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1301785, upper bound: 1.1392124
time: 4.55 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -15.7687864, -11.6553373, -15.7687855, -11.6553326, -2.4869862, 2.4602804
1: -7.1591148, -4.4076223, -7.1591153, -4.4076223, -2.7228489, 2.7264743
2: -8.7206879, -6.1691084, -8.7206879, -6.1691089, -2.3593054, 2.3946080
3: -4.9593015, -2.4537759, -4.9593029, -2.4537749, -2.2331190, 2.2328169
4: -7.9517260, -5.2917271, -7.9517303, -5.2917261, -1.8716421, 1.8981097
5: -6.2641282, -3.7503965, -6.2641287, -3.7503946, -2.1484160, 2.1233587
6: -14.3885374, -11.0028000, -14.3885374, -11.0027924, -2.2716980, 2.2542453
7: 2.2882261, 4.7757874, 2.2882204, 4.7757864, -1.7748015, 1.8091257
8: -1.2247832, 0.9384151, -1.2247856, 0.9384165, -1.8686829, 1.8670759
9: -8.7875395, -5.7901487, -8.7875423, -5.7901473, -2.0870814, 2.0818441

Time for backsubstitution: 14.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6156

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1392187, upper bound: 1.1301858
time: 5.66 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1392187, upper bound: 1.1392177
time: 5.47 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -15.7471771, -11.7135725, -15.8602085, -11.6388149, -2.4535401, 2.4811625
1: -7.1517663, -4.4234872, -7.1920528, -4.3838053, -2.7414513, 2.7428856
2: -8.6628361, -6.1823931, -8.7242002, -6.1417923, -2.3763218, 2.3850036
3: -4.9438305, -2.4636350, -5.0203242, -2.4234674, -2.2489357, 2.2642481
4: -7.9020762, -5.3117576, -7.9558640, -5.2724891, -1.8713408, 1.8874059
5: -6.2378035, -3.8075852, -6.3342819, -3.7270992, -2.1338420, 2.1552672
6: -14.3730278, -11.0555315, -14.4102955, -10.9799690, -2.2440920, 2.2123559
7: 2.3454614, 4.7627077, 2.2714963, 4.8360682, -1.7922163, 1.7862930
8: -1.2102590, 0.9327888, -1.3298168, 0.9760060, -1.8826883, 1.9111323
9: -8.7705936, -5.7961316, -8.8126945, -5.7174902, -2.1062574, 2.0983844

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1396137, upper bound: 1.1369992
time: 4.67 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1396137, upper bound: 1.1392093
time: 4.60 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -15.7687864, -11.6553373, -15.8640146, -11.6222239, -2.5104408, 2.5197661
1: -7.1591148, -4.4076223, -7.1941910, -4.3804197, -2.7503576, 2.7620792
2: -8.7206879, -6.1691084, -8.7406235, -6.1398859, -2.4006090, 2.4149604
3: -4.9593015, -2.4537759, -5.0244637, -2.4211154, -2.2656837, 2.2788167
4: -7.9517260, -5.2917271, -7.9695520, -5.2681890, -1.8966355, 1.9176068
5: -6.2641282, -3.7503965, -6.3387156, -3.7106481, -2.1805353, 2.1941860
6: -14.3885374, -11.0028000, -14.4132109, -10.9649429, -2.2980254, 2.2671905
7: 2.2882261, 4.7757874, 2.2551751, 4.8380909, -1.8155322, 1.8315713
8: -1.2247832, 0.9384151, -1.3332386, 0.9775133, -1.8990755, 1.9185812
9: -8.7875395, -5.7901487, -8.8181705, -5.7161155, -2.1319952, 2.1117880

Time for backsubstitution: 14.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6156

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1486492, upper bound: 1.1301840
time: 4.86 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1486491, upper bound: 1.1392167
time: 4.88 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -15.8424063, -11.6804485, -15.7649879, -11.6719265, -2.4800529, 2.4447269
1: -7.1868114, -4.3962359, -7.1570511, -4.4110212, -2.7495499, 2.7349448
2: -8.6827602, -6.1532893, -8.7042704, -6.1709166, -2.3553543, 2.4007497
3: -5.0091414, -2.4312415, -4.9551430, -2.4559221, -2.2629285, 2.2500834
4: -7.9200411, -5.2885308, -7.9379983, -5.2956653, -1.8663130, 1.8823411
5: -6.3122454, -3.7678833, -6.2597332, -3.7668066, -2.1565933, 2.1156566
6: -14.3977518, -11.0175600, -14.3856068, -11.0178375, -2.2358251, 2.2306535
7: 2.3124371, 4.8247752, 2.3045321, 4.7739925, -1.7739582, 1.7961950
8: -1.3185530, 0.9719143, -1.2215726, 0.9369025, -1.9026361, 1.8910694
9: -8.8001900, -5.7220025, -8.7829628, -5.7915521, -2.0882568, 2.1157546

Time for backsubstitution: 14.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1301753, upper bound: 1.1461348
time: 5.18 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1301753, upper bound: 1.1486401
time: 5.02 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -15.8640137, -11.6222305, -15.7687855, -11.6553326, -2.5403180, 2.4837337
1: -7.1941919, -4.3804216, -7.1591153, -4.4076223, -2.7585306, 2.7539835
2: -8.7406225, -6.1398869, -8.7206879, -6.1691089, -2.3796663, 2.4359107
3: -5.0244613, -2.4211173, -4.9593029, -2.4537749, -2.2792413, 2.2653806
4: -7.9695482, -5.2681885, -7.9517303, -5.2917261, -1.8911390, 1.9231021
5: -6.3387146, -3.7106519, -6.2641287, -3.7503946, -2.2137403, 2.1554794
6: -14.4132109, -10.9649487, -14.3885374, -11.0027924, -2.2846446, 2.2848127
7: 2.2551813, 4.8380909, 2.2882204, 4.7757864, -1.7972469, 1.8443937
8: -1.3332367, 0.9775128, -1.2247856, 0.9384165, -1.9193864, 1.8983281
9: -8.8181667, -5.7161150, -8.7875423, -5.7901473, -2.1170230, 2.1279426

Time for backsubstitution: 14.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6156

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1392156, upper bound: 1.1396219
time: 6.51 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1392155, upper bound: 1.1486485
time: 7.71 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -15.8424368, -11.6793060, -15.8602419, -11.6376762, -2.5141196, 2.4918637
1: -7.1887221, -4.3961544, -7.1939650, -4.3837218, -2.7979145, 2.7913914
2: -8.6829739, -6.1531687, -8.7244167, -6.1416712, -2.4096909, 2.4392943
3: -5.0092278, -2.4303646, -5.0204091, -2.4225903, -2.3342423, 2.3352742
4: -7.9208684, -5.2885032, -7.9566922, -5.2724648, -1.8978348, 1.9179308
5: -6.3123989, -3.7658811, -6.3344331, -3.7250955, -2.2489042, 2.2353027
6: -14.3979731, -11.0174685, -14.4105186, -10.9798756, -2.2557926, 2.2324929
7: 2.3113546, 4.8247833, 2.2704144, 4.8360767, -1.8182268, 1.8358049
8: -1.3185802, 0.9726424, -1.3298426, 0.9767342, -1.9742789, 1.9821181
9: -8.8003502, -5.7219272, -8.8128538, -5.7174168, -2.1325569, 2.1426022

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1365523, upper bound: 1.1506194
time: 5.12 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1365523, upper bound: 1.1529852
time: 5.12 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -15.8640432, -11.6210861, -15.8640451, -11.6210775, -2.5575733, 2.5308661
1: -7.1961031, -4.3803391, -7.1961031, -4.3803387, -2.8070230, 2.8105717
2: -8.7408390, -6.1397667, -8.7408390, -6.1397657, -2.4341002, 2.4693952
3: -5.0245476, -2.4202385, -5.0245490, -2.4202378, -2.3508186, 2.3505163
4: -7.9703741, -5.2681651, -7.9703798, -5.2681632, -1.9230275, 1.9494951
5: -6.3388648, -3.7086492, -6.3388667, -3.7086473, -2.3002210, 2.2751632
6: -14.4134331, -10.9648571, -14.4134312, -10.9648504, -2.3046970, 2.2872443
7: 2.2540984, 4.8381004, 2.2540932, 4.8381004, -1.8417530, 1.8760779
8: -1.3332634, 0.9782405, -1.3332644, 0.9782419, -1.9913640, 1.9898169
9: -8.8183212, -5.7160392, -8.8183231, -5.7160378, -2.1611876, 2.1559520

Time for backsubstitution: 14.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6156

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1455360, upper bound: 1.1440048
time: 5.42 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1455361, upper bound: 1.1529940
time: 5.20 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 25.00 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.00
Output dim: 7, lower bound: -1.1301785, upper bound: 1.1370011
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.00
Output dim: 7, lower bound: -1.1301785, upper bound: 1.1392124
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.00
Output dim: 7, lower bound: -1.1392187, upper bound: 1.1301858
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.00
Output dim: 7, lower bound: -1.1392187, upper bound: 1.1392177
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.00
Output dim: 7, lower bound: -1.1396137, upper bound: 1.1369992
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.00
Output dim: 7, lower bound: -1.1396137, upper bound: 1.1392093
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.00
Output dim: 7, lower bound: -1.1486492, upper bound: 1.1301840
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.00
Output dim: 7, lower bound: -1.1486491, upper bound: 1.1392167
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.00
Output dim: 7, lower bound: -1.1301753, upper bound: 1.1461348
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.00
Output dim: 7, lower bound: -1.1301753, upper bound: 1.1486401
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.00
Output dim: 7, lower bound: -1.1392156, upper bound: 1.1396219
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.00
Output dim: 7, lower bound: -1.1392155, upper bound: 1.1486485
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.00
Output dim: 7, lower bound: -1.1365523, upper bound: 1.1506194
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.00
Output dim: 7, lower bound: -1.1365523, upper bound: 1.1529852
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.00
Output dim: 7, lower bound: -1.1455360, upper bound: 1.1440048
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.00
Output dim: 7, lower bound: -1.1455361, upper bound: 1.1529940

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -15.7460518, -11.7158823, -15.7626648, -11.6765051, -2.4360061, 2.4132648
1: -7.1400557, -4.4241104, -7.1330023, -4.4123082, -2.7007017, 2.6825266
2: -8.6618347, -6.1861768, -8.7022095, -6.1786652, -2.3236585, 2.3575439
3: -4.9425535, -2.4667821, -4.9524169, -2.4623861, -2.2090802, 2.2119510
4: -7.9003553, -5.3144460, -7.9343815, -5.3011599, -1.8346686, 1.8600550
5: -6.2370043, -3.8083100, -6.2580624, -3.7683074, -2.1048646, 2.0808985
6: -14.3691378, -11.0558414, -14.3776608, -11.0184975, -2.2158704, 2.1864147
7: 2.3482094, 4.7617264, 2.3101697, 4.7719326, -1.7470145, 1.7672243
8: -1.2092667, 0.9322085, -1.2195308, 0.9356651, -1.8462906, 1.8551123
9: -8.7619686, -5.7963543, -8.7652988, -5.7920151, -2.0497856, 2.0504098

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6140

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1295595, upper bound: 1.1335484
time: 4.76 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1301746, upper bound: 1.1369995
time: 4.64 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -15.7471733, -11.7135754, -15.7835636, -11.6616459, -2.4525065, 2.4424958
1: -7.1517253, -4.4234896, -7.1623793, -4.3769064, -2.7467775, 2.7079883
2: -8.6628323, -6.1824007, -8.7187548, -6.1664991, -2.3397894, 2.3774996
3: -4.9438276, -2.4636452, -4.9794765, -2.4519053, -2.2210059, 2.2424924
4: -7.9020758, -5.3117609, -7.9595032, -5.2943282, -1.8474855, 1.8803666
5: -6.2378025, -3.8075874, -6.2789149, -3.7625706, -2.1152062, 2.1052859
6: -14.3730116, -11.0555315, -14.3957987, -11.0017519, -2.2323613, 2.2117572
7: 2.3454714, 4.7627048, 2.3003240, 4.7897825, -1.7695248, 1.7767968
8: -1.2102571, 0.9327879, -1.2275558, 0.9473562, -1.8647633, 1.8692491
9: -8.7705650, -5.7961335, -8.7896767, -5.7668581, -2.0829382, 2.0676754

Time for backsubstitution: 14.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6140

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1295595, upper bound: 1.1358167
time: 4.89 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1301746, upper bound: 1.1392053
time: 4.76 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -15.7687864, -11.6553373, -15.7471771, -11.7135725, -2.4249496, 2.4518428
1: -7.1591148, -4.4076223, -7.1517663, -4.4234872, -2.7103271, 2.7174449
2: -8.7206879, -6.1691084, -8.6628361, -6.1823931, -2.3743792, 2.3370090
3: -4.9593015, -2.4537759, -4.9438305, -2.4636350, -2.2212591, 2.2192497
4: -7.9517260, -5.2917271, -7.9020762, -5.3117576, -1.8732572, 1.8494930
5: -6.2641282, -3.7503965, -6.2378035, -3.8075852, -2.0871267, 2.1180193
6: -14.3885374, -11.0028000, -14.3730278, -11.0555315, -2.1995215, 2.2363551
7: 2.2882261, 4.7757874, 2.3454614, 4.7627077, -1.7758060, 1.7531753
8: -1.2247832, 0.9384151, -1.2102590, 0.9327888, -1.8629069, 1.8539462
9: -8.7875395, -5.7901487, -8.7705936, -5.7961316, -2.0744457, 2.0614557

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1370005, upper bound: 1.1301781
time: 5.01 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1392092, upper bound: 1.1301806
time: 4.82 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -15.7687864, -11.6553373, -15.7687864, -11.6553373, -2.4602795, 2.4602795
1: -7.1591148, -4.4076223, -7.1591148, -4.4076223, -2.7228489, 2.7228489
2: -8.7206879, -6.1691084, -8.7206879, -6.1691084, -2.3593049, 2.3593044
3: -4.9593015, -2.4537759, -4.9593015, -2.4537759, -2.2331181, 2.2331181
4: -7.9517260, -5.2917271, -7.9517260, -5.2917271, -1.8716412, 1.8716412
5: -6.2641282, -3.7503965, -6.2641282, -3.7503965, -2.1233578, 2.1233578
6: -14.3885374, -11.0028000, -14.3885374, -11.0028000, -2.2716947, 2.2716944
7: 2.2882261, 4.7757874, 2.2882261, 4.7757874, -1.7748008, 1.7748005
8: -1.2247832, 0.9384151, -1.2247832, 0.9384151, -1.8670745, 1.8670747
9: -8.7875395, -5.7901487, -8.7875395, -5.7901487, -2.0870805, 2.0870807

Time for backsubstitution: 14.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1370013, upper bound: 1.1301781
time: 4.84 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1392102, upper bound: 1.1301796
time: 4.81 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -15.7460518, -11.7158823, -15.8578892, -11.6434364, -2.4409134, 2.4718428
1: -7.1400557, -4.4241104, -7.1681323, -4.3850827, -2.7282314, 2.7181535
2: -8.6618347, -6.1861768, -8.7220058, -6.1494808, -2.3648272, 2.3777070
3: -4.9425535, -2.4667821, -5.0175657, -2.4300044, -2.2414131, 2.2578704
4: -7.9003553, -5.3144460, -7.9521847, -5.2779846, -1.8592854, 1.8771274
5: -6.2370043, -3.8083100, -6.3326011, -3.7287872, -2.1307580, 2.1525586
6: -14.3691378, -11.0558414, -14.4023895, -10.9806414, -2.2355332, 2.1992168
7: 2.3482094, 4.7617264, 2.2771349, 4.8339992, -1.7869873, 1.7796540
8: -1.2092667, 0.9322085, -1.3277779, 0.9746656, -1.8769870, 1.9061825
9: -8.7619686, -5.7963543, -8.7950392, -5.7179699, -2.0940804, 2.0791793

Time for backsubstitution: 14.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6140

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1389777, upper bound: 1.1335469
time: 4.75 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1396097, upper bound: 1.1369939
time: 4.87 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -15.7471733, -11.7135754, -15.8788204, -11.6286621, -2.4561718, 2.4887288
1: -7.1517253, -4.4234896, -7.1975474, -4.3496504, -2.7646747, 2.7435079
2: -8.6628323, -6.1824007, -8.7386284, -6.1372938, -2.3812494, 2.3946846
3: -4.9438276, -2.4636452, -5.0450473, -2.4196005, -2.2533188, 2.2762654
4: -7.9020758, -5.3117609, -7.9773350, -5.2711735, -1.8720903, 1.8959928
5: -6.2378025, -3.8075874, -6.3534317, -3.7230489, -2.1389549, 2.1682532
6: -14.3730116, -11.0555315, -14.4201431, -10.9638443, -2.2472863, 2.2246242
7: 2.3454714, 4.7627048, 2.2675347, 4.8519897, -1.7992887, 1.7891805
8: -1.2102571, 0.9327879, -1.3359671, 0.9864631, -1.8885970, 1.9164219
9: -8.7705650, -5.7961335, -8.8193665, -5.6927276, -2.1082306, 2.0967269

Time for backsubstitution: 14.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6140

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1389776, upper bound: 1.1358157
time: 4.47 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1396097, upper bound: 1.1392047
time: 4.65 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -15.7687864, -11.6553373, -15.8424063, -11.6804485, -2.4484153, 2.4820986
1: -7.1591148, -4.4076223, -7.1868114, -4.3962359, -2.7379026, 2.7530708
2: -8.7206879, -6.1691084, -8.6827602, -6.1532893, -2.4023108, 2.3572507
3: -4.9593015, -2.4537759, -5.0091414, -2.4312415, -2.2535748, 2.2651529
4: -7.9517260, -5.2917271, -7.9200411, -5.2885308, -1.8838124, 1.8690906
5: -6.2641282, -3.7503965, -6.3122454, -3.7678833, -2.1192164, 2.1613750
6: -14.3885374, -11.0028000, -14.3977518, -11.0175600, -2.2279744, 2.2438586
7: 2.2882261, 4.7757874, 2.3124371, 4.8247752, -1.7981222, 1.7754869
8: -1.2247832, 0.9384151, -1.3185530, 0.9719143, -1.8933859, 1.9035234
9: -8.7875395, -5.7901487, -8.8001900, -5.7220025, -2.1197557, 2.0902467

Time for backsubstitution: 14.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1461343, upper bound: 1.1301774
time: 4.43 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1486395, upper bound: 1.1301744
time: 5.17 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -15.7687864, -11.6553373, -15.8640137, -11.6222305, -2.4837332, 2.5165102
1: -7.1591148, -4.4076223, -7.1941919, -4.3804216, -2.7503567, 2.7585292
2: -8.7206879, -6.1691084, -8.7406225, -6.1398869, -2.4006081, 2.3796649
3: -4.9593015, -2.4537759, -5.0244613, -2.4211173, -2.2656827, 2.2791500
4: -7.9517260, -5.2917271, -7.9695482, -5.2681885, -1.8966336, 1.8911383
5: -6.2641282, -3.7503965, -6.3387146, -3.7106519, -2.1554785, 2.1914027
6: -14.3885374, -11.0028000, -14.4132109, -10.9649487, -2.2980237, 2.2846413
7: 2.2882261, 4.7757874, 2.2551813, 4.8380909, -1.8138068, 1.7972469
8: -1.2247832, 0.9384151, -1.3332367, 0.9775128, -1.8977318, 1.9179876
9: -8.7875395, -5.7901487, -8.8181667, -5.7161150, -2.1319947, 2.1170228

Time for backsubstitution: 14.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1461351, upper bound: 1.1301762
time: 4.90 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1486403, upper bound: 1.1301747
time: 5.46 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -15.8412800, -11.6828175, -15.7626648, -11.6765051, -2.4675031, 2.4366894
1: -7.1751065, -4.3968582, -7.1330023, -4.4123082, -2.7363234, 2.7101016
2: -8.6816845, -6.1570454, -8.7022095, -6.1786652, -2.3438201, 2.3924084
3: -5.0078316, -2.4344540, -4.9524169, -2.4623861, -2.2553639, 2.2442064
4: -7.9182653, -5.2912197, -7.9343815, -5.3011599, -1.8542266, 1.8720994
5: -6.3114395, -3.7687593, -6.2580624, -3.7683074, -2.1536875, 2.1128931
6: -14.3938789, -11.0178814, -14.3776608, -11.0184975, -2.2280662, 2.2176225
7: 2.3152161, 4.8237934, 2.3101697, 4.7719326, -1.7693155, 1.7895403
8: -1.3175602, 0.9712615, -1.2195308, 0.9356651, -1.8969898, 1.8861032
9: -8.7915688, -5.7222333, -8.7652988, -5.7920151, -2.0785198, 2.0965939

Time for backsubstitution: 14.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6140

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1295564, upper bound: 1.1426857
time: 6.48 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1301715, upper bound: 1.1461295
time: 5.18 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -15.8424072, -11.6804285, -15.7835636, -11.6616459, -2.4827638, 2.4624124
1: -7.1868143, -4.3962364, -7.1623793, -4.3769064, -2.7816448, 2.7355661
2: -8.6827602, -6.1532936, -8.7187548, -6.1664991, -2.3600397, 2.4054332
3: -5.0091395, -2.4312334, -4.9794765, -2.4519053, -2.2674098, 2.2695985
4: -7.9200583, -5.2885332, -7.9595032, -5.2943282, -1.8670936, 1.8909254
5: -6.3122473, -3.7678404, -6.2789149, -3.7625706, -2.1619296, 2.1373887
6: -14.3977442, -11.0175562, -14.3957987, -11.0017519, -2.2398713, 2.2363188
7: 2.3124218, 4.8247719, 2.3003240, 4.7897825, -1.7893994, 1.7991133
8: -1.3185511, 0.9719310, -1.2275558, 0.9473562, -1.9086113, 1.8963172
9: -8.8001671, -5.7220011, -8.7896767, -5.7668581, -2.0951295, 2.1120386

Time for backsubstitution: 14.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6140

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1295564, upper bound: 1.1451760
time: 5.29 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1301715, upper bound: 1.1486350
time: 5.23 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -15.8640137, -11.6222305, -15.7471771, -11.7135725, -2.4784203, 2.4555888
1: -7.1941919, -4.3804216, -7.1517663, -4.4234872, -2.7459316, 2.7449541
2: -8.7406225, -6.1398869, -8.6628361, -6.1823931, -2.3917465, 2.3783131
3: -5.0244613, -2.4211173, -4.9438305, -2.4636350, -2.2675738, 2.2518146
4: -7.9695482, -5.2681885, -7.9020762, -5.3117576, -1.8889160, 1.8744855
5: -6.3387146, -3.7106519, -6.2378035, -3.8075852, -2.1530311, 2.1385922
6: -14.4132109, -10.9649487, -14.3730278, -11.0555315, -2.2124677, 2.2513196
7: 2.2551813, 4.8380909, 2.3454614, 4.7627077, -1.7883010, 1.7885096
8: -1.3332367, 0.9775128, -1.2102590, 0.9327888, -1.9136758, 1.8835864
9: -8.8181667, -5.7161150, -8.7705936, -5.7961316, -2.1043882, 2.1073108

Time for backsubstitution: 14.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1369974, upper bound: 1.1396140
time: 7.06 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1392061, upper bound: 1.1396138
time: 6.71 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -15.8640137, -11.6222305, -15.7687864, -11.6553373, -2.5165100, 2.4837332
1: -7.1941919, -4.3804216, -7.1591148, -4.4076223, -2.7585297, 2.7503572
2: -8.7406225, -6.1398869, -8.7206879, -6.1691084, -2.3796659, 2.4006076
3: -5.0244613, -2.4211173, -4.9593015, -2.4537759, -2.2791500, 2.2656825
4: -7.9695482, -5.2681885, -7.9517260, -5.2917271, -1.8911386, 1.8966334
5: -6.3387146, -3.7106519, -6.2641282, -3.7503965, -2.1914027, 2.1554787
6: -14.4132109, -10.9649487, -14.3885374, -11.0028000, -2.2846413, 2.2980237
7: 2.2551813, 4.8380909, 2.2882261, 4.7757874, -1.7972465, 1.8138068
8: -1.3332367, 0.9775128, -1.2247832, 0.9384151, -1.9179873, 1.8977315
9: -8.8181667, -5.7161150, -8.7875395, -5.7901487, -2.1170225, 2.1319950

Time for backsubstitution: 14.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1369983, upper bound: 1.1396157
time: 5.95 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1392069, upper bound: 1.1396156
time: 5.91 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -15.8413153, -11.6816235, -15.8579197, -11.6422653, -2.5015268, 2.4838409
1: -7.1770997, -4.3967733, -7.1700931, -4.3849978, -2.7846956, 2.7665467
2: -8.6819096, -6.1569228, -8.7222271, -6.1493578, -2.3981409, 2.4319892
3: -5.0079212, -2.4335403, -5.0176511, -2.4291034, -2.3266854, 2.3294046
4: -7.9191246, -5.2911916, -7.9530306, -5.2779598, -1.8857751, 1.9076703
5: -6.3115950, -3.7666764, -6.3327518, -3.7267399, -2.2458794, 2.2325704
6: -14.3941088, -11.0177832, -14.4026117, -10.9805470, -2.2487278, 2.2194357
7: 2.3140860, 4.8238010, 2.2760253, 4.8340101, -1.8135900, 1.8291943
8: -1.3175893, 0.9720154, -1.3278050, 0.9754043, -1.9686146, 1.9775941
9: -8.7917309, -5.7221575, -8.7951984, -5.7178946, -2.1228251, 2.1234052

Time for backsubstitution: 14.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6140

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1359809, upper bound: 1.1471890
time: 4.86 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1365473, upper bound: 1.1506145
time: 5.18 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -15.8424349, -11.6793137, -15.8788548, -11.6274643, -2.5167866, 2.5134468
1: -7.1886830, -4.3961554, -7.1995344, -4.3495641, -2.8273621, 2.7916541
2: -8.6829710, -6.1531749, -8.7388496, -6.1371684, -2.4146399, 2.4465094
3: -5.0092254, -2.4303749, -5.0451360, -2.4186850, -2.3386693, 2.3516905
4: -7.9208646, -5.2885065, -7.9782028, -5.2711463, -1.8985882, 1.9265714
5: -6.3123980, -3.7658832, -6.3535891, -3.7209442, -2.2540934, 2.2570090
6: -14.3979635, -11.0174694, -14.4203749, -10.9637451, -2.2649562, 2.2451346
7: 2.3113637, 4.8247833, 2.2664027, 4.8520002, -1.8362057, 1.8387206
8: -1.3185773, 0.9726415, -1.3359952, 0.9872360, -1.9822097, 1.9900823
9: -8.8003197, -5.7219257, -8.8195314, -5.6926484, -2.1476851, 2.1409643

Time for backsubstitution: 14.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6140

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1359809, upper bound: 1.1495479
time: 4.81 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1365474, upper bound: 1.1529787
time: 4.88 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -15.8640432, -11.6210861, -15.8424368, -11.6793060, -2.4955478, 2.5161691
1: -7.1961031, -4.3803391, -7.1887221, -4.3961544, -2.7944908, 2.8014159
2: -8.7408390, -6.1397667, -8.6829739, -6.1531687, -2.4435539, 2.4116836
3: -5.0245476, -2.4202385, -5.0092278, -2.4303646, -2.3387570, 2.3371229
4: -7.9703741, -5.2681651, -7.9208684, -5.2885032, -1.9194410, 1.9009805
5: -6.3388648, -3.7086492, -6.3123989, -3.7658811, -2.2389021, 2.2536550
6: -14.4134331, -10.9648571, -14.3979731, -11.0174685, -2.2326369, 2.2688951
7: 2.2540984, 4.8381004, 2.3113546, 4.8247833, -1.8378119, 1.8199928
8: -1.3332634, 0.9782405, -1.3185802, 0.9726424, -1.9856119, 1.9763947
9: -8.8183212, -5.7160392, -8.8003502, -5.7219272, -2.1486077, 2.1344104

Time for backsubstitution: 14.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1369974, upper bound: 1.1301774
time: 5.13 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1455257, upper bound: 1.1439946
time: 4.73 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -15.8640432, -11.6210861, -15.8640432, -11.6210861, -2.5308647, 2.5308647
1: -7.1961031, -4.3803391, -7.1961031, -4.3803391, -2.8070211, 2.8070211
2: -8.7408390, -6.1397667, -8.7408390, -6.1397667, -2.4340992, 2.4340992
3: -5.0245476, -2.4202385, -5.0245476, -2.4202385, -2.3508186, 2.3508186
4: -7.9703741, -5.2681651, -7.9703741, -5.2681651, -1.9230266, 1.9230268
5: -6.3388648, -3.7086492, -6.3388648, -3.7086492, -2.2751627, 2.2751627
6: -14.4134331, -10.9648571, -14.4134331, -10.9648571, -2.3046932, 2.3046932
7: 2.2540984, 4.8381004, 2.2540984, 4.8381004, -1.8417530, 1.8417528
8: -1.3332634, 0.9782405, -1.3332634, 0.9782405, -1.9898157, 1.9898155
9: -8.8183212, -5.7160392, -8.8183212, -5.7160392, -2.1611886, 2.1611888

Time for backsubstitution: 14.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1431517, upper bound: 1.1439947
time: 4.62 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1455266, upper bound: 1.1439949
time: 4.80 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 23.64 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.64
Output dim: 7, lower bound: -1.1295595, upper bound: 1.1335484
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.64
Output dim: 7, lower bound: -1.1301746, upper bound: 1.1369995
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.64
Output dim: 7, lower bound: -1.1295595, upper bound: 1.1358167
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.64
Output dim: 7, lower bound: -1.1301746, upper bound: 1.1392053
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.64
Output dim: 7, lower bound: -1.1370005, upper bound: 1.1301781
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.64
Output dim: 7, lower bound: -1.1392092, upper bound: 1.1301806
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.64
Output dim: 7, lower bound: -1.1370013, upper bound: 1.1301781
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.64
Output dim: 7, lower bound: -1.1392102, upper bound: 1.1301796
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.64
Output dim: 7, lower bound: -1.1389777, upper bound: 1.1335469
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.64
Output dim: 7, lower bound: -1.1396097, upper bound: 1.1369939
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.64
Output dim: 7, lower bound: -1.1389776, upper bound: 1.1358157
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.64
Output dim: 7, lower bound: -1.1396097, upper bound: 1.1392047
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.64
Output dim: 7, lower bound: -1.1461343, upper bound: 1.1301774
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.64
Output dim: 7, lower bound: -1.1486395, upper bound: 1.1301744
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.64
Output dim: 7, lower bound: -1.1461351, upper bound: 1.1301762
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.64
Output dim: 7, lower bound: -1.1486403, upper bound: 1.1301747
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.64
Output dim: 7, lower bound: -1.1295564, upper bound: 1.1426857
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.64
Output dim: 7, lower bound: -1.1301715, upper bound: 1.1461295
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.64
Output dim: 7, lower bound: -1.1295564, upper bound: 1.1451760
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.64
Output dim: 7, lower bound: -1.1301715, upper bound: 1.1486350
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.64
Output dim: 7, lower bound: -1.1369974, upper bound: 1.1396140
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.64
Output dim: 7, lower bound: -1.1392061, upper bound: 1.1396138
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.64
Output dim: 7, lower bound: -1.1369983, upper bound: 1.1396157
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.64
Output dim: 7, lower bound: -1.1392069, upper bound: 1.1396156
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.64
Output dim: 7, lower bound: -1.1359809, upper bound: 1.1471890
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.64
Output dim: 7, lower bound: -1.1365473, upper bound: 1.1506145
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.64
Output dim: 7, lower bound: -1.1359809, upper bound: 1.1495479
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.64
Output dim: 7, lower bound: -1.1365474, upper bound: 1.1529787
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.64
Output dim: 7, lower bound: -1.1369974, upper bound: 1.1301774
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.64
Output dim: 7, lower bound: -1.1455257, upper bound: 1.1439946
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.64
Output dim: 7, lower bound: -1.1431517, upper bound: 1.1439947
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.64
Output dim: 7, lower bound: -1.1455266, upper bound: 1.1439949
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=1.8950552940368652
rel_dist={7: [-1.1530177532447787, 1.1530172261927087]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6192
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6192

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0263699, upper bound: 1.0197734
time: 4.62 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0298720, upper bound: 1.0298699
time: 4.50 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 9.29 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 9.29
Output dim: 7, lower bound: -1.0263699, upper bound: 1.0197734
IS_A2, status: Status.UNKNOWN, split count: 1, time: 9.29
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

Time for backsubstitution: 13.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 6192
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6156

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0263614, upper bound: 1.0130456
time: 4.46 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0263614, upper bound: 1.0197650
time: 4.89 seconds

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

Time for backsubstitution: 14.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6192
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6192

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0197746, upper bound: 1.0263690
time: 4.53 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0197745, upper bound: 1.0263691
time: 4.66 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 23.54 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 23.54
Output dim: 7, lower bound: -1.0263614, upper bound: 1.0130456
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 23.54
Output dim: 7, lower bound: -1.0263614, upper bound: 1.0197650
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 23.54
Output dim: 7, lower bound: -1.0197746, upper bound: 1.0263690
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 23.54
Output dim: 7, lower bound: -1.0197745, upper bound: 1.0263691

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -15.7639828, -11.6762304, -15.7932472, -11.6843491, -2.3032150, 2.3287897
1: -7.1565247, -4.4119267, -7.1723380, -4.4088759, -2.6695843, 2.6892881
2: -8.7000160, -6.1713810, -8.6770706, -6.1665435, -2.3189745, 2.2857537
3: -4.9540434, -2.4564714, -4.9772396, -2.4391391, -2.1849289, 2.1828251
4: -7.9344344, -5.2966719, -7.9152336, -5.2997608, -1.7907987, 1.7781513
5: -6.2585559, -3.7710693, -6.2737679, -3.7771773, -2.0448728, 2.0772903
6: -14.3848209, -11.0217342, -14.3895826, -11.0360298, -2.1062112, 2.1138127
7: 2.3087716, 4.7735333, 2.3203688, 4.7933016, -1.7093673, 1.6935973
8: -1.2207429, 0.9365120, -1.2635236, 0.9671330, -1.8314362, 1.8332369
9: -8.7818489, -5.7919159, -8.7961731, -5.7600245, -2.0006990, 1.9807363

Time for backsubstitution: 14.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0242520, upper bound: 1.0130399
time: 4.20 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0263521, upper bound: 1.0130392
time: 4.35 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -15.7687855, -11.6553326, -15.8148584, -11.6261206, -2.3418717, 2.3920043
1: -7.1591153, -4.4076223, -7.1797762, -4.3930459, -2.6894493, 2.6990066
2: -8.7206879, -6.1691089, -8.7349358, -6.1531682, -2.3531957, 2.3087316
3: -4.9593029, -2.4537749, -4.9926186, -2.4290650, -2.2011042, 2.1997969
4: -7.9517303, -5.2917261, -7.9648147, -5.2793937, -1.8239670, 1.8024092
5: -6.2641287, -3.7503946, -6.3001766, -3.7199492, -2.0843945, 2.1297479
6: -14.3885374, -11.0027924, -14.4050694, -10.9833241, -2.1614413, 2.1670177
7: 2.2882204, 4.7757864, 2.2631087, 4.8066192, -1.7602451, 1.7155335
8: -1.2247856, 0.9384165, -1.2782221, 0.9727449, -1.8396087, 1.8501921
9: -8.7875423, -5.7901473, -8.8140306, -5.7540789, -2.0143085, 2.0096068

Time for backsubstitution: 14.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6156

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0196401, upper bound: 1.0197649
time: 5.09 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0196401, upper bound: 1.0197654
time: 4.79 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -15.8636980, -11.6238155, -15.7687855, -11.6553326, -2.4021068, 2.3713861
1: -7.1914439, -4.3808165, -7.1591153, -4.4076223, -2.7072163, 2.7009459
2: -8.7402306, -6.1401229, -8.7206879, -6.1691089, -2.3507500, 2.3714023
3: -5.0243449, -2.4224696, -4.9593029, -2.4537749, -2.2157598, 2.2040184
4: -7.9683542, -5.2682576, -7.9517303, -5.2917261, -1.8325796, 1.8388753
5: -6.3385077, -3.7134347, -6.2641287, -3.7503946, -2.1462536, 2.1136918
6: -14.4128628, -10.9652958, -14.3885374, -11.0027924, -2.1545086, 2.1689198
7: 2.2567358, 4.8378487, 2.2882204, 4.7757864, -1.7556214, 1.7697797
8: -1.3331661, 0.9763775, -1.2247856, 0.9384165, -1.8559260, 1.8392310
9: -8.8178730, -5.7164869, -8.7875423, -5.7901473, -2.0071797, 2.0193455

Time for backsubstitution: 14.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6156

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0130467, upper bound: 1.0263593
time: 4.93 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0197659, upper bound: 1.0263596
time: 4.71 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -15.8640451, -11.6210775, -15.8640451, -11.6210775, -2.4167061, 2.4167061
1: -7.1961031, -4.3803387, -7.1961031, -4.3803387, -2.7579117, 2.7579122
2: -8.7408390, -6.1397657, -8.7408390, -6.1397657, -2.4048529, 2.4048533
3: -5.0245490, -2.4202378, -5.0245490, -2.4202378, -2.2898960, 2.2898962
4: -7.9703798, -5.2681632, -7.9703798, -5.2681632, -1.8653593, 1.8653588
5: -6.3388667, -3.7086473, -6.3388667, -3.7086473, -2.2329731, 2.2329726
6: -14.4134312, -10.9648504, -14.4134312, -10.9648504, -2.1734772, 2.1734769
7: 2.2540932, 4.8381004, 2.2540932, 4.8381004, -1.7997274, 1.7997277
8: -1.3332644, 0.9782419, -1.3332644, 0.9782419, -1.9317932, 1.9317932
9: -8.8183231, -5.7160378, -8.8183231, -5.7160378, -2.0504212, 2.0504217

Time for backsubstitution: 14.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6156
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6156

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0130467, upper bound: 1.0298654
time: 5.58 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0197658, upper bound: 1.0298622
time: 4.42 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.42 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 24.42
Output dim: 7, lower bound: -1.0242520, upper bound: 1.0130399
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.42
Output dim: 7, lower bound: -1.0263521, upper bound: 1.0130392
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 24.42
Output dim: 7, lower bound: -1.0196401, upper bound: 1.0197649
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 24.42
Output dim: 7, lower bound: -1.0196401, upper bound: 1.0197654
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.42
Output dim: 7, lower bound: -1.0130467, upper bound: 1.0263593
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.42
Output dim: 7, lower bound: -1.0197659, upper bound: 1.0263596
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.42
Output dim: 7, lower bound: -1.0130467, upper bound: 1.0298654
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.42
Output dim: 7, lower bound: -1.0197658, upper bound: 1.0298622

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -15.7825575, -11.6659489, -15.7932472, -11.6843529, -2.3247108, 2.3307276
1: -7.1618528, -4.3778119, -7.1722894, -4.4088783, -2.6695566, 2.7207727
2: -8.7145014, -6.1669641, -8.6770649, -6.1665502, -2.3316636, 2.2897558
3: -4.9783878, -2.4524548, -4.9772367, -2.4391522, -2.2096701, 2.1871223
4: -7.9559340, -5.2953377, -7.9152284, -5.2997656, -1.7999637, 1.7780128
5: -6.2777367, -3.7668316, -6.2737656, -3.7771821, -2.0665898, 2.0826099
6: -14.3950129, -11.0056477, -14.3895674, -11.0360317, -2.1171000, 2.1239581
7: 2.3045621, 4.7893219, 2.3203788, 4.7933006, -1.7121594, 1.7114737
8: -1.2267282, 0.9469652, -1.2635202, 0.9671326, -1.8410678, 1.8389840
9: -8.7885666, -5.7672224, -8.7961407, -5.7600260, -1.9969697, 2.0032468

Time for backsubstitution: 14.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6192
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6192

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0197590, upper bound: 1.0130365
time: 4.45 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0197590, upper bound: 1.0130366
time: 5.61 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -15.8420916, -11.6820450, -15.7639828, -11.6762304, -2.3326554, 2.3046970
1: -7.1840630, -4.3966312, -7.1565247, -4.4119267, -2.6937447, 2.6811481
2: -8.6823711, -6.1535273, -8.7000160, -6.1713810, -2.2906551, 2.3316920
3: -5.0090218, -2.4325938, -4.9540434, -2.4564714, -2.1987147, 2.1878047
4: -7.9188414, -5.2886009, -7.9344344, -5.2966719, -1.7805891, 1.7931989
5: -6.3120413, -3.7706697, -6.2585559, -3.7710693, -2.0803530, 2.0478594
6: -14.3973999, -11.0179100, -14.3848209, -11.0217342, -2.1180854, 2.1143229
7: 2.3139982, 4.8245325, 2.3087716, 4.7735333, -1.6976433, 1.7133925
8: -1.3184814, 0.9707785, -1.2207429, 0.9365120, -1.8378491, 1.8293626
9: -8.7999001, -5.7223740, -8.7818489, -5.7919159, -1.9833040, 2.0059040

Time for backsubstitution: 14.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0130375, upper bound: 1.0242502
time: 4.77 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0130399, upper bound: 1.0263497
time: 5.26 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -15.8636971, -11.6238251, -15.7687855, -11.6553326, -2.3959346, 2.3433480
1: -7.1914425, -4.3808160, -7.1591153, -4.4076223, -2.7034121, 2.7009439
2: -8.7402325, -6.1401234, -8.7206879, -6.1691089, -2.3136892, 2.3714018
3: -5.0243440, -2.4224699, -4.9593029, -2.4537749, -2.2155592, 2.2040164
4: -7.9683499, -5.2682600, -7.9517303, -5.2917261, -1.8047895, 1.8388739
5: -6.3385086, -3.7134378, -6.2641287, -3.7503946, -2.1407347, 2.0873830
6: -14.4128609, -10.9653006, -14.3885374, -11.0027924, -2.1712828, 2.1689177
7: 2.2567406, 4.8378477, 2.2882204, 4.7757864, -1.7195840, 1.7643075
8: -1.3331652, 0.9763765, -1.2247856, 0.9384165, -1.8549323, 1.8373046
9: -8.8178740, -5.7164860, -8.7875423, -5.7901473, -2.0122151, 2.0193448

Time for backsubstitution: 14.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6156

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0197660, upper bound: 1.0196382
time: 4.84 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0197659, upper bound: 1.0263595
time: 4.62 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -15.8424368, -11.6793060, -15.8592339, -11.6419744, -2.3686960, 2.3500242
1: -7.1887221, -4.3961544, -7.1934447, -4.3846245, -2.7443190, 2.7379637
2: -8.6829739, -6.1531687, -8.7201576, -6.1421351, -2.3446584, 2.3704967
3: -5.0092278, -2.4303646, -5.0193248, -2.4231448, -2.2729249, 2.2737474
4: -7.9208684, -5.2885032, -7.9531345, -5.2734671, -1.8130035, 1.8298123
5: -6.3123989, -3.7658811, -6.3332443, -3.7293587, -2.1754878, 2.1670928
6: -14.3979731, -11.0174685, -14.4097347, -10.9837589, -2.1369781, 2.1182909
7: 2.3113546, 4.8247833, 2.2746568, 4.8356156, -1.7414978, 1.7548144
8: -1.3185802, 0.9726424, -1.3290195, 0.9763451, -1.9141531, 1.9216750
9: -8.8003502, -5.7219272, -8.8117247, -5.7177711, -2.0266829, 2.0357671

Time for backsubstitution: 14.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0173640, upper bound: 1.0276986
time: 6.27 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0173644, upper bound: 1.0298553
time: 4.44 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -15.8640432, -11.6210861, -15.8640451, -11.6210775, -2.4167051, 2.3886681
1: -7.1961031, -4.3803391, -7.1961031, -4.3803387, -2.7541075, 2.7579103
2: -8.7408390, -6.1397667, -8.7408390, -6.1397657, -2.3677912, 2.4048514
3: -5.0245476, -2.4202385, -5.0245490, -2.4202378, -2.2900305, 2.2898951
4: -7.9703741, -5.2681651, -7.9703798, -5.2681632, -1.8375697, 1.8653584
5: -6.3388648, -3.7086492, -6.3388667, -3.7086473, -2.2329721, 2.2066650
6: -14.4134331, -10.9648571, -14.4134312, -10.9648504, -2.1902509, 2.1734729
7: 2.2540984, 4.8381004, 2.2540932, 4.8381004, -1.7636900, 1.7997272
8: -1.3332634, 0.9782405, -1.3332644, 0.9782419, -1.9317923, 1.9300632
9: -8.8183212, -5.7160392, -8.8183231, -5.7160378, -2.0554576, 2.0504220

Time for backsubstitution: 14.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6156

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0240367, upper bound: 1.0231978
time: 9.10 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0197660, upper bound: 1.0197650
time: 5.64 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 29.07 seconds
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 29.07
Output dim: 7, lower bound: -1.0197590, upper bound: 1.0130365
IS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 29.07
Output dim: 7, lower bound: -1.0197590, upper bound: 1.0130366
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 29.07
Output dim: 7, lower bound: -1.0130375, upper bound: 1.0242502
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.07
Output dim: 7, lower bound: -1.0130399, upper bound: 1.0263497
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 29.07
Output dim: 7, lower bound: -1.0197660, upper bound: 1.0196382
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.07
Output dim: 7, lower bound: -1.0197659, upper bound: 1.0263595
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.07
Output dim: 7, lower bound: -1.0173640, upper bound: 1.0276986
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.07
Output dim: 7, lower bound: -1.0173644, upper bound: 1.0298553
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 29.07
Output dim: 7, lower bound: -1.0240367, upper bound: 1.0231978
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 29.07
Output dim: 7, lower bound: -1.0197660, upper bound: 1.0197650

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -15.8420877, -11.6820278, -15.7825575, -11.6659489, -2.3345976, 2.3205330
1: -7.1840577, -4.3966293, -7.1618528, -4.3778119, -2.7252607, 2.6811218
2: -8.6823711, -6.1535320, -8.7145014, -6.1669641, -2.2946692, 2.3363724
3: -5.0090199, -2.4325886, -4.9783878, -2.4524548, -2.2030387, 2.2062037
4: -7.9188547, -5.2886038, -7.9559340, -5.2953377, -1.7804661, 1.8017837
5: -6.3120413, -3.7706313, -6.2777367, -3.7668316, -2.0856748, 2.0695896
6: -14.3973923, -11.0179081, -14.3950129, -11.0056477, -2.1226964, 2.1190894
7: 2.3139858, 4.8245311, 2.3045621, 4.7893219, -1.7109468, 1.7161856
8: -1.3184791, 0.9707928, -1.2267282, 0.9469652, -1.8435967, 1.8345952
9: -8.7998686, -5.7223725, -8.7885666, -5.7672224, -1.9891868, 2.0007126

Time for backsubstitution: 14.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6140

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0123119, upper bound: 1.0234389
time: 4.45 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0130330, upper bound: 1.0263457
time: 4.61 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -15.8636971, -11.6238251, -15.7687864, -11.6553373, -2.3707895, 2.3433471
1: -7.1914425, -4.3808160, -7.1591148, -4.4076223, -2.7034111, 2.6970639
2: -8.7402325, -6.1401234, -8.7206879, -6.1691084, -2.3136888, 2.3343353
3: -5.0243440, -2.4224699, -4.9593015, -2.4537759, -2.2154679, 2.2041504
4: -7.9683499, -5.2682600, -7.9517260, -5.2917271, -1.8047886, 1.8110840
5: -6.3385086, -3.7134378, -6.2641282, -3.7503965, -2.1171424, 2.0873826
6: -14.4128609, -10.9653006, -14.3885374, -11.0028000, -2.1712790, 2.1814511
7: 2.2567406, 4.8378477, 2.2882261, 4.7757874, -1.7195840, 1.7320018
8: -1.3331652, 0.9763765, -1.2247832, 0.9384151, -1.8533506, 1.8367083
9: -8.8178740, -5.7164860, -8.7875395, -5.7901487, -2.0122147, 2.0231934

Time for backsubstitution: 14.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0176566, upper bound: 1.0196288
time: 4.69 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0197566, upper bound: 1.0196289
time: 4.83 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -15.8410730, -11.6821051, -15.8569126, -11.6465645, -2.3558373, 2.3408136
1: -7.1746106, -4.3969069, -7.1695724, -4.3859034, -2.7285852, 2.7129664
2: -8.6816797, -6.1577239, -8.7179670, -6.1498237, -2.3328724, 2.3620987
3: -5.0076327, -2.4342177, -5.0165672, -2.4296587, -2.2650747, 2.2672310
4: -7.9187384, -5.2917681, -7.9494700, -5.2789636, -1.8005495, 1.8186105
5: -6.3114157, -3.7668483, -6.3315620, -3.7310021, -2.1722870, 2.1641254
6: -14.3932858, -11.0178547, -14.4018326, -10.9844322, -2.1285977, 2.1051345
7: 2.3146734, 4.8235865, 2.2802653, 4.8335485, -1.7362590, 1.7480159
8: -1.3173747, 0.9718761, -1.3269801, 0.9750147, -1.9082422, 1.9166772
9: -8.7898903, -5.7222056, -8.7940712, -5.7182503, -2.0149755, 2.0165107

Time for backsubstitution: 14.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6140

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0167171, upper bound: 1.0248398
time: 4.80 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0173597, upper bound: 1.0276944
time: 4.67 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -15.8424339, -11.6793098, -15.8778496, -11.6317711, -2.3705900, 2.3716049
1: -7.1886754, -4.3961568, -7.1990018, -4.3504648, -2.7725573, 2.7375755
2: -8.6829700, -6.1531768, -8.7345924, -6.1376324, -2.3489285, 2.3782206
3: -5.0092244, -2.4303770, -5.0440497, -2.4192438, -2.2771826, 2.2892106
4: -7.9208646, -5.2885070, -7.9746432, -5.2721491, -1.8128567, 1.8384517
5: -6.3123980, -3.7658837, -6.3523993, -3.7252064, -2.1806602, 2.1887972
6: -14.3979607, -11.0174713, -14.4195929, -10.9676352, -2.1467633, 2.1300364
7: 2.3113651, 4.8247814, 2.2706437, 4.8515391, -1.7594860, 1.7576041
8: -1.3185768, 0.9726400, -1.3351731, 0.9868464, -1.9210606, 1.9286153
9: -8.8003159, -5.7219267, -8.8184042, -5.6930027, -2.0409141, 2.0326514

Time for backsubstitution: 14.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6140

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0167187, upper bound: 1.0269887
time: 4.72 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0173601, upper bound: 1.0298488
time: 5.96 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 25.12 seconds
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 25.12
Output dim: 7, lower bound: -1.0123119, upper bound: 1.0234389
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 25.12
Output dim: 7, lower bound: -1.0130330, upper bound: 1.0263457
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 25.12
Output dim: 7, lower bound: -1.0176566, upper bound: 1.0196288
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 25.12
Output dim: 7, lower bound: -1.0197566, upper bound: 1.0196289
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 25.12
Output dim: 7, lower bound: -1.0167171, upper bound: 1.0248398
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.12
Output dim: 7, lower bound: -1.0173597, upper bound: 1.0276944
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.12
Output dim: 7, lower bound: -1.0167187, upper bound: 1.0269887
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.12
Output dim: 7, lower bound: -1.0173601, upper bound: 1.0298488

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -15.8526039, -11.6800289, -15.8569098, -11.6465702, -2.3676021, 2.3454304
1: -7.1959190, -4.3872051, -7.1695690, -4.3859196, -2.7560434, 2.7226686
2: -8.6884556, -6.1352463, -8.7179613, -6.1498280, -2.3434172, 2.3801770
3: -5.0178537, -2.3956289, -5.0165420, -2.4296637, -2.2687001, 2.2840941
4: -7.9780273, -5.2873635, -7.9494648, -5.2789898, -1.8394809, 1.8126831
5: -6.3292007, -3.7397404, -6.3315554, -3.7310050, -2.1881282, 2.1861014
6: -14.3995237, -10.9723272, -14.4018250, -10.9844322, -2.1301041, 2.1383700
7: 2.2959580, 4.8271708, 2.2802706, 4.8335414, -1.7580605, 1.7512648
8: -1.3327789, 0.9751639, -1.3269749, 0.9750137, -1.9200058, 1.9160838
9: -8.8015890, -5.6858873, -8.7940540, -5.7182531, -2.0284429, 2.0350845

Time for backsubstitution: 14.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6156

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0173597, upper bound: 1.0109342
time: 6.76 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0173601, upper bound: 1.0276968
time: 4.72 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -15.8408470, -11.6807833, -15.8770304, -11.6325312, -2.3676915, 2.3687983
1: -7.1853933, -4.4097905, -7.1972294, -4.3576560, -2.7621284, 2.7229638
2: -8.6761827, -6.1570864, -8.7310572, -6.1396580, -2.3346224, 2.3648801
3: -4.9947948, -2.4357095, -5.0364485, -2.4219878, -2.2591834, 2.2747540
4: -7.9156041, -5.3059325, -7.9718771, -5.2812910, -1.7983308, 1.8173714
5: -6.3061075, -3.7689471, -6.3490400, -3.7267838, -2.1714425, 2.1814179
6: -14.3846607, -11.0199108, -14.4126024, -10.9688883, -2.1319575, 2.1208720
7: 2.3150859, 4.8176775, 2.2725544, 4.8477621, -1.7527218, 1.7489209
8: -1.3136597, 0.9712753, -1.3325262, 0.9861360, -1.9090900, 1.9178946
9: -8.7864370, -5.7240005, -8.8111486, -5.6940861, -2.0263457, 2.0234263

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6156

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0167171, upper bound: 1.0203223
time: 4.87 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0167187, upper bound: 1.0269896
time: 4.91 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -15.8539639, -11.6772366, -15.8778477, -11.6317739, -2.3823555, 2.3762202
1: -7.2099943, -4.3864527, -7.1989975, -4.3504810, -2.7813196, 2.7472792
2: -8.6897364, -6.1306930, -8.7345877, -6.1376371, -2.3594570, 2.3941545
3: -5.0194311, -2.3917937, -5.0440254, -2.4192493, -2.2807965, 2.3033056
4: -7.9801435, -5.2841063, -7.9746375, -5.2721777, -1.8460720, 1.8325258
5: -6.3301830, -3.7387836, -6.3523922, -3.7252100, -2.1965108, 2.2020643
6: -14.4042072, -10.9719381, -14.4195805, -10.9676361, -2.1482656, 2.1563725
7: 2.2926445, 4.8283596, 2.2706485, 4.8515334, -1.7727189, 1.7608645
8: -1.3339763, 0.9759302, -1.3351669, 0.9868450, -1.9316649, 1.9278104
9: -8.8120098, -5.6856050, -8.8183880, -5.6930094, -2.0530138, 2.0491910

Time for backsubstitution: 14.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6156
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6156

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0130330, upper bound: 1.0130321
time: 5.57 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0173601, upper bound: 1.0298484
time: 4.73 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 24.68 seconds
IS_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 24.68
Output dim: 7, lower bound: -1.0173597, upper bound: 1.0109342
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 24.68
Output dim: 7, lower bound: -1.0173601, upper bound: 1.0276968
IS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 24.68
Output dim: 7, lower bound: -1.0167171, upper bound: 1.0203223
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 24.68
Output dim: 7, lower bound: -1.0167187, upper bound: 1.0269896
IS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 24.68
Output dim: 7, lower bound: -1.0130330, upper bound: 1.0130321
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 24.68
Output dim: 7, lower bound: -1.0173601, upper bound: 1.0298484

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -15.8526039, -11.6800289, -15.8616734, -11.6257114, -2.3701453, 2.3500175
1: -7.1959190, -4.3872051, -7.1720791, -4.3817182, -2.7604027, 2.7263613
2: -8.6884556, -6.1352463, -8.7385550, -6.1476078, -2.3457656, 2.3821154
3: -5.0178537, -2.3956289, -5.0216465, -2.4270086, -2.2720265, 2.2881906
4: -7.9780273, -5.2873635, -7.9667106, -5.2742734, -1.8375602, 1.8145752
5: -6.3292007, -3.7397404, -6.3370972, -3.7103438, -2.1940713, 2.1847057
6: -14.3995237, -10.9723272, -14.4054976, -10.9656620, -2.1441846, 2.1361299
7: 2.2959580, 4.8271708, 2.2597623, 4.8356662, -1.7561460, 1.7537537
8: -1.3327789, 0.9751639, -1.3307886, 0.9769030, -1.9211349, 1.9200084
9: -8.8015890, -5.6858873, -8.7993746, -5.7165279, -2.0305815, 2.0398231

Time for backsubstitution: 14.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 451

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0154270, upper bound: 1.0232410
time: 4.70 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0173593, upper bound: 1.0242456
time: 7.57 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -15.8408470, -11.6807833, -15.8816147, -11.6118011, -2.3700562, 2.3731403
1: -7.1853933, -4.4097905, -7.1997442, -4.3538575, -2.7652159, 2.7266293
2: -8.6761827, -6.1570864, -8.7512941, -6.1374907, -2.3369021, 2.3665700
3: -4.9947948, -2.4357095, -5.0411444, -2.4193463, -2.2625146, 2.2784290
4: -7.9156041, -5.3059325, -7.9890790, -5.2767701, -1.8014040, 1.8192271
5: -6.3061075, -3.7689471, -6.3541856, -3.7061410, -2.1773753, 2.1802630
6: -14.3846607, -11.0199108, -14.4161720, -10.9505157, -2.1403961, 2.1213841
7: 2.3150859, 4.8176775, 2.2522511, 4.8496952, -1.7491217, 1.7512283
8: -1.3136597, 0.9712753, -1.3358469, 0.9880033, -1.9102058, 1.9202094
9: -8.7864370, -5.7240005, -8.8160114, -5.6923900, -2.0274520, 2.0286644

Time for backsubstitution: 14.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 451

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0148015, upper bound: 1.0225695
time: 4.69 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0167166, upper bound: 1.0269887
time: 4.95 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -15.8539639, -11.6772366, -15.8825960, -11.6109161, -2.3848996, 2.3808060
1: -7.2099943, -4.3864527, -7.2015643, -4.3462954, -2.7847090, 2.7509794
2: -8.6897364, -6.1306930, -8.7551699, -6.1354194, -2.3617964, 2.3961072
3: -5.0194311, -2.3917937, -5.0491791, -2.4165761, -2.2841516, 2.3073254
4: -7.9801435, -5.2841063, -7.9918866, -5.2674603, -1.8441501, 1.8344189
5: -6.3301830, -3.7387836, -6.3579330, -3.7045503, -2.2024534, 2.2006755
6: -14.4042072, -10.9719381, -14.4232597, -10.9488525, -2.1571927, 2.1541362
7: 2.2926445, 4.8283596, 2.2501454, 4.8536558, -1.7689497, 1.7633424
8: -1.3339763, 0.9759302, -1.3389740, 0.9887328, -1.9327981, 1.9308038
9: -8.8120098, -5.6856050, -8.8236914, -5.6912827, -2.0543046, 2.0539403

Time for backsubstitution: 14.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 451

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0112482, upper bound: 1.0154295
time: 7.26 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0173592, upper bound: 1.0225697
time: 8.28 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 29.94 seconds
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 29.94
Output dim: 7, lower bound: -1.0154270, upper bound: 1.0232410
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 29.94
Output dim: 7, lower bound: -1.0173593, upper bound: 1.0242456
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 29.94
Output dim: 7, lower bound: -1.0148015, upper bound: 1.0225695
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 29.94
Output dim: 7, lower bound: -1.0167166, upper bound: 1.0269887
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 29.94
Output dim: 7, lower bound: -1.0112482, upper bound: 1.0154295
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 29.94
Output dim: 7, lower bound: -1.0173592, upper bound: 1.0225697

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -15.8408470, -11.6807852, -15.8816175, -11.6117973, -2.3624206, 2.3727121
1: -7.1853914, -4.4097900, -7.1997461, -4.3538570, -2.7489471, 2.7266273
2: -8.6761818, -6.1570892, -8.7512951, -6.1374912, -2.3368998, 2.3516557
3: -4.9947948, -2.4357119, -5.0411453, -2.4193473, -2.2625132, 2.2697260
4: -7.9156051, -5.3059335, -7.9890785, -5.2767706, -1.8012986, 1.8185780
5: -6.3061066, -3.7689486, -6.3541851, -3.7061410, -2.1716449, 2.1788785
6: -14.3846607, -11.0199089, -14.4161730, -10.9505177, -2.1163580, 2.1213841
7: 2.3150873, 4.8176751, 2.2522511, 4.8496957, -1.7463861, 1.7364373
8: -1.3136601, 0.9712763, -1.3358464, 0.9880018, -1.9097583, 1.9182057
9: -8.7864361, -5.7240043, -8.8160095, -5.6923914, -2.0237966, 2.0328784

Time for backsubstitution: 14.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 451

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0123217, upper bound: 1.0250495
time: 4.64 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0123216, upper bound: 1.0269886
time: 4.78 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 23.85 seconds
IS_A2_B2_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 23.85
Output dim: 7, lower bound: -1.0123217, upper bound: 1.0250495
IS_A2_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 23.85
Output dim: 7, lower bound: -1.0123216, upper bound: 1.0269886

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -15.8408470, -11.6807852, -15.8816185, -11.6118031, -2.3624194, 2.3655205
1: -7.1853914, -4.4097900, -7.1997423, -4.3538575, -2.7481389, 2.7104282
2: -8.6761818, -6.1570892, -8.7512941, -6.1374917, -2.3220620, 2.3506799
3: -4.9947948, -2.4357119, -5.0411434, -2.4193482, -2.2538700, 2.2688251
4: -7.9156051, -5.3059335, -7.9890780, -5.2767701, -1.8007979, 1.8185776
5: -6.3061066, -3.7689486, -6.3541822, -3.7061410, -2.1716454, 2.1745420
6: -14.3846607, -11.0199089, -14.4161692, -10.9505205, -2.1160076, 2.0973835
7: 2.3150873, 4.8176751, 2.2522521, 4.8496943, -1.7343836, 1.7364364
8: -1.3136601, 0.9712763, -1.3358483, 0.9880018, -1.9081874, 1.9180918
9: -8.7864361, -5.7240043, -8.8160095, -5.6923919, -2.0299053, 2.0328765

Time for backsubstitution: 14.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0101537, upper bound: 1.0269913
time: 4.87 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0101537, upper bound: 1.0228833
time: 5.62 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 24.96 seconds
IS_A2_B2_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 24.96
Output dim: 7, lower bound: -1.0101537, upper bound: 1.0269913
IS_A2_B2_A1_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 24.96
Output dim: 7, lower bound: -1.0101537, upper bound: 1.0228833

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -15.8385181, -11.6853657, -15.8816185, -11.6118031, -2.3579695, 2.3541589
1: -7.1615591, -4.4110713, -7.1997423, -4.3538575, -2.7240691, 2.7141986
2: -8.6740112, -6.1647787, -8.7512941, -6.1374917, -2.3180513, 2.3401794
3: -4.9920497, -2.4422123, -5.0411434, -2.4193482, -2.2521179, 2.2626448
4: -7.9119196, -5.3114281, -7.9890780, -5.2767701, -1.7937241, 1.8082646
5: -6.3044100, -3.7705786, -6.3541822, -3.7061410, -2.1699288, 2.1723225
6: -14.3767519, -11.0205612, -14.4161692, -10.9505205, -2.1033828, 2.0907283
7: 2.3206949, 4.8156137, 2.2522521, 4.8496943, -1.7286172, 1.7330372
8: -1.3116341, 0.9699473, -1.3358483, 0.9880018, -1.9051552, 1.9135303
9: -8.7688017, -5.7244692, -8.8160095, -5.6923919, -2.0109367, 2.0376587

Time for backsubstitution: 14.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6140

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0101367, upper bound: 1.0269891
time: 4.62 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0101367, upper bound: 1.0269887
time: 4.76 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 23.72 seconds
IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 23.72
Output dim: 7, lower bound: -1.0101367, upper bound: 1.0269891
IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 23.72
Output dim: 7, lower bound: -1.0101367, upper bound: 1.0269887

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -15.8385181, -11.6853657, -15.8808756, -11.6124535, -2.3572307, 2.3532276
1: -7.1615591, -4.4110713, -7.1981564, -4.3603520, -2.7185516, 2.7130775
2: -8.6740112, -6.1647787, -8.7481441, -6.1393671, -2.3144598, 2.3358245
3: -4.9920497, -2.4422123, -5.0343218, -2.4218750, -2.2495270, 2.2566411
4: -7.9119196, -5.3114281, -7.9865284, -5.2850442, -1.7848043, 1.8060957
5: -6.3044100, -3.7705786, -6.3511186, -3.7076044, -2.1682155, 2.1690621
6: -14.3767519, -11.0205612, -14.4098206, -10.9516754, -2.1025033, 2.0841446
7: 2.3206949, 4.8156137, 2.2539964, 4.8462934, -1.7256796, 1.7315767
8: -1.3116341, 0.9699473, -1.3333859, 0.9873476, -1.9028249, 1.9097569
9: -8.7688017, -5.7244692, -8.8094673, -5.6933866, -2.0099034, 2.0324268

Time for backsubstitution: 14.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6135

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0099296, upper bound: 1.0244332
time: 8.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0101329, upper bound: 1.0269843
time: 4.87 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -15.8385181, -11.6853657, -15.8940125, -11.6091061, -2.3614166, 2.3684461
1: -7.1615591, -4.4110713, -7.2226458, -4.3373313, -2.7364616, 2.7294950
2: -8.6740112, -6.1647787, -8.7612495, -6.1132946, -2.3427091, 2.3469243
3: -4.9920497, -2.4422123, -5.0583587, -2.3782153, -2.2692022, 2.2735307
4: -7.9119196, -5.3114281, -8.0500069, -5.2634354, -1.8055611, 1.8320394
5: -6.3044100, -3.7705786, -6.3752532, -3.6781166, -2.1844149, 2.1917231
6: -14.3767519, -11.0205612, -14.4294605, -10.9042425, -2.1221747, 2.1039400
7: 2.3206949, 4.8156137, 2.2319770, 4.8568678, -1.7332616, 1.7468901
8: -1.3116341, 0.9699473, -1.3530340, 0.9918780, -1.9076750, 1.9264512
9: -8.7688017, -5.7244692, -8.8342686, -5.6553617, -2.0188594, 2.0567310

Time for backsubstitution: 14.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6135

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0099296, upper bound: 1.0244332
time: 7.70 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0101329, upper bound: 1.0269832
time: 4.83 seconds

## Summary of splitting at layer (split count: 10)
- Time for IS candidates: 26.96 seconds
IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 11, time: 26.96
Output dim: 7, lower bound: -1.0099296, upper bound: 1.0244332
IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 11, time: 26.96
Output dim: 7, lower bound: -1.0101329, upper bound: 1.0269843
IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 11, time: 26.96
Output dim: 7, lower bound: -1.0099296, upper bound: 1.0244332
IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 26.96
Output dim: 7, lower bound: -1.0101329, upper bound: 1.0269832

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -15.8385172, -11.6853714, -15.8808756, -11.6124535, -2.3569398, 2.3417602
1: -7.1615605, -4.4110694, -7.1981564, -4.3603520, -2.6979923, 2.7130780
2: -8.6740112, -6.1647792, -8.7481441, -6.1393671, -2.3144617, 2.3095698
3: -4.9920492, -2.4422109, -5.0343218, -2.4218750, -2.2504339, 2.2566409
4: -7.9119182, -5.3114281, -7.9865284, -5.2850442, -1.7847872, 1.7820077
5: -6.3044109, -3.7705793, -6.3511186, -3.7076044, -2.1759279, 2.1690624
6: -14.3767490, -11.0205612, -14.4098206, -10.9516754, -2.0991526, 2.0841334
7: 2.3206944, 4.8156133, 2.2539964, 4.8462934, -1.7254255, 1.7185216
8: -1.3116336, 0.9699492, -1.3333859, 0.9873476, -1.8896747, 1.9095705
9: -8.7688026, -5.7244692, -8.8094673, -5.6933866, -2.0092397, 2.0195765

Time for backsubstitution: 14.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6135

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0031962, upper bound: 1.0232196
time: 5.03 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0076301, upper bound: 1.0267477
time: 5.38 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -15.8385172, -11.6853714, -15.8940125, -11.6091061, -2.3611257, 2.3569775
1: -7.1615605, -4.4110694, -7.2226458, -4.3373313, -2.7159019, 2.7290883
2: -8.6740112, -6.1647792, -8.7612495, -6.1132946, -2.3427110, 2.3206694
3: -4.9920492, -2.4422109, -5.0583587, -2.3782153, -2.2701092, 2.2735302
4: -7.9119182, -5.3114281, -8.0500069, -5.2634354, -1.8051987, 1.8079836
5: -6.3044109, -3.7705793, -6.3752532, -3.6781166, -2.1921268, 2.1917231
6: -14.3767490, -11.0205612, -14.4294605, -10.9042425, -2.1187227, 2.1039288
7: 2.3206944, 4.8156133, 2.2319770, 4.8568678, -1.7330074, 1.7338877
8: -1.3116336, 0.9699492, -1.3530340, 0.9918780, -1.8946443, 1.9262645
9: -8.7688026, -5.7244692, -8.8342686, -5.6553617, -2.0181961, 2.0437624

Time for backsubstitution: 14.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6135

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0098507, upper bound: 1.0267445
time: 7.85 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0098507, upper bound: 1.0267460
time: 5.18 seconds

## Summary of splitting at layer (split count: 11)
- Time for IS candidates: 27.36 seconds
IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 12, time: 27.36
Output dim: 7, lower bound: -1.0031962, upper bound: 1.0232196
IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 27.36
Output dim: 7, lower bound: -1.0076301, upper bound: 1.0267477
IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 27.36
Output dim: 7, lower bound: -1.0098507, upper bound: 1.0267445
IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 27.36
Output dim: 7, lower bound: -1.0098507, upper bound: 1.0267460

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -15.8385172, -11.6853714, -15.8808746, -11.6124535, -2.3456256, 2.3417597
1: -7.1615605, -4.4110694, -7.1981559, -4.3603525, -2.6978598, 2.6925354
2: -8.6740112, -6.1647792, -8.7481441, -6.1393671, -2.2882037, 2.3093426
3: -4.9920492, -2.4422109, -5.0343232, -2.4218767, -2.2504334, 2.2575443
4: -7.9119182, -5.3114281, -7.9865279, -5.2850442, -1.7606783, 1.7820078
5: -6.3044109, -3.7705793, -6.3511181, -3.7076051, -2.1759274, 2.1767795
6: -14.3767490, -11.0205612, -14.4098206, -10.9516745, -2.0991521, 2.0810149
7: 2.3206944, 4.8156133, 2.2539959, 4.8462930, -1.7126946, 1.7185214
8: -1.3116336, 0.9699492, -1.3333855, 0.9873486, -1.8896747, 1.8966315
9: -8.7688026, -5.7244692, -8.8094692, -5.6933870, -1.9972105, 2.0195768

Time for backsubstitution: 14.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0074446, upper bound: 1.0214074
time: 5.11 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0075681, upper bound: 1.0266832
time: 5.66 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -15.8374414, -11.6873226, -15.8838949, -11.6399412, -2.3263450, 2.3190596
1: -7.1599674, -4.4119534, -7.1802335, -4.3446550, -2.6952286, 2.6808519
2: -8.6734486, -6.1654162, -8.7498140, -6.1723671, -2.2827578, 2.2899740
3: -4.9914174, -2.4435105, -5.0479083, -2.4037924, -2.2375298, 2.2490888
4: -7.9112782, -5.3127885, -8.0436449, -5.3001461, -1.7676053, 1.7833219
5: -6.3035464, -3.7727845, -6.3619318, -3.6836934, -2.1745179, 2.1735530
6: -14.3743086, -11.0217133, -14.4135494, -10.9130497, -2.0935516, 2.0881429
7: 2.3219647, 4.8152370, 2.2512903, 4.8367658, -1.7104263, 1.7051251
8: -1.3115215, 0.9682770, -1.3339157, 0.9821987, -1.8842094, 1.9047875
9: -8.7680779, -5.7252617, -8.8290195, -5.7124124, -1.9487376, 2.0138969

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0096690, upper bound: 1.0211734
time: 4.57 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0097887, upper bound: 1.0266814
time: 7.14 seconds

## Summary of splitting at layer (split count: 12)
- Time for IS candidates: 26.09 seconds
IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 13, time: 26.09
Output dim: 7, lower bound: -1.0074446, upper bound: 1.0214074
IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 13, time: 26.09
Output dim: 7, lower bound: -1.0075681, upper bound: 1.0266832
IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 13, time: 26.09
Output dim: 7, lower bound: -1.0096690, upper bound: 1.0211734
IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 13, time: 26.09
Output dim: 7, lower bound: -1.0097887, upper bound: 1.0266814
IS_A2_B2_A1_B2_A1_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 26.09
Output dim: 7, lower bound: -1.0098507, upper bound: 1.0267460
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=1.8196511268615723
rel_dist={7: [-1.029880698625119, 1.029879313881131]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2418.65 seconds
