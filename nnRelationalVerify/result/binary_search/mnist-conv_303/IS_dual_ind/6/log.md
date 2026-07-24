## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 0.83234478323
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.5797892, 2.5797892)
1: (-3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.4460940, 2.4460940)
2: (1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9857671, 1.9857671)
3: (-7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391)
4: (-2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.9272163, 1.9272163)
5: (-4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8746789, 1.8746789)
6: (-4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.5114598, 2.5114598)
7: (-8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.8371913, 1.8371913)
8: (-4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.2200575, 2.2200575)
9: (-12.0749950, -9.7447462, -12.0749950, -9.7447462, -2.2258463, 2.2258465)

## BASE Result
execution time: IAR + LP analysis = 15.79 + 32.58 = 48.37 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3551.63 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=1.9464805126190186
rel_dist={2: [-1.1320473731522231, 1.1320474951997723]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=1.8296241760253906
rel_dist={2: [-0.835577596891917, 0.8355774930246449]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=1.7517199516296387
rel_dist={2: [-0.5785716541258044, 0.5785690000616741]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=1.7906718254089355
rel_dist={2: [-0.7104656444989477, 0.7104623396628322]}

## Binary Search Result
Binary search time: 207.40 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual_ind) starts
Time budget: 3344.23 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 5733
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4625

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2084882, upper bound: 1.1960950
time: 4.87 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2084840, upper bound: 1.2084825
time: 6.52 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 11.59 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 11.59
Output dim: 2, lower bound: -1.2084882, upper bound: 1.1960950
IS_A2, status: Status.UNKNOWN, split count: 1, time: 11.59
Output dim: 2, lower bound: -1.2084840, upper bound: 1.2084825

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -10.8091249, -8.2323895, -10.8108559, -8.2310667, -2.3614693, 2.3604422
1: -3.3391936, -0.9065280, -3.3504086, -0.9043145, -2.4104047, 2.4200850
2: 1.6559384, 3.6253042, 1.6547616, 3.6405287, -1.9843817, 1.9704247
3: -7.2581921, -5.2406983, -7.2597189, -5.2326798, -2.0255122, 2.0190206
4: -2.3505065, -0.4264469, -2.3514392, -0.4242229, -1.7102630, 1.7084417
5: -4.6192904, -2.7543774, -4.6279707, -2.7532918, -1.8659985, 1.8735933
6: -4.7066774, -2.2050157, -4.7157536, -2.2042937, -2.3376937, 2.3473353
7: -8.7201481, -6.8739891, -8.7202091, -6.8711286, -1.6217787, 1.6186253
8: -4.6400385, -2.4291024, -4.6486425, -2.4285851, -2.1563048, 2.1651754
9: -12.0747089, -9.7458115, -12.0749950, -9.7447462, -1.8228846, 1.8218961

Time for backsubstitution: 15.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4625

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1960950, upper bound: 1.1960953
time: 5.27 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1960950, upper bound: 1.1960974
time: 6.18 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -10.8349228, -8.2258577, -10.8108501, -8.2310715, -2.3918738, 2.3924861
1: -3.3603773, -0.8296475, -3.3503962, -0.9043207, -2.4484572, 2.4710927
2: 1.5636239, 3.6523647, 1.6547655, 3.6404843, -2.0213914, 1.9975992
3: -7.3162060, -5.2255607, -7.2597160, -5.2326984, -2.0835075, 2.0341554
4: -2.3792734, -0.4216075, -2.3514366, -0.4242287, -1.7403036, 1.7236032
5: -4.6341038, -2.6942487, -4.6279430, -2.7532945, -1.8808093, 1.9330084
6: -4.7340474, -2.1549864, -4.7157321, -2.2042959, -2.3945322, 2.3841028
7: -8.7316875, -6.8539867, -8.7202091, -6.8711371, -1.6402378, 1.6385422
8: -4.6648607, -2.3736062, -4.6486158, -2.4285865, -2.1785150, 2.2217104
9: -12.0853224, -9.7414150, -12.0749941, -9.7447462, -1.8353980, 1.8387198

Time for backsubstitution: 14.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4625

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1960957, upper bound: 1.2084851
time: 5.46 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1960950, upper bound: 1.2084860
time: 6.03 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 26.65 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 26.65
Output dim: 2, lower bound: -1.1960950, upper bound: 1.1960953
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 26.65
Output dim: 2, lower bound: -1.1960950, upper bound: 1.1960974
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 26.65
Output dim: 2, lower bound: -1.1960957, upper bound: 1.2084851
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 26.65
Output dim: 2, lower bound: -1.1960950, upper bound: 1.2084860

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -10.8091249, -8.2323895, -10.8091249, -8.2323895, -2.3569942, 2.3569946
1: -3.3391936, -0.9065280, -3.3391936, -0.9065280, -2.4079370, 2.4079366
2: 1.6559384, 3.6253042, 1.6559384, 3.6253042, -1.9693658, 1.9693658
3: -7.2581921, -5.2406983, -7.2581921, -5.2406983, -2.0174937, 2.0174937
4: -2.3505065, -0.4264469, -2.3505065, -0.4264469, -1.7068282, 1.7068279
5: -4.6192904, -2.7543774, -4.6192904, -2.7543774, -1.8649130, 1.8649130
6: -4.7066774, -2.2050157, -4.7066774, -2.2050157, -2.3362241, 2.3362238
7: -8.7201481, -6.8739891, -8.7201481, -6.8739891, -1.6164410, 1.6164407
8: -4.6400385, -2.4291024, -4.6400385, -2.4291024, -2.1557698, 2.1557696
9: -12.0747089, -9.7458115, -12.0747089, -9.7458115, -1.8204632, 1.8204632

Time for backsubstitution: 15.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5733
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5733

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1960744, upper bound: 1.1899116
time: 4.75 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1960713, upper bound: 1.1960601
time: 4.89 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -10.8091249, -8.2323895, -10.8349228, -8.2258577, -2.3674989, 2.3874116
1: -3.3391936, -0.9065280, -3.3603773, -0.8296475, -2.4589453, 2.4285221
2: 1.6559384, 3.6253042, 1.5636239, 3.6523647, -1.9964263, 2.0063572
3: -7.2581921, -5.2406983, -7.3162060, -5.2255607, -2.0326314, 2.0755076
4: -2.3505065, -0.4264469, -2.3792734, -0.4216075, -1.7124184, 1.7368739
5: -4.6192904, -2.7543774, -4.6341038, -2.6942487, -1.9244661, 1.8797264
6: -4.7066774, -2.2050157, -4.7340474, -2.1549864, -2.3729706, 2.3666034
7: -8.7201481, -6.8739891, -8.7316875, -6.8539867, -1.6363611, 1.6298505
8: -4.6400385, -2.4291024, -4.6648607, -2.3736062, -2.2123237, 2.1771808
9: -12.0747089, -9.7458115, -12.0853224, -9.7414150, -1.8248982, 1.8329818

Time for backsubstitution: 15.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5733
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5733

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1960744, upper bound: 1.1899120
time: 6.44 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1960722, upper bound: 1.1960610
time: 5.65 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -10.8349228, -8.2258577, -10.8091249, -8.2323895, -2.3874116, 2.3674989
1: -3.3603773, -0.8296475, -3.3391936, -0.9065280, -2.4285221, 2.4589453
2: 1.5636239, 3.6523647, 1.6559384, 3.6253042, -2.0063570, 1.9964263
3: -7.3162060, -5.2255607, -7.2581921, -5.2406983, -2.0755076, 2.0326314
4: -2.3792734, -0.4216075, -2.3505065, -0.4264469, -1.7368737, 1.7124183
5: -4.6341038, -2.6942487, -4.6192904, -2.7543774, -1.8797264, 1.9244664
6: -4.7340474, -2.1549864, -4.7066774, -2.2050157, -2.3666034, 2.3729703
7: -8.7316875, -6.8539867, -8.7201481, -6.8739891, -1.6298506, 1.6363615
8: -4.6648607, -2.3736062, -4.6400385, -2.4291024, -2.1771808, 2.2123237
9: -12.0853224, -9.7414150, -12.0747089, -9.7458115, -1.8329816, 1.8248982

Time for backsubstitution: 15.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5733
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5733

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1960648, upper bound: 1.2022138
time: 5.63 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1960618, upper bound: 1.2084476
time: 5.13 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -10.8349228, -8.2258577, -10.8349228, -8.2258577, -2.4101081, 2.4101071
1: -3.3603773, -0.8296475, -3.3603773, -0.8296475, -2.4677806, 2.4677806
2: 1.5636239, 3.6523647, 1.5636239, 3.6523647, -2.0354018, 2.0354018
3: -7.3162060, -5.2255607, -7.3162060, -5.2255607, -2.0906453, 2.0906453
4: -2.3792734, -0.4216075, -2.3792734, -0.4216075, -1.7381046, 1.7381045
5: -4.6341038, -2.6942487, -4.6341038, -2.6942487, -1.9227667, 1.9227672
6: -4.7340474, -2.1549864, -4.7340474, -2.1549864, -2.4028339, 2.4028339
7: -8.7316875, -6.8539867, -8.7316875, -6.8539867, -1.6528857, 1.6528857
8: -4.6648607, -2.3736062, -4.6648607, -2.3736062, -2.2294025, 2.2294028
9: -12.0853224, -9.7414150, -12.0853224, -9.7414150, -1.8462110, 1.8462114

Time for backsubstitution: 14.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5733
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5733

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1960642, upper bound: 1.2022144
time: 6.42 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1960617, upper bound: 1.2084471
time: 6.13 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 27.72 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 27.72
Output dim: 2, lower bound: -1.1960744, upper bound: 1.1899116
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 27.72
Output dim: 2, lower bound: -1.1960713, upper bound: 1.1960601
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 27.72
Output dim: 2, lower bound: -1.1960744, upper bound: 1.1899120
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 27.72
Output dim: 2, lower bound: -1.1960722, upper bound: 1.1960610
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 27.72
Output dim: 2, lower bound: -1.1960648, upper bound: 1.2022138
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 27.72
Output dim: 2, lower bound: -1.1960618, upper bound: 1.2084476
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 27.72
Output dim: 2, lower bound: -1.1960642, upper bound: 1.2022144
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 27.72
Output dim: 2, lower bound: -1.1960617, upper bound: 1.2084471

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -10.8084745, -8.2390308, -10.8091249, -8.2323895, -2.3564744, 2.3504043
1: -3.3329079, -0.9067831, -3.3391936, -0.9065280, -2.4004688, 2.4070816
2: 1.6565111, 3.6167736, 1.6559384, 3.6253042, -1.9687853, 1.9606941
3: -7.2572231, -5.2426581, -7.2581921, -5.2406983, -2.0165248, 2.0155339
4: -2.3337822, -0.4268136, -2.3505065, -0.4264469, -1.6900164, 1.7062099
5: -4.6163568, -2.7548733, -4.6192904, -2.7543774, -1.8619795, 1.8644171
6: -4.6937370, -2.2057319, -4.7066774, -2.2050157, -2.3224626, 2.3357122
7: -8.7166748, -6.8750615, -8.7201481, -6.8739891, -1.6112916, 1.6139733
8: -4.6383018, -2.4350104, -4.6400385, -2.4291024, -2.1512475, 2.1471510
9: -12.0737743, -9.7603779, -12.0747089, -9.7458115, -1.8187952, 1.8062973

Time for backsubstitution: 14.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5733

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1899222, upper bound: 1.1899224
time: 5.62 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1899222, upper bound: 1.1899245
time: 6.10 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -10.8637476, -8.2271538, -10.8091240, -8.2323971, -2.4012523, 2.3677669
1: -3.3551629, -0.8717890, -3.3391862, -0.9065289, -2.4339728, 2.4416270
2: 1.6043967, 3.6315553, 1.6559391, 3.6252933, -2.0017803, 1.9756162
3: -7.2652884, -5.2359872, -7.2581911, -5.2407017, -2.0245867, 2.0222039
4: -2.3649111, -0.3222642, -2.3504829, -0.4264479, -1.7383885, 1.7369146
5: -4.6255507, -2.7302155, -4.6192870, -2.7543774, -1.8711734, 1.8890715
6: -4.7280998, -2.1204927, -4.7066650, -2.2050164, -2.3649015, 2.3671803
7: -8.7214088, -6.8390384, -8.7201405, -6.8739910, -1.6369102, 1.6348984
8: -4.6898065, -2.4254057, -4.6400371, -2.4291124, -2.2016869, 2.1939752
9: -12.1676559, -9.7390251, -12.0747080, -9.7458324, -1.8475015, 1.8551562

Time for backsubstitution: 14.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5733

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1899222, upper bound: 1.1960717
time: 4.98 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1899222, upper bound: 1.1960739
time: 5.40 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -10.8084745, -8.2390308, -10.8349228, -8.2258577, -2.3669791, 2.3808212
1: -3.3329079, -0.9067831, -3.3603773, -0.8296475, -2.4514933, 2.4276667
2: 1.6565111, 3.6167736, 1.5636239, 3.6523647, -1.9958537, 1.9976652
3: -7.2572231, -5.2426581, -7.3162060, -5.2255607, -2.0316625, 2.0735478
4: -2.3337822, -0.4268136, -2.3792734, -0.4216075, -1.6956066, 1.7362554
5: -4.6163568, -2.7548733, -4.6341038, -2.6942487, -1.9189086, 1.8792305
6: -4.6937370, -2.2057319, -4.7340474, -2.1549864, -2.3592162, 2.3660917
7: -8.7166748, -6.8750615, -8.7316875, -6.8539867, -1.6312122, 1.6273831
8: -4.6383018, -2.4350104, -4.6648607, -2.3736062, -2.2078013, 2.1685619
9: -12.0737743, -9.7603779, -12.0853224, -9.7414150, -1.8232303, 1.8188162

Time for backsubstitution: 14.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5733

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2022181, upper bound: 1.1899101
time: 7.65 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2022181, upper bound: 1.1899122
time: 6.35 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -10.8637476, -8.2271538, -10.8349228, -8.2258663, -2.4071908, 2.3981833
1: -3.3551629, -0.8717890, -3.3603733, -0.8296473, -2.4794440, 2.4622130
2: 1.6043967, 3.6315553, 1.5636244, 3.6523528, -2.0308254, 2.0212977
3: -7.2652884, -5.2359872, -7.3162050, -5.2255630, -2.0397253, 2.0802178
4: -2.3649111, -0.3222642, -2.3792491, -0.4216084, -1.7439775, 1.7533388
5: -4.6255507, -2.7302155, -4.6340995, -2.6942494, -1.9313014, 1.9038839
6: -4.7280998, -2.1204927, -4.7340341, -2.1549869, -2.4019704, 2.3976068
7: -8.7214088, -6.8390384, -8.7316818, -6.8539877, -1.6553245, 1.6377854
8: -4.6898065, -2.4254057, -4.6648579, -2.3736148, -2.2582407, 2.2153854
9: -12.1676559, -9.7390251, -12.0853195, -9.7414398, -1.8519630, 1.8676753

Time for backsubstitution: 15.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5733

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2022181, upper bound: 1.1960616
time: 5.65 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2022180, upper bound: 1.1960618
time: 5.76 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -10.8342667, -8.2325087, -10.8091249, -8.2323895, -2.3868899, 2.3609042
1: -3.3540924, -0.8299019, -3.3391936, -0.9065280, -2.4210649, 2.4580972
2: 1.5641996, 3.6438277, 1.6559384, 3.6253042, -2.0057709, 1.9878893
3: -7.3152289, -5.2275262, -7.2581921, -5.2406983, -2.0745306, 2.0306659
4: -2.3626270, -0.4219837, -2.3505065, -0.4264469, -1.7200516, 1.7117803
5: -4.6311669, -2.6947558, -4.6192904, -2.7543774, -1.8767896, 1.9225254
6: -4.7211199, -2.1557031, -4.7066774, -2.2050157, -2.3528943, 2.3724554
7: -8.7282162, -6.8550620, -8.7201481, -6.8739891, -1.6247003, 1.6339036
8: -4.6630821, -2.3795116, -4.6400385, -2.4291024, -2.1726661, 2.2037134
9: -12.0843811, -9.7559776, -12.0747089, -9.7458115, -1.8313198, 1.8107421

Time for backsubstitution: 14.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5733

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1899124, upper bound: 1.2022200
time: 6.88 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1899120, upper bound: 1.2022185
time: 6.16 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -10.8896751, -8.2206631, -10.8091240, -8.2323971, -2.4209166, 2.3782549
1: -3.3763201, -0.7948638, -3.3391862, -0.9065289, -2.4546809, 2.4621947
2: 1.5120236, 3.6585867, 1.6559391, 3.6252933, -2.0135539, 2.0026476
3: -7.3235197, -5.2208347, -7.2581911, -5.2407017, -2.0828180, 2.0373564
4: -2.3939757, -0.3174410, -2.3504829, -0.4264479, -1.7685866, 1.7423782
5: -4.6403561, -2.6699915, -4.6192870, -2.7543774, -1.8859787, 1.9268653
6: -4.7553735, -2.0704236, -4.7066650, -2.2050164, -2.3952875, 2.3763371
7: -8.7329617, -6.8189249, -8.7201405, -6.8739910, -1.6402068, 1.6527355
8: -4.7146053, -2.3699040, -4.6400371, -2.4291124, -2.2232637, 2.2505424
9: -12.1783562, -9.7346020, -12.0747080, -9.7458324, -1.8555276, 1.8595963

Time for backsubstitution: 14.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5733

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1899124, upper bound: 1.2084521
time: 5.55 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1899124, upper bound: 1.2084526
time: 6.16 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -10.8342667, -8.2325087, -10.8349228, -8.2258577, -2.4095683, 2.4035125
1: -3.3540924, -0.8299019, -3.3603773, -0.8296475, -2.4603267, 2.4669356
2: 1.5641996, 3.6438277, 1.5636239, 3.6523647, -2.0348158, 2.0267029
3: -7.3152289, -5.2275262, -7.3162060, -5.2255607, -2.0896683, 2.0886798
4: -2.3626270, -0.4219837, -2.3792734, -0.4216075, -1.7213414, 1.7374750
5: -4.6311669, -2.6947558, -4.6341038, -2.6942487, -1.9171996, 1.9208417
6: -4.7211199, -2.1557031, -4.7340474, -2.1549864, -2.3891068, 2.4023213
7: -8.7282162, -6.8550620, -8.7316875, -6.8539867, -1.6477339, 1.6504226
8: -4.6630821, -2.3795116, -4.6648607, -2.3736062, -2.2248878, 2.2207925
9: -12.0843811, -9.7559776, -12.0853224, -9.7414150, -1.8445525, 1.8320475

Time for backsubstitution: 14.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5733

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1899123, upper bound: 1.2022137
time: 6.13 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1899120, upper bound: 1.2022157
time: 6.07 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -10.8896751, -8.2206631, -10.8349228, -8.2258663, -2.4309316, 2.4208608
1: -3.3763201, -0.7948638, -3.3603733, -0.8296473, -2.4939718, 2.4843340
2: 1.5120236, 3.6585867, 1.5636244, 3.6523528, -2.0425990, 2.0503423
3: -7.3235197, -5.2208347, -7.3162050, -5.2255630, -2.0979567, 2.0953703
4: -2.3939757, -0.3174410, -2.3792491, -0.4216084, -1.7700260, 1.7602110
5: -4.6403561, -2.6699915, -4.6340995, -2.6942494, -1.9461067, 1.9430346
6: -4.7553735, -2.0704236, -4.7340341, -2.1549869, -2.4315076, 2.4104276
7: -8.7329617, -6.8189249, -8.7316818, -6.8539877, -1.6619825, 1.6594090
8: -4.7146053, -2.3699040, -4.6648579, -2.3736148, -2.2754855, 2.2676208
9: -12.1783562, -9.7346020, -12.0853195, -9.7414398, -1.8626547, 1.8808618

Time for backsubstitution: 14.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5733

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1899120, upper bound: 1.2084477
time: 4.66 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1899123, upper bound: 1.2084502
time: 5.58 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 25.39 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.39
Output dim: 2, lower bound: -1.1899222, upper bound: 1.1899224
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.39
Output dim: 2, lower bound: -1.1899222, upper bound: 1.1899245
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.39
Output dim: 2, lower bound: -1.1899222, upper bound: 1.1960717
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.39
Output dim: 2, lower bound: -1.1899222, upper bound: 1.1960739
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.39
Output dim: 2, lower bound: -1.2022181, upper bound: 1.1899101
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.39
Output dim: 2, lower bound: -1.2022181, upper bound: 1.1899122
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.39
Output dim: 2, lower bound: -1.2022181, upper bound: 1.1960616
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.39
Output dim: 2, lower bound: -1.2022180, upper bound: 1.1960618
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.39
Output dim: 2, lower bound: -1.1899124, upper bound: 1.2022200
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.39
Output dim: 2, lower bound: -1.1899120, upper bound: 1.2022185
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.39
Output dim: 2, lower bound: -1.1899124, upper bound: 1.2084521
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.39
Output dim: 2, lower bound: -1.1899124, upper bound: 1.2084526
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.39
Output dim: 2, lower bound: -1.1899123, upper bound: 1.2022137
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.39
Output dim: 2, lower bound: -1.1899120, upper bound: 1.2022157
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.39
Output dim: 2, lower bound: -1.1899120, upper bound: 1.2084477
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.39
Output dim: 2, lower bound: -1.1899123, upper bound: 1.2084502

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -10.8084745, -8.2390308, -10.8084745, -8.2390308, -2.3498845, 2.3498845
1: -3.3329079, -0.9067831, -3.3329079, -0.9067831, -2.3996134, 2.3996134
2: 1.6565111, 3.6167736, 1.6565111, 3.6167736, -1.9601054, 1.9601057
3: -7.2572231, -5.2426581, -7.2572231, -5.2426581, -2.0145650, 2.0145650
4: -2.3337822, -0.4268136, -2.3337822, -0.4268136, -1.6893982, 1.6893981
5: -4.6163568, -2.7548733, -4.6163568, -2.7548733, -1.8614836, 1.8614836
6: -4.6937370, -2.2057319, -4.6937370, -2.2057319, -2.3219509, 2.3219504
7: -8.7166748, -6.8750615, -8.7166748, -6.8750615, -1.6088243, 1.6088241
8: -4.6383018, -2.4350104, -4.6383018, -2.4350104, -2.1426291, 2.1426289
9: -12.0737743, -9.7603779, -12.0737743, -9.7603779, -1.8046293, 1.8046293

Time for backsubstitution: 14.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1792549, upper bound: 1.1892982
time: 4.89 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1899209, upper bound: 1.1899166
time: 5.63 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -10.8084745, -8.2390308, -10.8637476, -8.2271538, -2.3613868, 2.3946619
1: -3.3329079, -0.9067831, -3.3551629, -0.8717890, -2.4341693, 2.4199357
2: 1.6565111, 3.6167736, 1.6043967, 3.6315553, -1.9749384, 1.9930921
3: -7.2572231, -5.2426581, -7.2652884, -5.2359872, -2.0212359, 2.0226302
4: -2.3337822, -0.4268136, -2.3649111, -0.3222642, -1.7200704, 1.7212043
5: -4.6163568, -2.7548733, -4.6255507, -2.7302155, -1.8861413, 1.8706775
6: -4.6937370, -2.2057319, -4.7280998, -2.1204927, -2.3534312, 2.3523800
7: -8.7166748, -6.8750615, -8.7214088, -6.8390384, -1.6297479, 1.6148307
8: -4.6383018, -2.4350104, -4.6898065, -2.4254057, -2.1530471, 2.1930783
9: -12.0737743, -9.7603779, -12.1676559, -9.7390251, -1.8280296, 1.8332801

Time for backsubstitution: 14.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1792561, upper bound: 1.1892974
time: 5.57 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1899209, upper bound: 1.1899180
time: 6.44 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -10.8637476, -8.2271538, -10.8084745, -8.2390308, -2.3946619, 2.3613868
1: -3.3551629, -0.8717890, -3.3329079, -0.9067831, -2.4199357, 2.4341693
2: 1.6043967, 3.6315553, 1.6565111, 3.6167736, -1.9930921, 1.9749386
3: -7.2652884, -5.2359872, -7.2572231, -5.2426581, -2.0226302, 2.0212359
4: -2.3649111, -0.3222642, -2.3337822, -0.4268136, -1.7212044, 1.7200704
5: -4.6255507, -2.7302155, -4.6163568, -2.7548733, -1.8706775, 1.8861413
6: -4.7280998, -2.1204927, -4.6937370, -2.2057319, -2.3523803, 2.3534312
7: -8.7214088, -6.8390384, -8.7166748, -6.8750615, -1.6148305, 1.6297477
8: -4.6898065, -2.4254057, -4.6383018, -2.4350104, -2.1930785, 2.1530471
9: -12.1676559, -9.7390251, -12.0737743, -9.7603779, -1.8332801, 1.8280296

Time for backsubstitution: 14.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1792497, upper bound: 1.1954246
time: 4.77 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1899157, upper bound: 1.1960666
time: 6.44 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -10.8637476, -8.2271538, -10.8637476, -8.2271538, -2.3937750, 2.3937759
1: -3.3551629, -0.8717890, -3.3551629, -0.8717890, -2.4377990, 2.4377990
2: 1.6043967, 3.6315553, 1.6043967, 3.6315553, -1.9969721, 1.9969723
3: -7.2652884, -5.2359872, -7.2652884, -5.2359872, -2.0293012, 2.0293012
4: -2.3649111, -0.3222642, -2.3649111, -0.3222642, -1.7519670, 1.7519670
5: -4.6255507, -2.7302155, -4.6255507, -2.7302155, -1.8953352, 1.8953352
6: -4.7280998, -2.1204927, -4.7280998, -2.1204927, -2.3841872, 2.3841872
7: -8.7214088, -6.8390384, -8.7214088, -6.8390384, -1.6382978, 1.6382978
8: -4.6898065, -2.4254057, -4.6898065, -2.4254057, -2.1991720, 2.1991718
9: -12.1676559, -9.7390251, -12.1676559, -9.7390251, -1.8580916, 1.8580918

Time for backsubstitution: 14.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1792496, upper bound: 1.1954255
time: 5.28 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1899157, upper bound: 1.1960647
time: 6.45 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -10.8084745, -8.2390308, -10.8342667, -8.2325087, -2.3603840, 2.3802991
1: -3.3329079, -0.9067831, -3.3540924, -0.8299019, -2.4506450, 2.4202094
2: 1.6565111, 3.6167736, 1.5641996, 3.6438277, -1.9873166, 1.9970789
3: -7.2572231, -5.2426581, -7.3152289, -5.2275262, -2.0296969, 2.0725708
4: -2.3337822, -0.4268136, -2.3626270, -0.4219837, -1.6949682, 1.7194332
5: -4.6163568, -2.7548733, -4.6311669, -2.6947558, -1.9169679, 1.8762937
6: -4.6937370, -2.2057319, -4.7211199, -2.1557031, -2.3587017, 2.3523824
7: -8.7166748, -6.8750615, -8.7282162, -6.8550620, -1.6287546, 1.6222328
8: -4.6383018, -2.4350104, -4.6630821, -2.3795116, -2.1991916, 2.1640472
9: -12.0737743, -9.7603779, -12.0843811, -9.7559776, -1.8090744, 1.8171544

Time for backsubstitution: 14.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1916708, upper bound: 1.1892872
time: 5.36 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2022162, upper bound: 1.1899058
time: 5.40 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -10.8084745, -8.2390308, -10.8896751, -8.2206631, -2.3718748, 2.4143262
1: -3.3329079, -0.9067831, -3.3763201, -0.7948638, -2.4547458, 2.4406114
2: 1.6565111, 3.6167736, 1.5120236, 3.6585867, -2.0020757, 2.0048656
3: -7.2572231, -5.2426581, -7.3235197, -5.2208347, -2.0363884, 2.0808616
4: -2.3337822, -0.4268136, -2.3939757, -0.3174410, -1.7255344, 1.7514024
5: -4.6163568, -2.7548733, -4.6403561, -2.6699915, -1.9213104, 1.8854828
6: -4.6937370, -2.2057319, -4.7553735, -2.0704236, -2.3625879, 2.3827670
7: -8.7166748, -6.8750615, -8.7329617, -6.8189249, -1.6475847, 1.6282346
8: -4.6383018, -2.4350104, -4.7146053, -2.3699040, -2.2096062, 2.2146549
9: -12.0737743, -9.7603779, -12.1783562, -9.7346020, -1.8324938, 1.8413062

Time for backsubstitution: 14.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1916708, upper bound: 1.1892879
time: 5.25 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2022162, upper bound: 1.1899059
time: 6.94 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -10.8637476, -8.2271538, -10.8342667, -8.2325087, -2.4006014, 2.3918018
1: -3.3551629, -0.8717890, -3.3540924, -0.8299019, -2.4711099, 2.4547653
2: 1.6043967, 3.6315553, 1.5641996, 3.6438277, -2.0221298, 2.0119035
3: -7.2652884, -5.2359872, -7.3152289, -5.2275262, -2.0377622, 2.0792418
4: -2.3649111, -0.3222642, -2.3626270, -0.4219837, -1.7267745, 1.7365305
5: -4.6255507, -2.7302155, -4.6311669, -2.6947558, -1.9264522, 1.9009514
6: -4.7280998, -2.1204927, -4.7211199, -2.1557031, -2.3894582, 2.3838897
7: -8.7214088, -6.8390384, -8.7282162, -6.8550620, -1.6347609, 1.6326344
8: -4.6898065, -2.4254057, -4.6630821, -2.3795116, -2.2496409, 2.1744652
9: -12.1676559, -9.7390251, -12.0843811, -9.7559776, -1.8377516, 1.8405542

Time for backsubstitution: 14.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1916657, upper bound: 1.1954140
time: 5.15 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2022110, upper bound: 1.1960547
time: 5.47 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -10.8637476, -8.2271538, -10.8896751, -8.2206631, -2.4042635, 2.4242921
1: -3.3551629, -0.8717890, -3.3763201, -0.7948638, -2.4764605, 2.4585071
2: 1.6043967, 3.6315553, 1.5120236, 3.6585867, -2.0259004, 2.0201275
3: -7.2652884, -5.2359872, -7.3235197, -5.2208347, -2.0444536, 2.0875325
4: -2.3649111, -0.3222642, -2.3939757, -0.3174410, -1.7574306, 1.7685964
5: -4.6255507, -2.7302155, -4.6403561, -2.6699915, -1.9349546, 1.9101405
6: -4.7280998, -2.1204927, -4.7553735, -2.0704236, -2.3933439, 2.4146044
7: -8.7214088, -6.8390384, -8.7329617, -6.8189249, -1.6561205, 1.6411541
8: -4.6898065, -2.4254057, -4.7146053, -2.3699040, -2.2557387, 2.2204332
9: -12.1676559, -9.7390251, -12.1783562, -9.7346020, -1.8625724, 1.8661095

Time for backsubstitution: 14.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1916657, upper bound: 1.1954149
time: 5.29 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2022110, upper bound: 1.1960543
time: 7.70 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -10.8342667, -8.2325087, -10.8084745, -8.2390308, -2.3802991, 2.3603845
1: -3.3540924, -0.8299019, -3.3329079, -0.9067831, -2.4202094, 2.4506450
2: 1.5641996, 3.6438277, 1.6565111, 3.6167736, -1.9970789, 1.9873166
3: -7.3152289, -5.2275262, -7.2572231, -5.2426581, -2.0725708, 2.0296969
4: -2.3626270, -0.4219837, -2.3337822, -0.4268136, -1.7194332, 1.6949685
5: -4.6311669, -2.6947558, -4.6163568, -2.7548733, -1.8762937, 1.9169681
6: -4.7211199, -2.1557031, -4.6937370, -2.2057319, -2.3523827, 2.3587017
7: -8.7282162, -6.8550620, -8.7166748, -6.8750615, -1.6222329, 1.6287545
8: -4.6630821, -2.3795116, -4.6383018, -2.4350104, -2.1640472, 2.1991916
9: -12.0843811, -9.7559776, -12.0737743, -9.7603779, -1.8171539, 1.8090742

Time for backsubstitution: 15.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1792446, upper bound: 1.2018572
time: 5.14 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1899105, upper bound: 1.2022118
time: 5.38 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -10.8342667, -8.2325087, -10.8637476, -8.2271538, -2.3918023, 2.4006016
1: -3.3540924, -0.8299019, -3.3551629, -0.8717890, -2.4547653, 2.4711099
2: 1.5641996, 3.6438277, 1.6043967, 3.6315553, -2.0119033, 2.0221300
3: -7.3152289, -5.2275262, -7.2652884, -5.2359872, -2.0792418, 2.0377622
4: -2.3626270, -0.4219837, -2.3649111, -0.3222642, -1.7365308, 1.7267747
5: -4.6311669, -2.6947558, -4.6255507, -2.7302155, -1.9009514, 1.9264519
6: -4.7211199, -2.1557031, -4.7280998, -2.1204927, -2.3838897, 2.3894582
7: -8.7282162, -6.8550620, -8.7214088, -6.8390384, -1.6326346, 1.6347613
8: -4.6630821, -2.3795116, -4.6898065, -2.4254057, -2.1744652, 2.2496409
9: -12.0843811, -9.7559776, -12.1676559, -9.7390251, -1.8405547, 1.8377515

Time for backsubstitution: 14.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1792446, upper bound: 1.2018573
time: 5.26 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1899105, upper bound: 1.2022117
time: 7.10 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -10.8896751, -8.2206631, -10.8084745, -8.2390308, -2.4143262, 2.3718743
1: -3.3763201, -0.7948638, -3.3329079, -0.9067831, -2.4406114, 2.4547458
2: 1.5120236, 3.6585867, 1.6565111, 3.6167736, -2.0048656, 2.0020757
3: -7.3235197, -5.2208347, -7.2572231, -5.2426581, -2.0808616, 2.0363884
4: -2.3939757, -0.3174410, -2.3337822, -0.4268136, -1.7514025, 1.7255342
5: -4.6403561, -2.6699915, -4.6163568, -2.7548733, -1.8854828, 1.9213104
6: -4.7553735, -2.0704236, -4.6937370, -2.2057319, -2.3827667, 2.3625879
7: -8.7329617, -6.8189249, -8.7166748, -6.8750615, -1.6282349, 1.6475847
8: -4.7146053, -2.3699040, -4.6383018, -2.4350104, -2.2146549, 2.2096057
9: -12.1783562, -9.7346020, -12.0737743, -9.7603779, -1.8413062, 1.8324933

Time for backsubstitution: 15.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1792395, upper bound: 1.2080619
time: 5.09 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1899053, upper bound: 1.2084450
time: 7.37 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -10.8896751, -8.2206631, -10.8637476, -8.2271538, -2.4242926, 2.4042635
1: -3.3763201, -0.7948638, -3.3551629, -0.8717890, -2.4585071, 2.4764605
2: 1.5120236, 3.6585867, 1.6043967, 3.6315553, -2.0201273, 2.0259004
3: -7.3235197, -5.2208347, -7.2652884, -5.2359872, -2.0875325, 2.0444536
4: -2.3939757, -0.3174410, -2.3649111, -0.3222642, -1.7685966, 1.7574306
5: -4.6403561, -2.6699915, -4.6255507, -2.7302155, -1.9101405, 1.9349554
6: -4.7553735, -2.0704236, -4.7280998, -2.1204927, -2.4146047, 2.3933439
7: -8.7329617, -6.8189249, -8.7214088, -6.8390384, -1.6411541, 1.6561208
8: -4.7146053, -2.3699040, -4.6898065, -2.4254057, -2.2204332, 2.2557390
9: -12.1783562, -9.7346020, -12.1676559, -9.7390251, -1.8661096, 1.8625722

Time for backsubstitution: 14.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1792395, upper bound: 1.2080627
time: 5.25 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1899053, upper bound: 1.2084481
time: 7.20 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -10.8342667, -8.2325087, -10.8342667, -8.2325087, -2.4029741, 2.4029737
1: -3.3540924, -0.8299019, -3.3540924, -0.8299019, -2.4594812, 2.4594812
2: 1.5641996, 3.6438277, 1.5641996, 3.6438277, -2.0261171, 2.0261168
3: -7.3152289, -5.2275262, -7.3152289, -5.2275262, -2.0877028, 2.0877028
4: -2.3626270, -0.4219837, -2.3626270, -0.4219837, -1.7207114, 1.7207115
5: -4.6311669, -2.6947558, -4.6311669, -2.6947558, -1.9152741, 1.9152741
6: -4.7211199, -2.1557031, -4.7211199, -2.1557031, -2.3885946, 2.3885942
7: -8.7282162, -6.8550620, -8.7282162, -6.8550620, -1.6452711, 1.6452709
8: -4.6630821, -2.3795116, -4.6630821, -2.3795116, -2.2162776, 2.2162776
9: -12.0843811, -9.7559776, -12.0843811, -9.7559776, -1.8303888, 1.8303890

Time for backsubstitution: 15.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1792446, upper bound: 1.2018558
time: 5.41 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1899107, upper bound: 1.2022087
time: 5.65 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -10.8342667, -8.2325087, -10.8896751, -8.2206631, -2.4144640, 2.4243419
1: -3.3540924, -0.8299019, -3.3763201, -0.7948638, -2.4768944, 2.4799109
2: 1.5641996, 3.6438277, 1.5120236, 3.6585867, -2.0409434, 2.0339036
3: -7.3152289, -5.2275262, -7.3235197, -5.2208347, -2.0943942, 2.0959935
4: -2.3626270, -0.4219837, -2.3939757, -0.3174410, -1.7434125, 1.7528315
5: -4.6311669, -2.6947558, -4.6403561, -2.6699915, -1.9374671, 1.9247737
6: -4.7211199, -2.1557031, -4.7553735, -2.0704236, -2.3967080, 2.4189866
7: -8.7282162, -6.8550620, -8.7329617, -6.8189249, -1.6542583, 1.6512541
8: -4.6630821, -2.3795116, -4.7146053, -2.3699040, -2.2266917, 2.2668850
9: -12.0843811, -9.7559776, -12.1783562, -9.7346020, -1.8537681, 1.8484421

Time for backsubstitution: 14.95 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=1.9854326248168945
rel_dist={2: [-1.2085888365066806, 1.2085886275699598]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 5733
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4625

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9483193, upper bound: 0.9355519
time: 5.05 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9483198, upper bound: 0.9483151
time: 5.38 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 10.63 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 10.63
Output dim: 2, lower bound: -0.9483193, upper bound: 0.9355519
IS_A2, status: Status.UNKNOWN, split count: 1, time: 10.63
Output dim: 2, lower bound: -0.9483198, upper bound: 0.9483151

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -10.8091249, -8.2323895, -10.8108559, -8.2310667, -2.1431274, 2.1421008
1: -3.3391936, -0.9065280, -3.3504086, -0.9043145, -2.2216730, 2.2313533
2: 1.6559384, 3.6253042, 1.6547616, 3.6405287, -1.8675251, 1.8535683
3: -7.2581921, -5.2406983, -7.2597189, -5.2326798, -1.9312358, 1.9247880
4: -2.3505065, -0.4264469, -2.3514392, -0.4242229, -1.5469065, 1.5450850
5: -4.6192904, -2.7543774, -4.6279707, -2.7532918, -1.7735519, 1.7810264
6: -4.7066774, -2.2050157, -4.7157536, -2.2042937, -2.1578889, 2.1675305
7: -8.7201481, -6.8739891, -8.7202091, -6.8711286, -1.4938424, 1.4906883
8: -4.6400385, -2.4291024, -4.6486425, -2.4285851, -1.9597683, 1.9686389
9: -12.0747089, -9.7458115, -12.0749950, -9.7447462, -1.5819683, 1.5809789

Time for backsubstitution: 14.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4625

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9355534, upper bound: 0.9355514
time: 5.61 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9355535, upper bound: 0.9355507
time: 5.83 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -10.8347559, -8.2258587, -10.8108482, -8.2310753, -2.1733379, 2.1717663
1: -3.3603764, -0.8297083, -3.3503861, -0.9043264, -2.2515898, 2.2758207
2: 1.5636322, 3.6523519, 1.6547675, 3.6404538, -1.9006627, 1.8829143
3: -7.3161755, -5.2255697, -7.2597136, -5.2327118, -1.9906330, 1.9435813
4: -2.3791616, -0.4216104, -2.3514352, -0.4242320, -1.5768292, 1.5579345
5: -4.6340709, -2.6942589, -4.6279249, -2.7532957, -1.7900777, 1.8215969
6: -4.7340093, -2.1549859, -4.7157164, -2.2042964, -2.2074871, 2.2005100
7: -8.7316866, -6.8541307, -8.7202091, -6.8711395, -1.5089247, 1.5104966
8: -4.6648216, -2.3736062, -4.6485977, -2.4285889, -1.9811883, 2.0251496
9: -12.0852623, -9.7414169, -12.0749931, -9.7447491, -1.5944114, 1.5964034

Time for backsubstitution: 15.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4625

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9355535, upper bound: 0.9483177
time: 5.91 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9355527, upper bound: 0.9483202
time: 6.25 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 27.39 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 27.39
Output dim: 2, lower bound: -0.9355534, upper bound: 0.9355514
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 27.39
Output dim: 2, lower bound: -0.9355535, upper bound: 0.9355507
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 27.39
Output dim: 2, lower bound: -0.9355535, upper bound: 0.9483177
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 27.39
Output dim: 2, lower bound: -0.9355527, upper bound: 0.9483202

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -10.8091249, -8.2323895, -10.8091249, -8.2323895, -2.1386523, 2.1386533
1: -3.3391936, -0.9065280, -3.3391936, -0.9065280, -2.2192049, 2.2192049
2: 1.6559384, 3.6253042, 1.6559384, 3.6253042, -1.8525171, 1.8525174
3: -7.2581921, -5.2406983, -7.2581921, -5.2406983, -1.9233689, 1.9233689
4: -2.3505065, -0.4264469, -2.3505065, -0.4264469, -1.5434717, 1.5434716
5: -4.6192904, -2.7543774, -4.6192904, -2.7543774, -1.7725086, 1.7725086
6: -4.7066774, -2.2050157, -4.7066774, -2.2050157, -2.1564193, 2.1564190
7: -8.7201481, -6.8739891, -8.7201481, -6.8739891, -1.4885042, 1.4885037
8: -4.6400385, -2.4291024, -4.6400385, -2.4291024, -1.9592333, 1.9592333
9: -12.0747089, -9.7458115, -12.0747089, -9.7458115, -1.5795460, 1.5795460

Time for backsubstitution: 15.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5733
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5733

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9355264, upper bound: 0.9291166
time: 5.06 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9355248, upper bound: 0.9355168
time: 5.05 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -10.8091249, -8.2323895, -10.8333969, -8.2258663, -2.1491528, 2.1673894
1: -3.3391936, -0.9065280, -3.3603547, -0.8302064, -2.2633338, 2.2397695
2: 1.6559384, 3.6253042, 1.5636967, 3.6522465, -1.8811269, 1.8853993
3: -7.2581921, -5.2406983, -7.3159356, -5.2256517, -1.9415269, 1.9824722
4: -2.3505065, -0.4264469, -2.3782504, -0.4216270, -1.5490271, 1.5724838
5: -4.6192904, -2.7543774, -4.6338038, -2.6943340, -1.8129320, 1.7883735
6: -4.7066774, -2.2050157, -4.7336979, -2.1549873, -2.1893368, 2.1863625
7: -8.7201481, -6.8739891, -8.7316875, -6.8552938, -1.5074573, 1.5014025
8: -4.6400385, -2.4291024, -4.6645169, -2.3736067, -2.0156674, 1.9803541
9: -12.0747089, -9.7458115, -12.0847664, -9.7414303, -1.5839310, 1.5914650

Time for backsubstitution: 15.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5733
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5733

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9355264, upper bound: 0.9291170
time: 5.84 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9355248, upper bound: 0.9355166
time: 6.25 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -10.8333969, -8.2258663, -10.8091249, -8.2323895, -2.1673894, 2.1491523
1: -3.3603547, -0.8302064, -3.3391936, -0.9065280, -2.2397695, 2.2633338
2: 1.5636967, 3.6522465, 1.6559384, 3.6253042, -1.8853991, 1.8811269
3: -7.3159356, -5.2256517, -7.2581921, -5.2406983, -1.9824719, 1.9415269
4: -2.3782504, -0.4216270, -2.3505065, -0.4264469, -1.5724834, 1.5490272
5: -4.6338038, -2.6943340, -4.6192904, -2.7543774, -1.7883735, 1.8129323
6: -4.7336979, -2.1549873, -4.7066774, -2.2050157, -2.1863627, 2.1893370
7: -8.7316875, -6.8552938, -8.7201481, -6.8739891, -1.5014026, 1.5074570
8: -4.6645169, -2.3736067, -4.6400385, -2.4291024, -1.9803543, 2.0156674
9: -12.0847664, -9.7414303, -12.0747089, -9.7458115, -1.5914650, 1.5839307

Time for backsubstitution: 14.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5733
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5733

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9355208, upper bound: 0.9418782
time: 5.46 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9355199, upper bound: 0.9482814
time: 5.24 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -10.8349228, -8.2258577, -10.8349228, -8.2258577, -2.1893935, 2.1893930
1: -3.3603773, -0.8296475, -3.3603773, -0.8296475, -2.2709193, 2.2709193
2: 1.5636239, 3.6523647, 1.5636239, 3.6523647, -1.9147067, 1.9147067
3: -7.3162060, -5.2255607, -7.3162060, -5.2255607, -1.9676185, 1.9676182
4: -2.3792734, -0.4216075, -2.3792734, -0.4216075, -1.5724447, 1.5724446
5: -4.6341038, -2.6942487, -4.6341038, -2.6942487, -1.8135900, 1.8135903
6: -4.7340474, -2.1549864, -4.7340474, -2.1549864, -2.2158489, 2.2158494
7: -8.7316875, -6.8539867, -8.7316875, -6.8539867, -1.5231943, 1.5231942
8: -4.6648607, -2.3736062, -4.6648607, -2.3736062, -2.0321093, 2.0321090
9: -12.0853224, -9.7414150, -12.0853224, -9.7414150, -1.6039038, 1.6039042

Time for backsubstitution: 15.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5733
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5733

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9355208, upper bound: 0.9418802
time: 5.71 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9355198, upper bound: 0.9482835
time: 5.92 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 26.94 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 26.94
Output dim: 2, lower bound: -0.9355264, upper bound: 0.9291166
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.94
Output dim: 2, lower bound: -0.9355248, upper bound: 0.9355168
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.94
Output dim: 2, lower bound: -0.9355264, upper bound: 0.9291170
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.94
Output dim: 2, lower bound: -0.9355248, upper bound: 0.9355166
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 26.94
Output dim: 2, lower bound: -0.9355208, upper bound: 0.9418782
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.94
Output dim: 2, lower bound: -0.9355199, upper bound: 0.9482814
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.94
Output dim: 2, lower bound: -0.9355208, upper bound: 0.9418802
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.94
Output dim: 2, lower bound: -0.9355198, upper bound: 0.9482835

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -10.8084745, -8.2390308, -10.8091249, -8.2323895, -2.1381335, 2.1320624
1: -3.3329079, -0.9067831, -3.3391936, -0.9065280, -2.2117367, 2.2183499
2: 1.6565111, 3.6167736, 1.6559384, 3.6253042, -1.8519287, 1.8438377
3: -7.2572231, -5.2426581, -7.2581921, -5.2406983, -1.9206223, 1.9189732
4: -2.3337822, -0.4268136, -2.3505065, -0.4264469, -1.5266598, 1.5428531
5: -4.6163568, -2.7548733, -4.6192904, -2.7543774, -1.7669530, 1.7705779
6: -4.6937370, -2.2057319, -4.7066774, -2.2050157, -2.1426578, 2.1559069
7: -8.7166748, -6.8750615, -8.7201481, -6.8739891, -1.4833548, 1.4860365
8: -4.6383018, -2.4350104, -4.6400385, -2.4291024, -1.9547114, 1.9506145
9: -12.0737743, -9.7603779, -12.0747089, -9.7458115, -1.5778780, 1.5653801

Time for backsubstitution: 15.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5733

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9291245, upper bound: 0.9291223
time: 5.05 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9291237, upper bound: 0.9291218
time: 6.27 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -10.8637476, -8.2271538, -10.8091259, -8.2323999, -2.1805525, 2.1452150
1: -3.3551629, -0.8717890, -3.3391817, -0.9065294, -2.2404017, 2.2528355
2: 1.6043967, 3.6315553, 1.6559397, 3.6252859, -1.8810821, 1.8646848
3: -7.2652884, -5.2359872, -7.2581897, -5.2407041, -1.9467449, 1.9269266
4: -2.3649111, -0.3222642, -2.3504672, -0.4264493, -1.5631142, 1.5697219
5: -4.6255507, -2.7302155, -4.6192842, -2.7543781, -1.8026791, 1.7945499
6: -4.7280998, -2.1204927, -4.7066541, -2.2050178, -2.1764627, 2.1835928
7: -8.7214088, -6.8390384, -8.7201366, -6.8739915, -1.5028059, 1.5052063
8: -4.6898065, -2.4254057, -4.6400347, -2.4291179, -2.0051441, 1.9889874
9: -12.1676559, -9.7390251, -12.0747070, -9.7458525, -1.6026347, 1.6008568

Time for backsubstitution: 15.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5733

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9291245, upper bound: 0.9355233
time: 5.17 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9291245, upper bound: 0.9355230
time: 6.04 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -10.8084745, -8.2390308, -10.8333969, -8.2258663, -2.1486320, 2.1607990
1: -3.3329079, -0.9067831, -3.3603547, -0.8302064, -2.2558813, 2.2389140
2: 1.6565111, 3.6167736, 1.5636967, 3.6522465, -1.8805389, 1.8767073
3: -7.2572231, -5.2426581, -7.3159356, -5.2256517, -1.9387808, 1.9780767
4: -2.3337822, -0.4268136, -2.3782504, -0.4216270, -1.5322157, 1.5718653
5: -4.6163568, -2.7548733, -4.6338038, -2.6943340, -1.8073750, 1.7864432
6: -4.6937370, -2.2057319, -4.7336979, -2.1549873, -2.1755829, 2.1858509
7: -8.7166748, -6.8750615, -8.7316875, -6.8552938, -1.5023079, 1.4989313
8: -4.6383018, -2.4350104, -4.6645169, -2.3736067, -2.0111456, 1.9717352
9: -12.0737743, -9.7603779, -12.0847664, -9.7414303, -1.5822630, 1.5772991

Time for backsubstitution: 15.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5733

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9418801, upper bound: 0.9291193
time: 6.07 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9418812, upper bound: 0.9291166
time: 6.39 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -10.8637476, -8.2271538, -10.8333950, -8.2258778, -2.1864858, 2.1739516
1: -3.3551629, -0.8717890, -3.3603437, -0.8302081, -2.2789869, 2.2734528
2: 1.6043967, 3.6315553, 1.5636970, 3.6522295, -1.9098449, 1.8943083
3: -7.2652884, -5.2359872, -7.3159332, -5.2256570, -1.9649019, 1.9841981
4: -2.3649111, -0.3222642, -2.3782105, -0.4216280, -1.5686691, 1.5853753
5: -4.6255507, -2.7302155, -4.6337967, -2.6943350, -1.8213692, 1.8104148
6: -4.7280998, -2.1204927, -4.7336750, -2.1549883, -2.2096915, 2.2135849
7: -8.7214088, -6.8390384, -8.7316761, -6.8552966, -1.5202770, 1.5080774
8: -4.6898065, -2.4254057, -4.6645136, -2.3736203, -2.0615788, 2.0101085
9: -12.1676559, -9.7390251, -12.0847654, -9.7414722, -1.6070399, 1.6127758

Time for backsubstitution: 15.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5733

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9418801, upper bound: 0.9355176
time: 6.46 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9418812, upper bound: 0.9355176
time: 6.50 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -10.8327427, -8.2325172, -10.8091249, -8.2323895, -2.1668677, 2.1425567
1: -3.3540702, -0.8304625, -3.3391936, -0.9065280, -2.2323117, 2.2624855
2: 1.5642705, 3.6437109, 1.6559384, 3.6253042, -1.8848131, 1.8724394
3: -7.3149586, -5.2276173, -7.2581921, -5.2406983, -1.9797182, 1.9371352
4: -2.3616037, -0.4220018, -2.3505065, -0.4264469, -1.5556613, 1.5483892
5: -4.6308661, -2.6948409, -4.6192904, -2.7543774, -1.7828059, 1.8109915
6: -4.7207699, -2.1557038, -4.7066774, -2.2050157, -2.1726537, 2.1888225
7: -8.7282143, -6.8563685, -8.7201481, -6.8739891, -1.4962504, 1.5049994
8: -4.6627421, -2.3795121, -4.6400385, -2.4291024, -1.9758387, 2.0070572
9: -12.0838232, -9.7559948, -12.0747089, -9.7458115, -1.5898032, 1.5697756

Time for backsubstitution: 15.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5733

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9291189, upper bound: 0.9418780
time: 5.12 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9291189, upper bound: 0.9418780
time: 5.39 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -10.8881435, -8.2206697, -10.8091259, -8.2323999, -2.1991301, 2.1556969
1: -3.3762975, -0.7954248, -3.3391817, -0.9065294, -2.2610884, 2.2665815
2: 1.5120947, 3.6584711, 1.6559397, 3.6252859, -1.8925934, 1.8933263
3: -7.3232489, -5.2209263, -7.2581897, -5.2407041, -1.9924600, 1.9451046
4: -2.3929527, -0.3174589, -2.3504672, -0.4264493, -1.5922780, 1.5751517
5: -4.6400547, -2.6700764, -4.6192842, -2.7543781, -1.8186331, 1.8153293
6: -4.7550316, -2.0704255, -4.7066541, -2.2050178, -2.2064142, 2.1927013
7: -8.7329617, -6.8202329, -8.7201366, -6.8739915, -1.5056465, 1.5225455
8: -4.7142630, -2.3699050, -4.6400347, -2.4291179, -2.0264292, 2.0454359
9: -12.1778002, -9.7346172, -12.0747070, -9.7458525, -1.6102252, 1.6052470

Time for backsubstitution: 15.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5733

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9291182, upper bound: 0.9482823
time: 5.40 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9291182, upper bound: 0.9482821
time: 5.49 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -10.8342667, -8.2325087, -10.8349228, -8.2258577, -2.1888537, 2.1827979
1: -3.3540924, -0.8299019, -3.3603773, -0.8296475, -2.2634654, 2.2700744
2: 1.5641996, 3.6438277, 1.5636239, 3.6523647, -1.9141207, 1.9060080
3: -7.3152289, -5.2275262, -7.3162060, -5.2255607, -1.9648638, 1.9632266
4: -2.3626270, -0.4219837, -2.3792734, -0.4216075, -1.5556815, 1.5718151
5: -4.6311669, -2.6947558, -4.6341038, -2.6942487, -1.8080230, 1.8116646
6: -4.7211199, -2.1557031, -4.7340474, -2.1549864, -2.2021217, 2.2153368
7: -8.7282162, -6.8550620, -8.7316875, -6.8539867, -1.5180426, 1.5207311
8: -4.6630821, -2.3795116, -4.6648607, -2.3736062, -2.0275936, 2.0234988
9: -12.0843811, -9.7559776, -12.0853224, -9.7414150, -1.6022458, 1.5897405

Time for backsubstitution: 15.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5733

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9291189, upper bound: 0.9418808
time: 5.95 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9291182, upper bound: 0.9418790
time: 5.29 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -10.8896751, -8.2206631, -10.8349218, -8.2258701, -2.2102308, 2.1959367
1: -3.3763201, -0.7948638, -3.3603687, -0.8296471, -2.2922702, 2.2891037
2: 1.5120236, 3.6585867, 1.5636247, 3.6523457, -1.9219007, 1.9236152
3: -7.3235197, -5.2208347, -7.3162041, -5.2255645, -1.9912491, 1.9711969
4: -2.3939757, -0.3174410, -2.3792336, -0.4216094, -1.5924487, 1.5930188
5: -4.6403561, -2.6699915, -4.6340971, -2.6942496, -1.8377090, 1.8316379
6: -4.7553735, -2.0704236, -4.7340236, -2.1549881, -2.2358880, 2.2268410
7: -8.7329617, -6.8189249, -8.7316761, -6.8539891, -1.5274379, 1.5297167
8: -4.7146053, -2.3699040, -4.6648555, -2.3736198, -2.0781851, 2.0618761
9: -12.1783562, -9.7346020, -12.0853195, -9.7414570, -1.6177881, 1.6251733

Time for backsubstitution: 15.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5733

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9291188, upper bound: 0.9482823
time: 5.98 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9291189, upper bound: 0.9482818
time: 6.44 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 27.67 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.67
Output dim: 2, lower bound: -0.9291245, upper bound: 0.9291223
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.67
Output dim: 2, lower bound: -0.9291237, upper bound: 0.9291218
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.67
Output dim: 2, lower bound: -0.9291245, upper bound: 0.9355233
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.67
Output dim: 2, lower bound: -0.9291245, upper bound: 0.9355230
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.67
Output dim: 2, lower bound: -0.9418801, upper bound: 0.9291193
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.67
Output dim: 2, lower bound: -0.9418812, upper bound: 0.9291166
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.67
Output dim: 2, lower bound: -0.9418801, upper bound: 0.9355176
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.67
Output dim: 2, lower bound: -0.9418812, upper bound: 0.9355176
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.67
Output dim: 2, lower bound: -0.9291189, upper bound: 0.9418780
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.67
Output dim: 2, lower bound: -0.9291189, upper bound: 0.9418780
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.67
Output dim: 2, lower bound: -0.9291182, upper bound: 0.9482823
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.67
Output dim: 2, lower bound: -0.9291182, upper bound: 0.9482821
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.67
Output dim: 2, lower bound: -0.9291189, upper bound: 0.9418808
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.67
Output dim: 2, lower bound: -0.9291182, upper bound: 0.9418790
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.67
Output dim: 2, lower bound: -0.9291188, upper bound: 0.9482823
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.67
Output dim: 2, lower bound: -0.9291189, upper bound: 0.9482818

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -10.8084745, -8.2390308, -10.8084745, -8.2390308, -2.1315427, 2.1315427
1: -3.3329079, -0.9067831, -3.3329079, -0.9067831, -2.2108817, 2.2108817
2: 1.6565111, 3.6167736, 1.6565111, 3.6167736, -1.8432493, 1.8432493
3: -7.2572231, -5.2426581, -7.2572231, -5.2426581, -1.9162273, 1.9162269
4: -2.3337822, -0.4268136, -2.3337822, -0.4268136, -1.5260416, 1.5260417
5: -4.6163568, -2.7548733, -4.6163568, -2.7548733, -1.7650223, 1.7650225
6: -4.6937370, -2.2057319, -4.6937370, -2.2057319, -2.1421461, 2.1421456
7: -8.7166748, -6.8750615, -8.7166748, -6.8750615, -1.4808874, 1.4808874
8: -4.6383018, -2.4350104, -4.6383018, -2.4350104, -1.9460926, 1.9460926
9: -12.0737743, -9.7603779, -12.0737743, -9.7603779, -1.5637121, 1.5637121

Time for backsubstitution: 15.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9236055, upper bound: 0.9290975
time: 5.46 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9291012, upper bound: 0.9290979
time: 5.71 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -10.8084745, -8.2390308, -10.8637476, -8.2271538, -2.1430459, 2.1739640
1: -3.3329079, -0.9067831, -3.3551629, -0.8717890, -2.2453895, 2.2312040
2: 1.6565111, 3.6167736, 1.6043967, 3.6315553, -1.8580823, 1.8723969
3: -7.2572231, -5.2426581, -7.2652884, -5.2359872, -1.9241881, 1.9258187
4: -2.3337822, -0.4268136, -2.3649111, -0.3222642, -1.5528812, 1.5578480
5: -4.6163568, -2.7548733, -4.6255507, -2.7302155, -1.7890038, 1.7744818
6: -4.6937370, -2.2057319, -4.7280998, -2.1204927, -2.1698470, 2.1725752
7: -8.7166748, -6.8750615, -8.7214088, -6.8390384, -1.5000563, 1.4868939
8: -4.6383018, -2.4350104, -4.6898065, -2.4254057, -1.9565105, 1.9965420
9: -12.0737743, -9.7603779, -12.1676559, -9.7390251, -1.5871124, 1.5884147

Time for backsubstitution: 15.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9236055, upper bound: 0.9290983
time: 6.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9291012, upper bound: 0.9291007
time: 6.19 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -10.8637476, -8.2271538, -10.8084745, -8.2390308, -2.1739640, 2.1430454
1: -3.3551629, -0.8717890, -3.3329079, -0.9067831, -2.2312040, 2.2453895
2: 1.6043967, 3.6315553, 1.6565111, 3.6167736, -1.8723969, 1.8580823
3: -7.2652884, -5.2359872, -7.2572231, -5.2426581, -1.9258184, 1.9241881
4: -2.3649111, -0.3222642, -2.3337822, -0.4268136, -1.5578479, 1.5528815
5: -4.6255507, -2.7302155, -4.6163568, -2.7548733, -1.7744818, 1.7890038
6: -4.7280998, -2.1204927, -4.6937370, -2.2057319, -2.1725755, 2.1698470
7: -8.7214088, -6.8390384, -8.7166748, -6.8750615, -1.4868941, 1.5000563
8: -4.6898065, -2.4254057, -4.6383018, -2.4350104, -1.9965420, 1.9565105
9: -12.1676559, -9.7390251, -12.0737743, -9.7603779, -1.5884147, 1.5871124

Time for backsubstitution: 15.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9236024, upper bound: 0.9355001
time: 5.29 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9290982, upper bound: 0.9355005
time: 6.18 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -10.8637476, -8.2271538, -10.8637476, -8.2271538, -2.1712236, 2.1712246
1: -3.3551629, -0.8717890, -3.3551629, -0.8717890, -2.2442279, 2.2442284
2: 1.6043967, 3.6315553, 1.6043967, 3.6315553, -1.8740969, 1.8740966
3: -7.2652884, -5.2359872, -7.2652884, -5.2359872, -1.9491987, 1.9491985
4: -2.3649111, -0.3222642, -2.3649111, -0.3222642, -1.5832382, 1.5832386
5: -4.6255507, -2.7302155, -4.6255507, -2.7302155, -1.8064504, 1.8064504
6: -4.7280998, -2.1204927, -4.7280998, -2.1204927, -2.2006035, 2.2006035
7: -8.7214088, -6.8390384, -8.7214088, -6.8390384, -1.5086064, 1.5086064
8: -4.6898065, -2.4254057, -4.6898065, -2.4254057, -1.9941883, 1.9941888
9: -12.1676559, -9.7390251, -12.1676559, -9.7390251, -1.6045375, 1.6045375

Time for backsubstitution: 15.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9236024, upper bound: 0.9355017
time: 10.73 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9290982, upper bound: 0.9355011
time: 8.80 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -10.8084745, -8.2390308, -10.8327427, -8.2325172, -2.1420374, 2.1602774
1: -3.3329079, -0.9067831, -3.3540702, -0.8304625, -2.2550330, 2.2314563
2: 1.6565111, 3.6167736, 1.5642705, 3.6437109, -1.8718510, 1.8761210
3: -7.2572231, -5.2426581, -7.3149586, -5.2276173, -1.9343886, 1.9753225
4: -2.3337822, -0.4268136, -2.3616037, -0.4220018, -1.5315775, 1.5550427
5: -4.6163568, -2.7548733, -4.6308661, -2.6948409, -1.8054342, 1.7808752
6: -4.6937370, -2.2057319, -4.7207699, -2.1557038, -2.1750684, 2.1721416
7: -8.7166748, -6.8750615, -8.7282143, -6.8563685, -1.4998503, 1.4937793
8: -4.6383018, -2.4350104, -4.6627421, -2.3795121, -2.0025353, 1.9672198
9: -12.0737743, -9.7603779, -12.0838232, -9.7559948, -1.5681076, 1.5756378

Time for backsubstitution: 15.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9363567, upper bound: 0.9290926
time: 7.12 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9418545, upper bound: 0.9290950
time: 6.67 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -10.8084745, -8.2390308, -10.8881435, -8.2206697, -2.1535273, 2.1925416
1: -3.3329079, -0.9067831, -3.3762975, -0.7954248, -2.2591352, 2.2518587
2: 1.6565111, 3.6167736, 1.5120947, 3.6584711, -1.8866863, 1.8839083
3: -7.2572231, -5.2426581, -7.3232489, -5.2209263, -1.9423666, 1.9851162
4: -2.3337822, -0.4268136, -2.3929527, -0.3174589, -1.5583112, 1.5870116
5: -4.6163568, -2.7548733, -4.6400547, -2.6700764, -1.8097763, 1.7903748
6: -4.6937370, -2.2057319, -4.7550316, -2.0704255, -2.1789560, 2.2025268
7: -8.7166748, -6.8750615, -8.7329617, -6.8202329, -1.5173953, 1.4997628
8: -4.6383018, -2.4350104, -4.7142630, -2.3699050, -2.0129499, 2.0178268
9: -12.0737743, -9.7603779, -12.1778002, -9.7346172, -1.5915279, 1.5960053

Time for backsubstitution: 15.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9363567, upper bound: 0.9290929
time: 5.89 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9418545, upper bound: 0.9290923
time: 5.47 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -10.8637476, -8.2271538, -10.8327427, -8.2325172, -2.1798992, 2.1717801
1: -3.3551629, -0.8717890, -3.3540702, -0.8304625, -2.2754979, 2.2660122
2: 1.6043967, 3.6315553, 1.5642705, 3.6437109, -1.9011526, 1.8909457
3: -7.2652884, -5.2359872, -7.3149586, -5.2276173, -1.9439807, 1.9813924
4: -2.3649111, -0.3222642, -2.3616037, -0.4220018, -1.5633835, 1.5685706
5: -4.6255507, -2.7302155, -4.6308661, -2.6948409, -1.8149176, 1.8048563
6: -4.7280998, -2.1204927, -4.7207699, -2.1557038, -2.2058249, 2.1998711
7: -8.7214088, -6.8390384, -8.7282143, -6.8563685, -1.5058570, 1.5029267
8: -4.6898065, -2.4254057, -4.6627421, -2.3795121, -2.0529847, 1.9776378
9: -12.1676559, -9.7390251, -12.0838232, -9.7559948, -1.5928295, 1.5990379

Time for backsubstitution: 14.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9363536, upper bound: 0.9354951
time: 6.97 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9418515, upper bound: 0.9354949
time: 10.66 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -10.8637476, -8.2271538, -10.8881435, -8.2206697, -2.1817069, 2.2000580
1: -3.3551629, -0.8717890, -3.3762975, -0.7954248, -2.2808504, 2.2649150
2: 1.6043967, 3.6315553, 1.5120947, 3.6584711, -1.9027381, 1.8991699
3: -7.2652884, -5.2359872, -7.3232489, -5.2209263, -1.9673295, 1.9942029
4: -2.3649111, -0.3222642, -2.3929527, -0.3174589, -1.5887311, 1.6006360
5: -4.6255507, -2.7302155, -4.6400547, -2.6700764, -1.8234205, 1.8224044
6: -4.7280998, -2.1204927, -4.7550316, -2.0704255, -2.2097120, 2.2305908
7: -8.7214088, -6.8390384, -8.7329617, -6.8202329, -1.5259318, 1.5114467
8: -4.6898065, -2.4254057, -4.7142630, -2.3699050, -2.0506372, 2.0151620
9: -12.1676559, -9.7390251, -12.1778002, -9.7346172, -1.6089277, 1.6164489

Time for backsubstitution: 15.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9363549, upper bound: 0.9354952
time: 6.61 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9418515, upper bound: 0.9354958
time: 5.36 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -10.8327427, -8.2325172, -10.8084745, -8.2390308, -2.1602778, 2.1420369
1: -3.3540702, -0.8304625, -3.3329079, -0.9067831, -2.2314563, 2.2550330
2: 1.5642705, 3.6437109, 1.6565111, 3.6167736, -1.8761210, 1.8718510
3: -7.3149586, -5.2276173, -7.2572231, -5.2426581, -1.9753222, 1.9343889
4: -2.3616037, -0.4220018, -2.3337822, -0.4268136, -1.5550429, 1.5315773
5: -4.6308661, -2.6948409, -4.6163568, -2.7548733, -1.7808752, 1.8054340
6: -4.7207699, -2.1557038, -4.6937370, -2.2057319, -2.1721416, 2.1750684
7: -8.7282143, -6.8563685, -8.7166748, -6.8750615, -1.4937794, 1.4998503
8: -4.6627421, -2.3795121, -4.6383018, -2.4350104, -1.9672198, 2.0025353
9: -12.0838232, -9.7559948, -12.0737743, -9.7603779, -1.5756378, 1.5681076

Time for backsubstitution: 15.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9235999, upper bound: 0.9418509
time: 5.25 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9290956, upper bound: 0.9418512
time: 5.56 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -10.8327427, -8.2325172, -10.8637476, -8.2271538, -2.1717801, 2.1798990
1: -3.3540702, -0.8304625, -3.3551629, -0.8717890, -2.2660122, 2.2754979
2: 1.5642705, 3.6437109, 1.6043967, 3.6315553, -1.8909454, 1.9011526
3: -7.3149586, -5.2276173, -7.2652884, -5.2359872, -1.9813924, 1.9439805
4: -2.3616037, -0.4220018, -2.3649111, -0.3222642, -1.5685706, 1.5633836
5: -4.6308661, -2.6948409, -4.6255507, -2.7302155, -1.8048563, 1.8149180
6: -4.7207699, -2.1557038, -4.7280998, -2.1204927, -2.1998711, 2.2058253
7: -8.7282143, -6.8563685, -8.7214088, -6.8390384, -1.5029268, 1.5058568
8: -4.6627421, -2.3795121, -4.6898065, -2.4254057, -1.9776378, 2.0529847
9: -12.0838232, -9.7559948, -12.1676559, -9.7390251, -1.5990376, 1.5928295

Time for backsubstitution: 15.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9235999, upper bound: 0.9418540
time: 7.20 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9290956, upper bound: 0.9418510
time: 9.64 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -10.8881435, -8.2206697, -10.8084745, -8.2390308, -2.1925416, 2.1535277
1: -3.3762975, -0.7954248, -3.3329079, -0.9067831, -2.2518587, 2.2591352
2: 1.5120947, 3.6584711, 1.6565111, 3.6167736, -1.8839083, 1.8866861
3: -7.3232489, -5.2209263, -7.2572231, -5.2426581, -1.9851165, 1.9423664
4: -2.3929527, -0.3174589, -2.3337822, -0.4268136, -1.5870115, 1.5583112
5: -4.6400547, -2.6700764, -4.6163568, -2.7548733, -1.7903748, 1.8097761
6: -4.7550316, -2.0704255, -4.6937370, -2.2057319, -2.2025270, 2.1789560
7: -8.7329617, -6.8202329, -8.7166748, -6.8750615, -1.4997628, 1.5173954
8: -4.7142630, -2.3699050, -4.6383018, -2.4350104, -2.0178270, 2.0129499
9: -12.1778002, -9.7346172, -12.0737743, -9.7603779, -1.5960052, 1.5915275

Time for backsubstitution: 15.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9235968, upper bound: 0.9482542
time: 5.33 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9290925, upper bound: 0.9482544
time: 5.60 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -10.8881435, -8.2206697, -10.8637476, -8.2271538, -2.2000580, 2.1817064
1: -3.3762975, -0.7954248, -3.3551629, -0.8717890, -2.2649150, 2.2808504
2: 1.5120947, 3.6584711, 1.6043967, 3.6315553, -1.8991699, 1.9027383
3: -7.3232489, -5.2209263, -7.2652884, -5.2359872, -1.9942031, 1.9673295
4: -2.3929527, -0.3174589, -2.3649111, -0.3222642, -1.6006365, 1.5887313
5: -4.6400547, -2.6700764, -4.6255507, -2.7302155, -1.8224044, 1.8234210
6: -4.7550316, -2.0704255, -4.7280998, -2.1204927, -2.2305903, 2.2097120
7: -8.7329617, -6.8202329, -8.7214088, -6.8390384, -1.5114467, 1.5259318
8: -4.7142630, -2.3699050, -4.6898065, -2.4254057, -2.0151620, 2.0506372
9: -12.1778002, -9.7346172, -12.1676559, -9.7390251, -1.6164484, 1.6089277

Time for backsubstitution: 18.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9235968, upper bound: 0.9482561
time: 5.61 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9290925, upper bound: 0.9482561
time: 7.53 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -10.8342667, -8.2325087, -10.8342667, -8.2325087, -2.1822596, 2.1822591
1: -3.3540924, -0.8299019, -3.3540924, -0.8299019, -2.2626200, 2.2626200
2: 1.5641996, 3.6438277, 1.5641996, 3.6438277, -1.9054220, 1.9054217
3: -7.3152289, -5.2275262, -7.3152289, -5.2275262, -1.9604721, 1.9604719
4: -2.3626270, -0.4219837, -2.3626270, -0.4219837, -1.5550514, 1.5550516
5: -4.6311669, -2.6947558, -4.6311669, -2.6947558, -1.8060970, 1.8060973
6: -4.7211199, -2.1557031, -4.7211199, -2.1557031, -2.2016096, 2.2016096
7: -8.7282162, -6.8550620, -8.7282162, -6.8550620, -1.5155797, 1.5155795
8: -4.6630821, -2.3795116, -4.6630821, -2.3795116, -2.0189838, 2.0189838
9: -12.0843811, -9.7559776, -12.0843811, -9.7559776, -1.5880821, 1.5880821

Time for backsubstitution: 15.17 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=1.8685765266418457
rel_dist={2: [-0.9484739266423019, 0.9484713204465995]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 5733
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4625

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8355526, upper bound: 0.8258965
time: 5.30 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8355526, upper bound: 0.8355513
time: 5.13 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 10.64 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 10.64
Output dim: 2, lower bound: -0.8355526, upper bound: 0.8258965
IS_A2, status: Status.UNKNOWN, split count: 1, time: 10.64
Output dim: 2, lower bound: -0.8355526, upper bound: 0.8355513

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -10.8091249, -8.2323895, -10.8105335, -8.2313166, -2.0695071, 2.0686722
1: -3.3391936, -0.9065280, -3.3483040, -0.9047279, -2.1583028, 2.1661739
2: 1.6559384, 3.6253042, 1.6549829, 3.6376810, -1.8257685, 1.8144190
3: -7.2581921, -5.2406983, -7.2594333, -5.2341785, -1.8833628, 1.8781242
4: -2.3505065, -0.4264469, -2.3512645, -0.4246421, -1.4918129, 1.4903264
5: -4.6192904, -2.7543774, -4.6263461, -2.7534943, -1.7388844, 1.7449594
6: -4.7066774, -2.2050157, -4.7140570, -2.2044270, -2.0976725, 2.1055214
7: -8.7201481, -6.8739891, -8.7201986, -6.8716650, -1.4501972, 1.4476295
8: -4.6400385, -2.4291024, -4.6470275, -2.4286847, -1.8941536, 1.9013643
9: -12.0747089, -9.7458115, -12.0749407, -9.7449484, -1.5012100, 1.5004005

Time for backsubstitution: 15.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4625

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8258969, upper bound: 0.8258940
time: 7.16 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8258969, upper bound: 0.8258954
time: 9.94 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -10.8334141, -8.2258654, -10.8108492, -8.2310753, -2.0990772, 2.0981865
1: -3.3603547, -0.8302000, -3.3503830, -0.9043274, -2.1859446, 2.2104073
2: 1.5636945, 3.6522486, 1.6547688, 3.6404383, -1.8602004, 1.8412619
3: -7.3159385, -5.2256513, -7.2597122, -5.2327185, -1.9431777, 1.8952103
4: -2.3782635, -0.4216280, -2.3514342, -0.4242325, -1.5214677, 1.5026731
5: -4.6338072, -2.6943326, -4.6279144, -2.7532964, -1.7534189, 1.7843430
6: -4.7337027, -2.1549873, -4.7157078, -2.2042971, -2.1447034, 2.1392715
7: -8.7316866, -6.8552785, -8.7202091, -6.8711433, -1.4651389, 1.4670002
8: -4.6645212, -2.3736076, -4.6485896, -2.4285898, -1.9151678, 1.9595249
9: -12.0847721, -9.7414331, -12.0749931, -9.7447510, -1.5135770, 1.5155725

Time for backsubstitution: 15.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4625

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8258969, upper bound: 0.8355497
time: 5.63 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8258969, upper bound: 0.8355519
time: 13.64 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 34.49 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 34.49
Output dim: 2, lower bound: -0.8258969, upper bound: 0.8258940
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 34.49
Output dim: 2, lower bound: -0.8258969, upper bound: 0.8258954
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 34.49
Output dim: 2, lower bound: -0.8258969, upper bound: 0.8355497
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 34.49
Output dim: 2, lower bound: -0.8258969, upper bound: 0.8355519

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -10.8319349, -8.2258739, -10.8091249, -8.2323895, -2.0930014, 2.0763664
1: -3.3603342, -0.8307433, -3.3391936, -0.9065280, -2.1768379, 2.1978908
2: 1.5637659, 3.6521349, 1.6559384, 3.6253042, -1.8449171, 1.8419013
3: -7.3156757, -5.2257400, -7.2581921, -5.2406983, -1.9349985, 1.8948183
4: -2.3772869, -0.4216452, -2.3505065, -0.4264469, -1.5171188, 1.4945419
5: -4.6335154, -2.6944149, -4.6192904, -2.7543774, -1.7536111, 1.7756679
6: -4.7333646, -2.1549883, -4.7066774, -2.2050157, -2.1260109, 2.1280954
7: -8.7316866, -6.8565445, -8.7201481, -6.8739891, -1.4581563, 1.4638863
8: -4.6641922, -2.3736088, -4.6400385, -2.4291024, -1.9145641, 1.9500413
9: -12.0842342, -9.7414494, -12.0747089, -9.7458115, -1.5105853, 1.5035777

Time for backsubstitution: 15.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5733
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5733

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8258628, upper bound: 0.8290373
time: 5.28 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8258640, upper bound: 0.8355145
time: 4.84 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -10.8349228, -8.2258577, -10.8349228, -8.2258577, -2.1158214, 2.1158214
1: -3.3603773, -0.8296475, -3.3603773, -0.8296475, -2.2052989, 2.2052989
2: 1.5636239, 3.6523647, 1.5636239, 3.6523647, -1.8724542, 1.8724539
3: -7.3162060, -5.2255607, -7.3162060, -5.2255607, -1.9195347, 1.9195347
4: -2.3792734, -0.4216075, -2.3792734, -0.4216075, -1.5172246, 1.5172249
5: -4.6341038, -2.6942487, -4.6341038, -2.6942487, -1.7771978, 1.7771981
6: -4.7340474, -2.1549864, -4.7340474, -2.1549864, -2.1535206, 2.1535211
7: -8.7316875, -6.8539867, -8.7316875, -6.8539867, -1.4795229, 1.4795229
8: -4.6648607, -2.3736062, -4.6648607, -2.3736062, -1.9663439, 1.9663444
9: -12.0853224, -9.7414150, -12.0853224, -9.7414150, -1.5231352, 1.5231352

Time for backsubstitution: 14.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5733
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5733

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8258642, upper bound: 0.8290416
time: 6.89 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8258640, upper bound: 0.8355175
time: 5.74 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 27.75 seconds
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 27.75
Output dim: 2, lower bound: -0.8258628, upper bound: 0.8290373
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 27.75
Output dim: 2, lower bound: -0.8258640, upper bound: 0.8355145
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 27.75
Output dim: 2, lower bound: -0.8258642, upper bound: 0.8290416
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 27.75
Output dim: 2, lower bound: -0.8258640, upper bound: 0.8355175

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -10.8866825, -8.2206783, -10.8091249, -8.2324028, -2.1245241, 2.0815077
1: -3.3762760, -0.7959613, -3.3391802, -0.9065299, -2.1965446, 2.2011375
2: 1.5121646, 3.6583600, 1.6559399, 3.6252818, -1.8521101, 1.8520937
3: -7.3229899, -5.2210135, -7.2581882, -5.2407036, -1.9435854, 1.8983936
4: -2.3919737, -0.3174767, -2.3504586, -0.4264479, -1.5328747, 1.5189996
5: -4.6397653, -2.6701579, -4.6192818, -2.7543776, -1.7825055, 1.7780638
6: -4.7547040, -2.0704260, -4.7066488, -2.2050190, -2.1431875, 2.1314583
7: -8.7329607, -6.8214846, -8.7201347, -6.8739920, -1.4607830, 1.4788382
8: -4.7139339, -2.3699050, -4.6400323, -2.4291220, -1.9606347, 1.9769914
9: -12.1772680, -9.7346354, -12.0747061, -9.7458611, -1.5281863, 1.5204334

Time for backsubstitution: 15.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5733

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8193851, upper bound: 0.8355195
time: 6.27 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8193851, upper bound: 0.8355194
time: 7.06 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -10.8896751, -8.2206631, -10.8349199, -8.2258720, -2.1360126, 2.1209612
1: -3.3763201, -0.7948638, -3.3603649, -0.8296475, -2.2250366, 2.2217984
2: 1.5120236, 3.6585867, 1.5636256, 3.6523423, -1.8796461, 1.8793509
3: -7.3235197, -5.2208347, -7.3162031, -5.2255659, -1.9425039, 1.9231119
4: -2.3939757, -0.3174410, -2.3792248, -0.4216084, -1.5332563, 1.5366540
5: -4.6403561, -2.6699915, -4.6340952, -2.6942499, -1.7976313, 1.7929235
6: -4.7553735, -2.0704236, -4.7340174, -2.1549883, -2.1706810, 2.1636720
7: -8.7329617, -6.8189249, -8.7316742, -6.8539886, -1.4821489, 1.4860454
8: -4.7146053, -2.3699040, -4.6648555, -2.3736238, -2.0124173, 1.9932933
9: -12.1783562, -9.7346020, -12.0853186, -9.7414646, -1.5357833, 1.5399420

Time for backsubstitution: 14.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5733

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8193851, upper bound: 0.8355198
time: 9.11 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8193838, upper bound: 0.8355209
time: 7.49 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 31.69 seconds
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.69
Output dim: 2, lower bound: -0.8193851, upper bound: 0.8355195
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.69
Output dim: 2, lower bound: -0.8193851, upper bound: 0.8355194
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.69
Output dim: 2, lower bound: -0.8193851, upper bound: 0.8355198
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.69
Output dim: 2, lower bound: -0.8193838, upper bound: 0.8355209

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -10.8866825, -8.2206783, -10.8084745, -8.2390308, -2.1179361, 2.0807419
1: -3.3762760, -0.7959613, -3.3329079, -0.9067831, -2.1889281, 2.1936929
2: 1.5121646, 3.6583600, 1.6565111, 3.6167736, -1.8434262, 1.8474619
3: -7.3229899, -5.2210135, -7.2572231, -5.2426581, -1.9372864, 1.8956578
4: -2.3919737, -0.3174767, -2.3337822, -0.4268136, -1.5315806, 1.5021613
5: -4.6397653, -2.6701579, -4.6163568, -2.7548733, -1.7556109, 1.7725110
6: -4.7547040, -2.0704260, -4.6937370, -2.2057319, -2.1421781, 2.1177144
7: -8.7329607, -6.8214846, -8.7166748, -6.8750615, -1.4565170, 1.4736880
8: -4.7139339, -2.3699050, -4.6383018, -2.4350104, -1.9520350, 1.9473233
9: -12.1772680, -9.7346354, -12.0737743, -9.7603779, -1.5139668, 1.5111752

Time for backsubstitution: 14.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8153674, upper bound: 0.8354753
time: 5.26 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8193591, upper bound: 0.8354795
time: 6.80 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -10.8866825, -8.2206783, -10.8637476, -8.2271538, -2.1242642, 2.1075182
1: -3.3762760, -0.7959613, -3.3551629, -0.8717890, -2.2003717, 2.2154086
2: 1.5121646, 3.6583600, 1.6043967, 3.6315553, -1.8586884, 1.8615057
3: -7.3229899, -5.2210135, -7.2652884, -5.2359872, -1.9459910, 1.9199619
4: -2.3919737, -0.3174767, -2.3649111, -0.3222642, -1.5441685, 1.5302744
5: -4.6397653, -2.6701579, -4.6255507, -2.7302155, -1.7862782, 1.7861567
6: -4.7547040, -2.0704260, -4.7280998, -2.1204927, -2.1676164, 2.1484704
7: -8.7329607, -6.8214846, -8.7214088, -6.8390384, -1.4682007, 1.4822248
8: -4.7139339, -2.3699050, -4.6898065, -2.4254057, -1.9465580, 1.9821951
9: -12.1772680, -9.7346354, -12.1676559, -9.7390251, -1.5311089, 1.5241144

Time for backsubstitution: 14.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8153674, upper bound: 0.8354790
time: 5.97 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8193594, upper bound: 0.8354788
time: 6.37 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -10.8896751, -8.2206631, -10.8342667, -8.2325087, -2.1294255, 2.1201777
1: -3.3763201, -0.7948638, -3.3540924, -0.8299019, -2.2174292, 2.2143631
2: 1.5120236, 3.6585867, 1.5641996, 3.6438277, -1.8709557, 1.8779953
3: -7.3235197, -5.2208347, -7.3152289, -5.2275262, -1.9221859, 1.9203668
4: -2.3939757, -0.3174410, -2.3626270, -0.4219837, -1.5319520, 1.5198607
5: -4.6403561, -2.6699915, -4.6311669, -2.6947558, -1.7792048, 1.7873588
6: -4.7553735, -2.0704236, -4.7211199, -2.1557031, -2.1696734, 2.1499562
7: -8.7329617, -6.8189249, -8.7282162, -6.8550620, -1.4778912, 1.4808955
8: -4.7146053, -2.3699040, -4.6630821, -2.3795116, -2.0038266, 1.9636335
9: -12.1783562, -9.7346020, -12.0843811, -9.7559776, -1.5215728, 1.5306921

Time for backsubstitution: 14.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8153674, upper bound: 0.8354768
time: 5.34 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8193593, upper bound: 0.8354775
time: 6.85 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -10.8896751, -8.2206631, -10.8896751, -8.2206631, -2.1409464, 2.1409464
1: -3.3763201, -0.7948638, -3.3763201, -0.7948638, -2.2288451, 2.2288451
2: 1.5120236, 3.6585867, 1.5120236, 3.6585867, -1.8862240, 1.8862243
3: -7.3235197, -5.2208347, -7.3235197, -5.2208347, -1.9449339, 1.9449339
4: -2.3939757, -0.3174410, -2.3939757, -0.3174410, -1.5519505, 1.5519507
5: -4.6403561, -2.6699915, -4.6403561, -2.6699915, -1.8010483, 1.8010483
6: -4.7553735, -2.0704236, -4.7553735, -2.0704236, -2.1806717, 2.1806719
7: -8.7329617, -6.8189249, -8.7329617, -6.8189249, -1.4894011, 1.4894010
8: -4.7146053, -2.3699040, -4.7146053, -2.3699040, -1.9983478, 1.9983478
9: -12.1783562, -9.7346020, -12.1783562, -9.7346020, -1.5435503, 1.5435500

Time for backsubstitution: 15.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8153674, upper bound: 0.8354794
time: 9.33 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8193593, upper bound: 0.8354820
time: 6.71 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 31.32 seconds
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 31.32
Output dim: 2, lower bound: -0.8153674, upper bound: 0.8354753
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 31.32
Output dim: 2, lower bound: -0.8193591, upper bound: 0.8354795
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 31.32
Output dim: 2, lower bound: -0.8153674, upper bound: 0.8354790
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 31.32
Output dim: 2, lower bound: -0.8193594, upper bound: 0.8354788
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 31.32
Output dim: 2, lower bound: -0.8153674, upper bound: 0.8354768
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 31.32
Output dim: 2, lower bound: -0.8193593, upper bound: 0.8354775
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 31.32
Output dim: 2, lower bound: -0.8153674, upper bound: 0.8354794
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 31.32
Output dim: 2, lower bound: -0.8193593, upper bound: 0.8354820

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -10.8802710, -8.2369661, -10.8076305, -8.2469330, -2.0981154, 2.0647769
1: -3.3706245, -0.7978315, -3.3302343, -0.9070048, -2.1829672, 2.1883090
2: 1.5269938, 3.6548347, 1.6643476, 3.6165299, -1.8286083, 1.8361108
3: -7.2741613, -5.2384348, -7.2308593, -5.2431850, -1.8879130, 1.8516555
4: -2.3871140, -0.3253405, -2.3324037, -0.4308715, -1.5220037, 1.4925892
5: -4.6174064, -2.6785080, -4.6046419, -2.7556231, -1.7327356, 1.7472138
6: -4.7498960, -2.0832381, -4.6931572, -2.2127879, -2.1300182, 2.1042502
7: -8.7250614, -6.8420334, -8.7161407, -6.8865275, -1.4310424, 1.4522889
8: -4.6985722, -2.4138370, -4.6376963, -2.4582958, -1.9134502, 1.9029567
9: -12.1749859, -9.7382517, -12.0729799, -9.7622128, -1.5078585, 1.5059495

Time for backsubstitution: 14.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6111

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8076463, upper bound: 0.8341257
time: 6.04 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8153604, upper bound: 0.8354720
time: 5.45 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -10.8866806, -8.2206860, -10.8084755, -8.2390337, -2.1174121, 2.0799546
1: -3.3762741, -0.7959613, -3.3329055, -0.9067848, -2.1876826, 2.1927471
2: 1.5121701, 3.6583598, 1.6565132, 3.6167746, -1.8345461, 1.8474591
3: -7.3229647, -5.2210145, -7.2572117, -5.2426596, -1.9156666, 1.8956447
4: -2.3919706, -0.3174822, -2.3337820, -0.4268165, -1.5315777, 1.4986670
5: -4.6397533, -2.6701581, -4.6163511, -2.7548735, -1.7529693, 1.7659009
6: -4.7547045, -2.0704365, -4.6937361, -2.2057354, -2.1419082, 2.1061282
7: -8.7329607, -6.8214989, -8.7166739, -6.8750672, -1.4495101, 1.4514096
8: -4.7139330, -2.3699214, -4.6383028, -2.4350181, -1.9520245, 1.9241409
9: -12.1772671, -9.7346363, -12.0737734, -9.7603788, -1.5132382, 1.5100932

Time for backsubstitution: 15.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6111

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8116414, upper bound: 0.8341232
time: 8.22 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8193532, upper bound: 0.8354730
time: 7.03 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -10.8802710, -8.2369661, -10.8628988, -8.2350512, -2.1053123, 2.0915565
1: -3.3706245, -0.7978315, -3.3524895, -0.8720105, -2.1944094, 2.2100000
2: 1.5269938, 3.6548347, 1.6122674, 3.6313133, -1.8438711, 1.8501229
3: -7.2741613, -5.2384348, -7.2389140, -5.2365251, -1.8966277, 1.8759775
4: -2.3871140, -0.3253405, -2.3635516, -0.3263297, -1.5328674, 1.5207750
5: -4.6174064, -2.6785080, -4.6138067, -2.7309709, -1.7634521, 1.7608593
6: -4.7498960, -2.0832381, -4.7275381, -2.1275663, -2.1525974, 2.1350224
7: -8.7250614, -6.8420334, -8.7208862, -6.8505154, -1.4427211, 1.4608419
8: -4.6985722, -2.4138370, -4.6891947, -2.4486856, -1.9079895, 1.9378836
9: -12.1749859, -9.7382517, -12.1668510, -9.7408695, -1.5256982, 1.5189095

Time for backsubstitution: 14.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6111

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8076463, upper bound: 0.8341239
time: 7.10 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8162373, upper bound: 0.8354718
time: 6.20 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -10.8866806, -8.2206860, -10.8637486, -8.2271566, -2.1289029, 2.1067300
1: -3.3762741, -0.7959613, -3.3551629, -0.8717885, -2.1991258, 2.2144630
2: 1.5121701, 3.6583598, 1.6044002, 3.6315565, -1.8498092, 1.8615036
3: -7.3229647, -5.2210145, -7.2652779, -5.2359872, -1.9243751, 1.9199498
4: -2.3919706, -0.3174822, -2.3649116, -0.3222661, -1.5420125, 1.5275042
5: -4.6397533, -2.6701581, -4.6255460, -2.7302153, -1.7822762, 1.7795472
6: -4.7547045, -2.0704365, -4.7280993, -2.1204967, -2.1639404, 2.1369133
7: -8.7329607, -6.8214989, -8.7214088, -6.8390455, -1.4611948, 1.4599707
8: -4.7139330, -2.3699214, -4.6898074, -2.4254131, -1.9465466, 1.9590106
9: -12.1772671, -9.7346363, -12.1676540, -9.7390289, -1.5311062, 1.5230324

Time for backsubstitution: 14.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6111

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8124892, upper bound: 0.8341227
time: 5.62 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8202315, upper bound: 0.8354725
time: 7.10 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -10.8832664, -8.2369509, -10.8334198, -8.2404108, -2.1096148, 2.1042328
1: -3.3706682, -0.7967327, -3.3514245, -0.8301234, -2.2114329, 2.2089252
2: 1.5268528, 3.6550603, 1.5720558, 3.6435905, -1.8561437, 1.8638079
3: -7.2746930, -5.2382579, -7.2888584, -5.2280455, -1.8731694, 1.8763540
4: -2.3890810, -0.3253050, -2.3611207, -0.4260387, -1.5224309, 1.5103359
5: -4.6179972, -2.6783412, -4.6194582, -2.6955109, -1.7563195, 1.7620685
6: -4.7505674, -2.0832353, -4.7205606, -2.1627779, -2.1574931, 2.1365099
7: -8.7250633, -6.8394732, -8.7276726, -6.8665414, -1.4524133, 1.4595156
8: -4.6992459, -2.4138360, -4.6625047, -2.4028015, -1.9600234, 1.9192705
9: -12.1760731, -9.7382193, -12.0835724, -9.7578135, -1.5154750, 1.5255070

Time for backsubstitution: 14.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6111

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8076463, upper bound: 0.8341233
time: 6.88 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8153604, upper bound: 0.8354734
time: 5.31 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -10.8896751, -8.2206717, -10.8342676, -8.2325115, -2.1289020, 2.1193905
1: -3.3763158, -0.7948618, -3.3540921, -0.8299012, -2.2161822, 2.2134178
2: 1.5120302, 3.6585872, 1.5642015, 3.6438277, -1.8620768, 1.8750896
3: -7.3234940, -5.2208357, -7.3152170, -5.2275267, -1.9067955, 1.9203525
4: -2.3939738, -0.3174472, -2.3626261, -0.4219847, -1.5319488, 1.5163844
5: -4.6403446, -2.6699924, -4.6311607, -2.6947560, -1.7765636, 1.7807491
6: -4.7553740, -2.0704331, -4.7211194, -2.1557064, -2.1694050, 2.1384065
7: -8.7329626, -6.8189392, -8.7282152, -6.8550682, -1.4708850, 1.4586482
8: -4.7146058, -2.3699193, -4.6630850, -2.3795173, -1.9952898, 1.9404502
9: -12.1783552, -9.7346039, -12.0843773, -9.7559776, -1.5208437, 1.5296099

Time for backsubstitution: 14.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6111

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8116422, upper bound: 0.8341254
time: 6.68 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8193535, upper bound: 0.8354732
time: 5.93 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -10.8832664, -8.2369509, -10.8888235, -8.2285595, -2.1211429, 2.1251264
1: -3.3706682, -0.7967327, -3.3736498, -0.7950845, -2.2228489, 2.2242608
2: 1.5268528, 3.6550603, 1.5199126, 3.6583533, -1.8714128, 1.8720303
3: -7.2746930, -5.2382579, -7.2971387, -5.2213612, -1.8959265, 1.9009404
4: -2.3890810, -0.3253050, -2.3924901, -0.3215048, -1.5406761, 1.5424464
5: -4.6179972, -2.6783412, -4.6286178, -2.6707654, -1.7782154, 1.7757347
6: -4.7505674, -2.0832353, -4.7548332, -2.0775151, -2.1656122, 2.1672406
7: -8.7250633, -6.8394732, -8.7324295, -6.8304133, -1.4639184, 1.4680369
8: -4.6992459, -2.4138360, -4.7140236, -2.3931892, -1.9597702, 1.9540391
9: -12.1760731, -9.7382193, -12.1775379, -9.7364426, -1.5382030, 1.5383842

Time for backsubstitution: 15.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6111

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8076463, upper bound: 0.8341278
time: 6.65 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8162374, upper bound: 0.8354738
time: 8.33 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -10.8896751, -8.2206717, -10.8896751, -8.2206678, -2.1404209, 2.1377034
1: -3.3763158, -0.7948618, -3.3763187, -0.7948623, -2.2275987, 2.2288437
2: 1.5120302, 3.6585872, 1.5120273, 3.6585860, -1.8773463, 1.8833175
3: -7.3234940, -5.2208357, -7.3235064, -5.2208352, -1.9295421, 1.9449215
4: -2.3939738, -0.3174472, -2.3939750, -0.3174436, -1.5497952, 1.5484879
5: -4.6403446, -2.6699924, -4.6403503, -2.6699915, -1.7953477, 1.7944379
6: -4.7553740, -2.0704331, -4.7553759, -2.0704281, -2.1769614, 2.1691494
7: -8.7329626, -6.8189392, -8.7329617, -6.8189297, -1.4823952, 1.4671774
8: -4.7146058, -2.3699193, -4.7146053, -2.3699107, -1.9983373, 1.9751644
9: -12.1783552, -9.7346039, -12.1783552, -9.7346029, -1.5435476, 1.5424681

Time for backsubstitution: 14.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6111

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8124895, upper bound: 0.8341242
time: 5.21 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8202316, upper bound: 0.8354736
time: 5.49 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 25.77 seconds
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 25.77
Output dim: 2, lower bound: -0.8076463, upper bound: 0.8341257
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 25.77
Output dim: 2, lower bound: -0.8153604, upper bound: 0.8354720
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 25.77
Output dim: 2, lower bound: -0.8116414, upper bound: 0.8341232
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 25.77
Output dim: 2, lower bound: -0.8193532, upper bound: 0.8354730
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 25.77
Output dim: 2, lower bound: -0.8076463, upper bound: 0.8341239
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 25.77
Output dim: 2, lower bound: -0.8162373, upper bound: 0.8354718
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 25.77
Output dim: 2, lower bound: -0.8124892, upper bound: 0.8341227
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 25.77
Output dim: 2, lower bound: -0.8202315, upper bound: 0.8354725
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 25.77
Output dim: 2, lower bound: -0.8076463, upper bound: 0.8341233
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 25.77
Output dim: 2, lower bound: -0.8153604, upper bound: 0.8354734
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 25.77
Output dim: 2, lower bound: -0.8116422, upper bound: 0.8341254
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 25.77
Output dim: 2, lower bound: -0.8193535, upper bound: 0.8354732
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 25.77
Output dim: 2, lower bound: -0.8076463, upper bound: 0.8341278
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 25.77
Output dim: 2, lower bound: -0.8162374, upper bound: 0.8354738
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 25.77
Output dim: 2, lower bound: -0.8124895, upper bound: 0.8341242
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 25.77
Output dim: 2, lower bound: -0.8202316, upper bound: 0.8354736

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -10.8779221, -8.2472839, -10.7978926, -8.2699575, -2.0729494, 2.0426607
1: -3.3573744, -0.8034081, -3.3104799, -0.9220741, -2.1533585, 2.1493764
2: 1.5295174, 3.6298022, 1.6776460, 3.5633278, -1.7718353, 1.7893519
3: -7.2713695, -5.2596149, -7.2139316, -5.2698002, -1.8457761, 1.7796969
4: -2.3707826, -0.3327980, -2.3068810, -0.4638810, -1.4751184, 1.4405236
5: -4.6039219, -2.6799393, -4.5822620, -2.7623916, -1.7070212, 1.7183547
6: -4.7357121, -2.0888944, -4.6684880, -2.2319288, -2.1001735, 2.0644331
7: -8.7165489, -6.8483992, -8.6755695, -6.9017215, -1.3908100, 1.3999202
8: -4.6914568, -2.4172144, -4.6246328, -2.4744601, -1.8882976, 1.8835349
9: -12.1510086, -9.7487688, -11.9832211, -9.7837801, -1.4103649, 1.4109712

Time for backsubstitution: 14.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4632

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8076405, upper bound: 0.8330282
time: 5.16 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8076446, upper bound: 0.8341213
time: 6.04 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -10.8798943, -8.2371655, -10.8076296, -8.2469330, -2.0976901, 2.0688534
1: -3.3658442, -0.7979180, -3.3302345, -0.9070063, -2.1788640, 2.1988728
2: 1.5275314, 3.6536393, 1.6643467, 3.6165276, -1.7931147, 1.8348167
3: -7.2739453, -5.2389674, -7.2308602, -5.2431870, -1.8869789, 1.8689992
4: -2.3831937, -0.3254442, -2.3324034, -0.4308729, -1.5054793, 1.4916980
5: -4.6168242, -2.6788807, -4.6046405, -2.7556233, -1.7320409, 1.7415166
6: -4.7457018, -2.0835769, -4.6931567, -2.2127876, -2.1254902, 2.1050162
7: -8.7248993, -6.8431759, -8.7161388, -6.8865271, -1.4282699, 1.4387331
8: -4.6981897, -2.4147301, -4.6376963, -2.4582949, -1.9131355, 1.8988748
9: -12.1744308, -9.7401543, -12.0729799, -9.7622108, -1.5029061, 1.4428356

Time for backsubstitution: 14.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6111

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8139556, upper bound: 0.8278134
time: 5.92 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8139556, upper bound: 0.8278150
time: 5.25 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -10.8843107, -8.2309971, -10.7986908, -8.2620411, -2.0922661, 2.0577078
1: -3.3630254, -0.8015410, -3.3131640, -0.9218526, -2.1580629, 2.1538339
2: 1.5146911, 3.6333210, 1.6698211, 3.5635633, -1.7777743, 1.8006852
3: -7.3201852, -5.2421956, -7.2402973, -5.2692709, -1.8735375, 1.8237343
4: -2.3755646, -0.3249335, -2.3082166, -0.4598083, -1.4846926, 1.4465520
5: -4.6262755, -2.6715949, -4.5940523, -2.7616546, -1.7272491, 1.7371271
6: -4.7405043, -2.0760844, -4.6690388, -2.2248645, -2.1120648, 2.0663662
7: -8.7244568, -6.8278637, -8.6761150, -6.8902617, -1.4093084, 1.3990868
8: -4.7068343, -2.3732846, -4.6252418, -2.4511676, -1.9268966, 1.9047306
9: -12.1533031, -9.7451458, -11.9840260, -9.7819462, -1.4158142, 1.4151394

Time for backsubstitution: 14.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4632

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8116373, upper bound: 0.8330285
time: 5.74 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8116397, upper bound: 0.8341238
time: 7.82 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -10.8863077, -8.2208872, -10.8084745, -8.2390337, -2.1169882, 2.0840316
1: -3.3714876, -0.7960486, -3.3329067, -0.9067831, -2.1835747, 2.2033114
2: 1.5127076, 3.6571670, 1.6565137, 3.6167734, -1.7990584, 1.8461657
3: -7.3227453, -5.2215476, -7.2572117, -5.2426605, -1.9147429, 1.9129918
4: -2.3880248, -0.3175859, -2.3337815, -0.4268169, -1.5150576, 1.4977744
5: -4.6391692, -2.6705275, -4.6163502, -2.7548728, -1.7522755, 1.7601986
6: -4.7505097, -2.0707746, -4.6937361, -2.2057371, -2.1373858, 2.1068933
7: -8.7328024, -6.8226452, -8.7166729, -6.8750677, -1.4467413, 1.4378579
8: -4.7135501, -2.3708110, -4.6383023, -2.4350162, -1.9517117, 1.9200850
9: -12.1767206, -9.7365379, -12.0737705, -9.7603779, -1.5082932, 1.4469800

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6111

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8179505, upper bound: 0.8278162
time: 6.45 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8179506, upper bound: 0.8354725
time: 5.59 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -10.8782320, -8.2470388, -10.8532457, -8.2593069, -2.0795140, 2.0697961
1: -3.3622892, -0.8033355, -3.3246124, -0.8867388, -2.1731596, 2.1706412
2: 1.5290279, 3.6324701, 1.6254092, 3.5745840, -1.7840343, 1.8007455
3: -7.2715497, -5.2594423, -7.2218218, -5.2637358, -1.8441372, 1.8027170
4: -2.3760681, -0.3327062, -2.3314753, -0.3593993, -1.4913769, 1.4734173
5: -4.6047339, -2.6795468, -4.5900221, -2.7380559, -1.7390332, 1.7355509
6: -4.7408805, -2.0885777, -4.6965151, -2.1464548, -2.1284266, 2.0964880
7: -8.7167072, -6.8479056, -8.6798515, -6.8657427, -1.4010901, 1.4113886
8: -4.6916618, -2.4162197, -4.6760716, -2.4660668, -1.8839431, 1.9188461
9: -12.1512556, -9.7456369, -12.0771570, -9.7657728, -1.4404705, 1.4279106

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4632

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8076405, upper bound: 0.8330287
time: 5.84 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8076446, upper bound: 0.8341221
time: 8.69 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -10.8802710, -8.2369661, -10.8628979, -8.2350512, -2.1053109, 2.0958610
1: -3.3706245, -0.7978315, -3.3524892, -0.8720117, -2.1944075, 2.2205706
2: 1.5269938, 3.6548347, 1.6122668, 3.6313119, -1.8094583, 1.8501232
3: -7.2741613, -5.2384348, -7.2389131, -5.2365274, -1.8966267, 1.8941185
4: -2.3871140, -0.3253405, -2.3635521, -0.3263316, -1.5211368, 1.5207741
5: -4.6174064, -2.6785080, -4.6138058, -2.7309709, -1.7634082, 1.7557659
6: -4.7498960, -2.0832381, -4.7275381, -2.1275668, -2.1525974, 2.1360765
7: -8.7250614, -6.8420334, -8.7208843, -6.8505139, -1.4403527, 1.4483449
8: -4.6985722, -2.4138370, -4.6891952, -2.4486859, -1.9079885, 1.9349236
9: -12.1749859, -9.7382517, -12.1668453, -9.7408695, -1.5256975, 1.4585800

Time for backsubstitution: 14.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6111

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8139556, upper bound: 0.8278157
time: 5.45 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8147997, upper bound: 0.8278165
time: 10.29 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -10.8846149, -8.2307510, -10.8540478, -8.2513943, -2.1029596, 2.0848408
1: -3.3679435, -0.8014679, -3.3272934, -0.8865144, -2.1778750, 2.1751523
2: 1.5142026, 3.6359885, 1.6175439, 3.5748148, -1.7899714, 1.8120220
3: -7.3203659, -5.2420235, -7.2482018, -5.2631979, -1.8718915, 1.8467348
4: -2.3808846, -0.3248410, -2.3327851, -0.3553188, -1.5006220, 1.4795063
5: -4.6270914, -2.6712079, -4.6018028, -2.7373140, -1.7572687, 1.7543390
6: -4.7456717, -2.0757699, -4.6970463, -2.1393712, -2.1397753, 2.0984361
7: -8.7246113, -6.8273687, -8.6803856, -6.8542719, -1.4195902, 1.4105635
8: -4.7070384, -2.3722875, -4.6766825, -2.4427819, -1.9225326, 1.9400041
9: -12.1535425, -9.7420139, -12.0779743, -9.7639284, -1.4458890, 1.4320550

Time for backsubstitution: 14.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4632

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8124758, upper bound: 0.8330287
time: 6.05 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8124875, upper bound: 0.8341209
time: 5.38 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -10.8866806, -8.2206860, -10.8637466, -8.2271576, -2.1289029, 2.1110349
1: -3.3762741, -0.7959613, -3.3551619, -0.8717902, -2.1991248, 2.2250333
2: 1.5121701, 3.6583598, 1.6043988, 3.6315539, -1.8154035, 1.8615031
3: -7.3229647, -5.2210145, -7.2652769, -5.2359891, -1.9243741, 1.9380918
4: -2.3919706, -0.3174822, -2.3649111, -0.3222678, -1.5302815, 1.5275037
5: -4.6397533, -2.6701581, -4.6255450, -2.7302158, -1.7814560, 1.7744458
6: -4.7547045, -2.0704365, -4.7281003, -2.1204977, -2.1639404, 2.1379669
7: -8.7329607, -6.8214989, -8.7214088, -6.8390455, -1.4588263, 1.4474777
8: -4.7139330, -2.3699214, -4.6898065, -2.4254134, -1.9465470, 1.9560785
9: -12.1772671, -9.7346363, -12.1676483, -9.7390280, -1.5311060, 1.4627028

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6111

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8187984, upper bound: 0.8278151
time: 5.89 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8187985, upper bound: 0.8278164
time: 11.20 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 31.93 seconds
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 31.93
Output dim: 2, lower bound: -0.8076405, upper bound: 0.8330282
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 31.93
Output dim: 2, lower bound: -0.8076446, upper bound: 0.8341213
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 31.93
Output dim: 2, lower bound: -0.8139556, upper bound: 0.8278134
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 31.93
Output dim: 2, lower bound: -0.8139556, upper bound: 0.8278150
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 31.93
Output dim: 2, lower bound: -0.8116373, upper bound: 0.8330285
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 31.93
Output dim: 2, lower bound: -0.8116397, upper bound: 0.8341238
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 31.93
Output dim: 2, lower bound: -0.8179505, upper bound: 0.8278162
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 31.93
Output dim: 2, lower bound: -0.8179506, upper bound: 0.8354725
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 31.93
Output dim: 2, lower bound: -0.8076405, upper bound: 0.8330287
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 31.93
Output dim: 2, lower bound: -0.8076446, upper bound: 0.8341221
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 31.93
Output dim: 2, lower bound: -0.8139556, upper bound: 0.8278157
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 31.93
Output dim: 2, lower bound: -0.8147997, upper bound: 0.8278165
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 31.93
Output dim: 2, lower bound: -0.8124758, upper bound: 0.8330287
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 31.93
Output dim: 2, lower bound: -0.8124875, upper bound: 0.8341209
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 31.93
Output dim: 2, lower bound: -0.8187984, upper bound: 0.8278151
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 31.93
Output dim: 2, lower bound: -0.8187985, upper bound: 0.8278164
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 31.93
Output dim: 2, lower bound: -0.8076463, upper bound: 0.8341233
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 31.93
Output dim: 2, lower bound: -0.8153604, upper bound: 0.8354734
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 31.93
Output dim: 2, lower bound: -0.8116422, upper bound: 0.8341254
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 31.93
Output dim: 2, lower bound: -0.8193535, upper bound: 0.8354732
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 31.93
Output dim: 2, lower bound: -0.8076463, upper bound: 0.8341278
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 31.93
Output dim: 2, lower bound: -0.8162374, upper bound: 0.8354738
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 31.93
Output dim: 2, lower bound: -0.8124895, upper bound: 0.8341242
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 31.93
Output dim: 2, lower bound: -0.8202316, upper bound: 0.8354736
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=1.8296241760253906
rel_dist={2: [-0.835577596891917, 0.8355774930246449]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2408.95 seconds
