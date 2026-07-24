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
execution time: IAR + LP analysis = 15.41 + 32.93 = 48.34 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3551.66 seconds, max iter: 100)

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
Binary search time: 207.68 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual) starts
Time budget: 3343.97 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 5733
type: B, layer: 1, pos: 5733
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4625

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2084882, upper bound: 1.1960950
time: 4.83 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2084840, upper bound: 1.2084825
time: 6.47 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 11.52 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 11.52
Output dim: 2, lower bound: -1.2084882, upper bound: 1.1960950
IS_A2, status: Status.UNKNOWN, split count: 1, time: 11.52
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

Time for backsubstitution: 14.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 5733
type: A, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4625

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1960950, upper bound: 1.1960953
time: 5.36 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1960950, upper bound: 1.1960974
time: 6.76 seconds

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

Time for backsubstitution: 15.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 5733
type: B, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 4625

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1960957, upper bound: 1.2084851
time: 5.62 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1960950, upper bound: 1.2084860
time: 6.03 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 27.49 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 27.49
Output dim: 2, lower bound: -1.1960950, upper bound: 1.1960953
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 27.49
Output dim: 2, lower bound: -1.1960950, upper bound: 1.1960974
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 27.49
Output dim: 2, lower bound: -1.1960957, upper bound: 1.2084851
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 27.49
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

Time for backsubstitution: 14.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5733
type: B, layer: 1, pos: 5733
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5733

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1960752, upper bound: 1.1899124
time: 5.02 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1960720, upper bound: 1.1960610
time: 5.37 seconds

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

Time for backsubstitution: 15.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5733
type: B, layer: 1, pos: 5733
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5733

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1960744, upper bound: 1.1899120
time: 6.61 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1960722, upper bound: 1.1960610
time: 5.59 seconds

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

Time for backsubstitution: 14.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5733
type: A, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5733

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1899124, upper bound: 1.2084534
time: 6.30 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1960609, upper bound: 1.2084486
time: 5.12 seconds

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

Time for backsubstitution: 15.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5733
type: B, layer: 1, pos: 5733
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5733

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1960642, upper bound: 1.2022144
time: 6.77 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1960617, upper bound: 1.2084471
time: 6.27 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 28.56 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 28.56
Output dim: 2, lower bound: -1.1960752, upper bound: 1.1899124
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.56
Output dim: 2, lower bound: -1.1960720, upper bound: 1.1960610
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.56
Output dim: 2, lower bound: -1.1960744, upper bound: 1.1899120
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.56
Output dim: 2, lower bound: -1.1960722, upper bound: 1.1960610
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 28.56
Output dim: 2, lower bound: -1.1899124, upper bound: 1.2084534
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 28.56
Output dim: 2, lower bound: -1.1960609, upper bound: 1.2084486
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.56
Output dim: 2, lower bound: -1.1960642, upper bound: 1.2022144
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.56
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

Time for backsubstitution: 15.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 1, pos: 5733

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1899222, upper bound: 1.1899224
time: 5.61 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1899222, upper bound: 1.1899245
time: 6.19 seconds

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

Time for backsubstitution: 15.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

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
time: 5.42 seconds

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

Time for backsubstitution: 14.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5733
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5733

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2022181, upper bound: 1.1899101
time: 7.83 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2022181, upper bound: 1.1899122
time: 6.32 seconds

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

Time for backsubstitution: 14.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5733

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2022181, upper bound: 1.1960616
time: 5.66 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2022180, upper bound: 1.1960618
time: 5.85 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -10.8349228, -8.2258577, -10.8084745, -8.2390308, -2.3808208, 2.3669791
1: -3.3603773, -0.8296475, -3.3329079, -0.9067831, -2.4276667, 2.4514930
2: 1.5636239, 3.6523647, 1.6565111, 3.6167736, -1.9976654, 1.9958537
3: -7.3162060, -5.2255607, -7.2572231, -5.2426581, -2.0735478, 2.0316625
4: -2.3792734, -0.4216075, -2.3337822, -0.4268136, -1.7362555, 1.6956065
5: -4.6341038, -2.6942487, -4.6163568, -2.7548733, -1.8792305, 1.9189088
6: -4.7340474, -2.1549864, -4.6937370, -2.2057319, -2.3660917, 2.3592167
7: -8.7316875, -6.8539867, -8.7166748, -6.8750615, -1.6273832, 1.6312124
8: -4.6648607, -2.3736062, -4.6383018, -2.4350104, -2.1685619, 2.2078018
9: -12.0853224, -9.7414150, -12.0737743, -9.7603779, -1.8188157, 1.8232303

Time for backsubstitution: 14.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5733

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1899124, upper bound: 1.2022177
time: 10.75 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1899124, upper bound: 1.2084524
time: 7.04 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -10.8349228, -8.2258663, -10.8637476, -8.2271538, -2.3981829, 2.4071908
1: -3.3603733, -0.8296473, -3.3551629, -0.8717890, -2.4622130, 2.4794440
2: 1.5636244, 3.6523528, 1.6043967, 3.6315553, -2.0212975, 2.0308251
3: -7.3162050, -5.2255630, -7.2652884, -5.2359872, -2.0802178, 2.0397253
4: -2.3792491, -0.4216084, -2.3649111, -0.3222642, -1.7533388, 1.7439775
5: -4.6340995, -2.6942494, -4.6255507, -2.7302155, -1.9038839, 1.9313014
6: -4.7340341, -2.1549869, -4.7280998, -2.1204927, -2.3976068, 2.4019704
7: -8.7316818, -6.8539877, -8.7214088, -6.8390384, -1.6377854, 1.6553243
8: -4.6648579, -2.3736148, -4.6898065, -2.4254057, -2.2153854, 2.2582409
9: -12.0853195, -9.7414398, -12.1676559, -9.7390251, -1.8676748, 1.8519629

Time for backsubstitution: 14.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5733
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 5841
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5733

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1960611, upper bound: 1.2022171
time: 4.54 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1960611, upper bound: 1.2084522
time: 5.12 seconds

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

Time for backsubstitution: 15.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5733

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1899123, upper bound: 1.2022137
time: 6.09 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1899120, upper bound: 1.2022157
time: 6.14 seconds

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

Time for backsubstitution: 15.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5733

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1899120, upper bound: 1.2084477
time: 4.62 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1899123, upper bound: 1.2084502
time: 5.57 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 25.41 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.41
Output dim: 2, lower bound: -1.1899222, upper bound: 1.1899224
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.41
Output dim: 2, lower bound: -1.1899222, upper bound: 1.1899245
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.41
Output dim: 2, lower bound: -1.1899222, upper bound: 1.1960717
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.41
Output dim: 2, lower bound: -1.1899222, upper bound: 1.1960739
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.41
Output dim: 2, lower bound: -1.2022181, upper bound: 1.1899101
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.41
Output dim: 2, lower bound: -1.2022181, upper bound: 1.1899122
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.41
Output dim: 2, lower bound: -1.2022181, upper bound: 1.1960616
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.41
Output dim: 2, lower bound: -1.2022180, upper bound: 1.1960618
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 25.41
Output dim: 2, lower bound: -1.1899124, upper bound: 1.2022177
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 25.41
Output dim: 2, lower bound: -1.1899124, upper bound: 1.2084524
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 25.41
Output dim: 2, lower bound: -1.1960611, upper bound: 1.2022171
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 25.41
Output dim: 2, lower bound: -1.1960611, upper bound: 1.2084522
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.41
Output dim: 2, lower bound: -1.1899123, upper bound: 1.2022137
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.41
Output dim: 2, lower bound: -1.1899120, upper bound: 1.2022157
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.41
Output dim: 2, lower bound: -1.1899120, upper bound: 1.2084477
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.41
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

Time for backsubstitution: 15.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1792549, upper bound: 1.1892982
time: 5.00 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1899209, upper bound: 1.1899166
time: 5.69 seconds

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

Time for backsubstitution: 14.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1792561, upper bound: 1.1892974
time: 5.55 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1899209, upper bound: 1.1899180
time: 6.55 seconds

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

Time for backsubstitution: 15.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 927

## Relational analysis of IS_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1892970, upper bound: 1.1853602
time: 5.34 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1899156, upper bound: 1.1960643
time: 5.83 seconds

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

Time for backsubstitution: 15.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1792496, upper bound: 1.1954255
time: 5.44 seconds

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

Time for backsubstitution: 15.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1916708, upper bound: 1.1892872
time: 5.52 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2022162, upper bound: 1.1899058
time: 5.39 seconds

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

Time for backsubstitution: 15.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1916708, upper bound: 1.1892879
time: 5.24 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2022162, upper bound: 1.1899059
time: 6.98 seconds

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

Time for backsubstitution: 14.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 927

## Relational analysis of IS_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2018576, upper bound: 1.1853495
time: 5.00 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2022116, upper bound: 1.1960541
time: 6.96 seconds

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

Time for backsubstitution: 15.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1916657, upper bound: 1.1954149
time: 5.31 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2022110, upper bound: 1.1960543
time: 7.78 seconds

## BFS IS instance: IS_A2_B1_B1_A1

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

Time for backsubstitution: 14.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 927

## Relational analysis of IS_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1892873, upper bound: 1.1916708
time: 5.37 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1899054, upper bound: 1.2022161
time: 5.60 seconds

## BFS IS instance: IS_A2_B1_B1_A2

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

Time for backsubstitution: 15.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 927

## Relational analysis of IS_A2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1892868, upper bound: 1.1978771
time: 5.25 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1899059, upper bound: 1.2084486
time: 6.11 seconds

## BFS IS instance: IS_A2_B1_B2_A1

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
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 5841
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1792395, upper bound: 1.2018596
time: 6.00 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1899053, upper bound: 1.2022114
time: 7.55 seconds

## BFS IS instance: IS_A2_B1_B2_A2

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

Time for backsubstitution: 14.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 927

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1892868, upper bound: 1.1916651
time: 5.82 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1899059, upper bound: 1.2022112
time: 6.17 seconds

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

Time for backsubstitution: 15.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105

Time for candidate selection: 0.22 seconds

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

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 26.38 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 26.38
Output dim: 2, lower bound: -1.1792549, upper bound: 1.1892982
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 26.38
Output dim: 2, lower bound: -1.1899209, upper bound: 1.1899166
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 26.38
Output dim: 2, lower bound: -1.1792561, upper bound: 1.1892974
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 26.38
Output dim: 2, lower bound: -1.1899209, upper bound: 1.1899180
IS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 26.38
Output dim: 2, lower bound: -1.1892970, upper bound: 1.1853602
IS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 26.38
Output dim: 2, lower bound: -1.1899156, upper bound: 1.1960643
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 26.38
Output dim: 2, lower bound: -1.1792496, upper bound: 1.1954255
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 26.38
Output dim: 2, lower bound: -1.1899157, upper bound: 1.1960647
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 26.38
Output dim: 2, lower bound: -1.1916708, upper bound: 1.1892872
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 26.38
Output dim: 2, lower bound: -1.2022162, upper bound: 1.1899058
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 26.38
Output dim: 2, lower bound: -1.1916708, upper bound: 1.1892879
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 26.38
Output dim: 2, lower bound: -1.2022162, upper bound: 1.1899059
IS_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 26.38
Output dim: 2, lower bound: -1.2018576, upper bound: 1.1853495
IS_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 26.38
Output dim: 2, lower bound: -1.2022116, upper bound: 1.1960541
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 26.38
Output dim: 2, lower bound: -1.1916657, upper bound: 1.1954149
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 26.38
Output dim: 2, lower bound: -1.2022110, upper bound: 1.1960543
IS_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 26.38
Output dim: 2, lower bound: -1.1892873, upper bound: 1.1916708
IS_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 26.38
Output dim: 2, lower bound: -1.1899054, upper bound: 1.2022161
IS_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 26.38
Output dim: 2, lower bound: -1.1892868, upper bound: 1.1978771
IS_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 26.38
Output dim: 2, lower bound: -1.1899059, upper bound: 1.2084486
IS_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 26.38
Output dim: 2, lower bound: -1.1792395, upper bound: 1.2018596
IS_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 26.38
Output dim: 2, lower bound: -1.1899053, upper bound: 1.2022114
IS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 26.38
Output dim: 2, lower bound: -1.1892868, upper bound: 1.1916651
IS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 26.38
Output dim: 2, lower bound: -1.1899059, upper bound: 1.2022112
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 26.38
Output dim: 2, lower bound: -1.1792446, upper bound: 1.2018558
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 26.38
Output dim: 2, lower bound: -1.1899107, upper bound: 1.2022087
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.38
Output dim: 2, lower bound: -1.1899120, upper bound: 1.2022157
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.38
Output dim: 2, lower bound: -1.1899120, upper bound: 1.2084477
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.38
Output dim: 2, lower bound: -1.1899123, upper bound: 1.2084502
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=1.9854326248168945
rel_dist={2: [-1.2085886744209708, 1.2085882328582862]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 5733
type: B, layer: 1, pos: 5733
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4625

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9483193, upper bound: 0.9355519
time: 5.35 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9483198, upper bound: 0.9483151
time: 5.42 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 11.00 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 11.00
Output dim: 2, lower bound: -0.9483193, upper bound: 0.9355519
IS_A2, status: Status.UNKNOWN, split count: 1, time: 11.00
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

Time for backsubstitution: 15.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 5733
type: A, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 4625

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9355534, upper bound: 0.9355514
time: 5.67 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9355535, upper bound: 0.9355507
time: 5.82 seconds

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

Time for backsubstitution: 14.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5733
type: B, layer: 1, pos: 5733
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5733

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9482871, upper bound: 0.9418774
time: 4.94 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9482862, upper bound: 0.9482806
time: 4.96 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 25.02 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 25.02
Output dim: 2, lower bound: -0.9355534, upper bound: 0.9355514
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 25.02
Output dim: 2, lower bound: -0.9355535, upper bound: 0.9355507
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 25.02
Output dim: 2, lower bound: -0.9482871, upper bound: 0.9418774
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 25.02
Output dim: 2, lower bound: -0.9482862, upper bound: 0.9482806

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

Time for backsubstitution: 15.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5733
type: B, layer: 1, pos: 5733
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5733

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9355264, upper bound: 0.9291166
time: 5.12 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9355248, upper bound: 0.9355168
time: 5.06 seconds

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

Time for backsubstitution: 14.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5733
type: B, layer: 1, pos: 5733
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5733

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9355264, upper bound: 0.9291170
time: 5.77 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9355248, upper bound: 0.9355166
time: 6.12 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -10.8340988, -8.2325096, -10.8108482, -8.2310753, -2.1728153, 2.1651716
1: -3.3540900, -0.8299637, -3.3503861, -0.9043264, -2.2441354, 2.2749729
2: 1.5642068, 3.6438155, 1.6547675, 3.6404538, -1.9000776, 1.8742266
3: -7.3151984, -5.2275357, -7.2597136, -5.2327118, -1.9878774, 1.9391897
4: -2.3625145, -0.4219861, -2.3514352, -0.4242320, -1.5600071, 1.5573051
5: -4.6311340, -2.6947653, -4.6279249, -2.7532957, -1.7845097, 1.8196564
6: -4.7210808, -2.1557038, -4.7157164, -2.2042964, -2.1937599, 2.1999950
7: -8.7282162, -6.8552046, -8.7202091, -6.8711395, -1.5037732, 1.5080395
8: -4.6630459, -2.3795109, -4.6485977, -2.4285889, -1.9766731, 2.0165384
9: -12.0843201, -9.7559795, -12.0749931, -9.7447491, -1.5927501, 1.5822389

Time for backsubstitution: 14.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4625

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9355208, upper bound: 0.9418782
time: 5.42 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9355208, upper bound: 0.9418802
time: 5.45 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -10.8895082, -8.2206650, -10.8108501, -8.2310867, -2.2045965, 2.1783094
1: -3.3763175, -0.7949233, -3.3503745, -0.9043264, -2.2729392, 2.2790670
2: 1.5120306, 3.6585739, 1.6547685, 3.6404355, -1.9078577, 1.8951142
3: -7.3234897, -5.2208452, -7.2597103, -5.2327156, -2.0006127, 1.9471600
4: -2.3938639, -0.3174448, -2.3513966, -0.4242325, -1.5966241, 1.5780520
5: -4.6403222, -2.6700008, -4.6279182, -2.7532969, -1.8203392, 1.8239956
6: -4.7553353, -2.0704236, -4.7156925, -2.2042973, -2.2275267, 2.2038748
7: -8.7329626, -6.8190675, -8.7201996, -6.8711429, -1.5131688, 1.5252054
8: -4.7145672, -2.3699040, -4.6485939, -2.4286008, -2.0272641, 2.0549164
9: -12.1782970, -9.7346039, -12.0749912, -9.7447891, -1.6130362, 1.6176715

Time for backsubstitution: 15.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 5733
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4625

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9355199, upper bound: 0.9482814
time: 5.17 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9355191, upper bound: 0.9482833
time: 8.33 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 28.72 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 28.72
Output dim: 2, lower bound: -0.9355264, upper bound: 0.9291166
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.72
Output dim: 2, lower bound: -0.9355248, upper bound: 0.9355168
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.72
Output dim: 2, lower bound: -0.9355264, upper bound: 0.9291170
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.72
Output dim: 2, lower bound: -0.9355248, upper bound: 0.9355166
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 28.72
Output dim: 2, lower bound: -0.9355208, upper bound: 0.9418782
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 28.72
Output dim: 2, lower bound: -0.9355208, upper bound: 0.9418802
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 28.72
Output dim: 2, lower bound: -0.9355199, upper bound: 0.9482814
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 28.72
Output dim: 2, lower bound: -0.9355191, upper bound: 0.9482833

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

Time for backsubstitution: 14.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5733

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9291245, upper bound: 0.9291223
time: 5.06 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9291237, upper bound: 0.9291218
time: 6.29 seconds

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

Time for backsubstitution: 15.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 5733
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 927

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9355039, upper bound: 0.9300051
time: 5.58 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9355039, upper bound: 0.9355032
time: 5.62 seconds

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

Time for backsubstitution: 14.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5733
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5733

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9418801, upper bound: 0.9291193
time: 5.75 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9418812, upper bound: 0.9291166
time: 6.33 seconds

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

Time for backsubstitution: 15.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 5733
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 927

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9482571, upper bound: 0.9299973
time: 5.35 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9482557, upper bound: 0.9354950
time: 5.55 seconds

## BFS IS instance: IS_A2_A1_B1

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

Time for backsubstitution: 14.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5733

## Relational analysis of IS_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9291220, upper bound: 0.9418782
time: 5.08 seconds

## Relational analysis of IS_A2_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9291219, upper bound: 0.9418778
time: 5.98 seconds

## BFS IS instance: IS_A2_A1_B2

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

Time for backsubstitution: 14.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5733

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9291219, upper bound: 0.9418801
time: 5.22 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9291220, upper bound: 0.9418795
time: 6.23 seconds

## BFS IS instance: IS_A2_A2_B1

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

Time for backsubstitution: 15.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 5733
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 927

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9354983, upper bound: 0.9427589
time: 5.13 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9354983, upper bound: 0.9482568
time: 5.38 seconds

## BFS IS instance: IS_A2_A2_B2

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

Time for backsubstitution: 15.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 5733
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 927

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9354982, upper bound: 0.9427605
time: 5.29 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9354983, upper bound: 0.9482584
time: 5.82 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 26.33 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.33
Output dim: 2, lower bound: -0.9291245, upper bound: 0.9291223
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.33
Output dim: 2, lower bound: -0.9291237, upper bound: 0.9291218
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.33
Output dim: 2, lower bound: -0.9355039, upper bound: 0.9300051
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.33
Output dim: 2, lower bound: -0.9355039, upper bound: 0.9355032
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.33
Output dim: 2, lower bound: -0.9418801, upper bound: 0.9291193
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.33
Output dim: 2, lower bound: -0.9418812, upper bound: 0.9291166
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.33
Output dim: 2, lower bound: -0.9482571, upper bound: 0.9299973
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.33
Output dim: 2, lower bound: -0.9482557, upper bound: 0.9354950
IS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 26.33
Output dim: 2, lower bound: -0.9291220, upper bound: 0.9418782
IS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 26.33
Output dim: 2, lower bound: -0.9291219, upper bound: 0.9418778
IS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 26.33
Output dim: 2, lower bound: -0.9291219, upper bound: 0.9418801
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 26.33
Output dim: 2, lower bound: -0.9291220, upper bound: 0.9418795
IS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 26.33
Output dim: 2, lower bound: -0.9354983, upper bound: 0.9427589
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 26.33
Output dim: 2, lower bound: -0.9354983, upper bound: 0.9482568
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 26.33
Output dim: 2, lower bound: -0.9354982, upper bound: 0.9427605
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 26.33
Output dim: 2, lower bound: -0.9354983, upper bound: 0.9482584

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

Time for backsubstitution: 15.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9236055, upper bound: 0.9290975
time: 5.65 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9291012, upper bound: 0.9290979
time: 5.95 seconds

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

Time for backsubstitution: 14.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9236055, upper bound: 0.9290983
time: 6.57 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9291012, upper bound: 0.9291007
time: 6.19 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -10.8630352, -8.2337971, -10.8026915, -8.2486391, -2.1649485, 2.1279020
1: -3.3529129, -0.8719759, -3.3335636, -0.9084094, -2.2362785, 2.2469873
2: 1.6110201, 3.6313517, 1.6706698, 3.6217496, -1.8677778, 1.8499205
3: -7.2430949, -5.2364383, -7.2093935, -5.2581315, -1.9068985, 1.8780162
4: -2.3637688, -0.3256850, -2.3458266, -0.4342990, -1.5538883, 1.5587204
5: -4.6156688, -2.7308497, -4.5969033, -2.7627027, -1.7843900, 1.7718949
6: -4.7276278, -2.1264453, -4.7017846, -2.2177587, -2.1631598, 2.1693072
7: -8.7209682, -6.8486948, -8.7122288, -6.8945026, -1.4815633, 1.4808588
8: -4.6892915, -2.4449949, -4.6246161, -2.4730108, -1.9608583, 1.9540763
9: -12.1669788, -9.7405767, -12.0724573, -9.7494450, -1.5976624, 1.5956821

Time for backsubstitution: 15.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5733
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5733

## Relational analysis of IS_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9291003, upper bound: 0.9300031
time: 6.66 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9290993, upper bound: 0.9300035
time: 8.68 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -10.8637486, -8.2271557, -10.8091240, -8.2324085, -2.1782780, 2.1510906
1: -3.3551614, -0.8717887, -3.3391783, -0.9065292, -2.2404008, 2.2514372
2: 1.6043987, 3.6315558, 1.6559470, 3.6252851, -1.8781686, 1.8563735
3: -7.2652812, -5.2359867, -7.2581644, -5.2407041, -1.9467359, 1.9122672
4: -2.3649111, -0.3222656, -2.3504655, -0.4264522, -1.5604761, 1.5675063
5: -4.6255479, -2.7302155, -4.6192694, -2.7543786, -1.8023739, 1.7920346
6: -4.7281003, -2.1204967, -4.7066545, -2.2050285, -2.1647315, 2.1798835
7: -8.7214098, -6.8390436, -8.7201357, -6.8740063, -1.4816568, 1.4982001
8: -4.6898065, -2.4254107, -4.6400342, -2.4291339, -1.9830637, 1.9889803
9: -12.1676550, -9.7390289, -12.0747070, -9.7458553, -1.6014488, 1.6008539

Time for backsubstitution: 15.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5733
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 5841
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5733

## Relational analysis of IS_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9291003, upper bound: 0.9355034
time: 5.41 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9291003, upper bound: 0.9355010
time: 5.68 seconds

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

Time for backsubstitution: 15.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9363567, upper bound: 0.9290926
time: 7.20 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9418545, upper bound: 0.9290950
time: 6.80 seconds

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

Time for backsubstitution: 15.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9363567, upper bound: 0.9290929
time: 6.10 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9418545, upper bound: 0.9290923
time: 5.59 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -10.8630352, -8.2337971, -10.8269615, -8.2421112, -2.1708694, 2.1566491
1: -3.3529129, -0.8719759, -3.3547153, -0.8320813, -2.2738843, 2.2675691
2: 1.6110201, 3.6313517, 1.5784667, 3.6486955, -1.8965492, 1.8795280
3: -7.2430949, -5.2364383, -7.2671251, -5.2430716, -1.9250875, 1.9350061
4: -2.3637688, -0.3256850, -2.3734074, -0.4294801, -1.5594611, 1.5744572
5: -4.6156688, -2.7308497, -4.6114092, -2.7026575, -1.7972817, 1.7877479
6: -4.7276278, -2.1264453, -4.7288380, -2.1677680, -2.1963239, 2.1993117
7: -8.7209682, -6.8486948, -8.7237587, -6.8758287, -1.4990306, 1.4837565
8: -4.6892915, -2.4449949, -4.6491451, -2.4175205, -2.0173230, 1.9752116
9: -12.1669788, -9.7405767, -12.0824947, -9.7450695, -1.6020691, 1.6076248

Time for backsubstitution: 15.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5733
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5733

## Relational analysis of IS_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9418529, upper bound: 0.9299995
time: 6.09 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9418537, upper bound: 0.9299995
time: 5.78 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -10.8637486, -8.2271557, -10.8333960, -8.2258854, -2.1842108, 2.1798277
1: -3.3551614, -0.8717887, -3.3603418, -0.8302047, -2.2780375, 2.2721436
2: 1.6043987, 3.6315558, 1.5637039, 3.6522288, -1.9069343, 1.8858876
3: -7.2652812, -5.2359867, -7.3159065, -5.2256584, -1.9648943, 1.9651763
4: -2.3649111, -0.3222656, -2.3782089, -0.4216318, -1.5660313, 1.5831940
5: -4.6255479, -2.7302155, -4.6337824, -2.6943357, -1.8147488, 1.8078997
6: -4.7281003, -2.1204967, -4.7336755, -2.1549973, -2.1987195, 2.2098761
7: -8.7214098, -6.8390436, -8.7316751, -6.8553114, -1.4991279, 1.5010706
8: -4.6898065, -2.4254107, -4.6645131, -2.3736379, -2.0394983, 2.0101011
9: -12.1676550, -9.7390289, -12.0847616, -9.7414742, -1.6058536, 1.6127734

Time for backsubstitution: 14.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5733
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5733

## Relational analysis of IS_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9418530, upper bound: 0.9354954
time: 7.00 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9418516, upper bound: 0.9354945
time: 8.06 seconds

## BFS IS instance: IS_A2_A1_B1_B1

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

Time for backsubstitution: 15.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 927

## Relational analysis of IS_A2_A1_B1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9290978, upper bound: 0.9363531
time: 12.22 seconds

## Relational analysis of IS_A2_A1_B1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9290978, upper bound: 0.9418508
time: 5.22 seconds

## BFS IS instance: IS_A2_A1_B1_B2

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

Time for backsubstitution: 14.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A2_A1_B1_B2_A1

### Relational analysis result of IS_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9235999, upper bound: 0.9418540
time: 7.09 seconds

## Relational analysis of IS_A2_A1_B1_B2_A2

### Relational analysis result of IS_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9290956, upper bound: 0.9418510
time: 9.78 seconds

## BFS IS instance: IS_A2_A1_B2_B1

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

Time for backsubstitution: 15.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 927

## Relational analysis of IS_A2_A1_B2_B1_B1

### Relational analysis result of IS_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9290977, upper bound: 0.9363526
time: 5.07 seconds

## Relational analysis of IS_A2_A1_B2_B1_B2

### Relational analysis result of IS_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9290978, upper bound: 0.9418506
time: 5.23 seconds

## BFS IS instance: IS_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -10.8342667, -8.2325087, -10.8896751, -8.2206631, -2.1937494, 2.2036440
1: -3.3540924, -0.8299019, -3.3763201, -0.7948638, -2.2816672, 2.2830496
2: 1.5641996, 3.6438277, 1.5120236, 3.6585867, -1.9202483, 1.9132085
3: -7.3152289, -5.2275262, -7.3235197, -5.2208347, -1.9684501, 1.9702692
4: -2.3626270, -0.4219837, -2.3939757, -0.3174410, -1.5762234, 1.5871718
5: -4.6311669, -2.6947558, -4.6403561, -2.6699915, -1.8260722, 1.8155971
6: -4.7211199, -2.1557031, -4.7553735, -2.0704236, -2.2131243, 2.2320020
7: -8.7282162, -6.8550620, -8.7329617, -6.8189249, -1.5245669, 1.5215626
8: -4.6630821, -2.3795116, -4.7146053, -2.3699040, -2.0293980, 2.0695913
9: -12.0843811, -9.7559776, -12.1783562, -9.7346020, -1.6114614, 1.6035767

Time for backsubstitution: 14.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 548

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of IS_A2_A1_B2_B2_A1

### Relational analysis result of IS_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9235999, upper bound: 0.9418527
time: 7.03 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2

### Relational analysis result of IS_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9290966, upper bound: 0.9418511
time: 7.19 seconds

## BFS IS instance: IS_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -10.8874283, -8.2273159, -10.8026915, -8.2486391, -2.1835384, 2.1383815
1: -3.3740520, -0.7956104, -3.3335636, -0.9084094, -2.2569466, 2.2607384
2: 1.5187333, 3.6582747, 1.6706698, 3.6217496, -1.8792768, 1.8785694
3: -7.3010497, -5.2213683, -7.2093935, -5.2581315, -1.9388549, 1.8962064
4: -2.3917043, -0.3208776, -2.3458266, -0.4342990, -1.5829940, 1.5641589
5: -4.6301765, -2.6707275, -4.5969033, -2.7627027, -1.8003263, 1.7927108
6: -4.7545762, -2.0763938, -4.7017846, -2.2177587, -2.1931252, 2.1784048
7: -8.7325153, -6.8299003, -8.7122288, -6.8945026, -1.4844184, 1.4981927
8: -4.7137723, -2.3894997, -4.6246161, -2.4730108, -1.9821463, 2.0105162
9: -12.1771088, -9.7361698, -12.0724573, -9.7494450, -1.6052842, 1.6000755

Time for backsubstitution: 15.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5733
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5733

## Relational analysis of IS_A2_A2_B1_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9290947, upper bound: 0.9427592
time: 6.12 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2

### Relational analysis result of IS_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9290947, upper bound: 0.9427574
time: 5.70 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 27.14 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 27.14
Output dim: 2, lower bound: -0.9236055, upper bound: 0.9290975
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 27.14
Output dim: 2, lower bound: -0.9291012, upper bound: 0.9290979
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 27.14
Output dim: 2, lower bound: -0.9236055, upper bound: 0.9290983
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 27.14
Output dim: 2, lower bound: -0.9291012, upper bound: 0.9291007
IS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 27.14
Output dim: 2, lower bound: -0.9291003, upper bound: 0.9300031
IS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 27.14
Output dim: 2, lower bound: -0.9290993, upper bound: 0.9300035
IS_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 27.14
Output dim: 2, lower bound: -0.9291003, upper bound: 0.9355034
IS_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 27.14
Output dim: 2, lower bound: -0.9291003, upper bound: 0.9355010
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 27.14
Output dim: 2, lower bound: -0.9363567, upper bound: 0.9290926
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 27.14
Output dim: 2, lower bound: -0.9418545, upper bound: 0.9290950
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 27.14
Output dim: 2, lower bound: -0.9363567, upper bound: 0.9290929
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 27.14
Output dim: 2, lower bound: -0.9418545, upper bound: 0.9290923
IS_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 27.14
Output dim: 2, lower bound: -0.9418529, upper bound: 0.9299995
IS_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 27.14
Output dim: 2, lower bound: -0.9418537, upper bound: 0.9299995
IS_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 27.14
Output dim: 2, lower bound: -0.9418530, upper bound: 0.9354954
IS_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 27.14
Output dim: 2, lower bound: -0.9418516, upper bound: 0.9354945
IS_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 27.14
Output dim: 2, lower bound: -0.9290978, upper bound: 0.9363531
IS_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 27.14
Output dim: 2, lower bound: -0.9290978, upper bound: 0.9418508
IS_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 27.14
Output dim: 2, lower bound: -0.9235999, upper bound: 0.9418540
IS_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 27.14
Output dim: 2, lower bound: -0.9290956, upper bound: 0.9418510
IS_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 27.14
Output dim: 2, lower bound: -0.9290977, upper bound: 0.9363526
IS_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 27.14
Output dim: 2, lower bound: -0.9290978, upper bound: 0.9418506
IS_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 27.14
Output dim: 2, lower bound: -0.9235999, upper bound: 0.9418527
IS_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 27.14
Output dim: 2, lower bound: -0.9290966, upper bound: 0.9418511
IS_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 27.14
Output dim: 2, lower bound: -0.9290947, upper bound: 0.9427592
IS_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 27.14
Output dim: 2, lower bound: -0.9290947, upper bound: 0.9427574
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 27.14
Output dim: 2, lower bound: -0.9354983, upper bound: 0.9482568
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 27.14
Output dim: 2, lower bound: -0.9354982, upper bound: 0.9427605
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 27.14
Output dim: 2, lower bound: -0.9354983, upper bound: 0.9482584
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=1.8685765266418457
rel_dist={2: [-0.9484739266423019, 0.9484713204465995]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 5733
type: B, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 5841
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4625

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8355526, upper bound: 0.8258965
time: 5.26 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8355526, upper bound: 0.8355513
time: 5.02 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 10.51 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 10.51
Output dim: 2, lower bound: -0.8355526, upper bound: 0.8258965
IS_A2, status: Status.UNKNOWN, split count: 1, time: 10.51
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

Time for backsubstitution: 14.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 5733
type: A, layer: 1, pos: 5733
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4625

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8258969, upper bound: 0.8258940
time: 6.93 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8258969, upper bound: 0.8258954
time: 9.71 seconds

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

Time for backsubstitution: 14.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5733
type: B, layer: 1, pos: 5733
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5733

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8355199, upper bound: 0.8290390
time: 5.29 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8355197, upper bound: 0.8355174
time: 5.24 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 25.71 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 25.71
Output dim: 2, lower bound: -0.8258969, upper bound: 0.8258940
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 25.71
Output dim: 2, lower bound: -0.8258969, upper bound: 0.8258954
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 25.71
Output dim: 2, lower bound: -0.8355199, upper bound: 0.8290390
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 25.71
Output dim: 2, lower bound: -0.8355197, upper bound: 0.8355174

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -10.8327579, -8.2325172, -10.8108492, -8.2310753, -2.0985556, 2.0915918
1: -3.3540699, -0.8304555, -3.3503830, -0.9043274, -2.1784911, 2.2095582
2: 1.5642707, 3.6437125, 1.6547688, 3.6404383, -1.8596139, 1.8325758
3: -7.3149619, -5.2276163, -7.2597122, -5.2327185, -1.9403701, 1.8908184
4: -2.3616152, -0.4220018, -2.3514342, -0.4242325, -1.5046446, 1.5020437
5: -4.6308694, -2.6948390, -4.6279144, -2.7532964, -1.7478514, 1.7824030
6: -4.7207742, -2.1557045, -4.7157078, -2.2042971, -2.1309757, 2.1387565
7: -8.7282162, -6.8563538, -8.7202091, -6.8711433, -1.4599876, 1.4645426
8: -4.6627455, -2.3795128, -4.6485896, -2.4285898, -1.9106522, 1.9509146
9: -12.0838318, -9.7559938, -12.0749931, -9.7447510, -1.5119138, 1.5014086

Time for backsubstitution: 14.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5733
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5733

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8290402, upper bound: 0.8290404
time: 6.10 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8290402, upper bound: 0.8290380
time: 9.56 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -10.8881626, -8.2206697, -10.8108463, -8.2310905, -2.1300745, 2.1033273
1: -3.3762982, -0.7954175, -3.3503692, -0.9043295, -2.2056823, 2.2136528
2: 1.5120946, 3.6584725, 1.6547698, 3.6404152, -1.8673930, 1.8514547
3: -7.3232536, -5.2209253, -7.2597084, -5.2327232, -1.9517591, 1.8987861
4: -2.3929644, -0.3174610, -2.3513870, -0.4242349, -1.5372883, 1.5215154
5: -4.6400576, -2.6700749, -4.6279063, -2.7532983, -1.7823157, 1.7867386
6: -4.7550354, -2.0704260, -4.7156792, -2.2042985, -2.1618643, 2.1426342
7: -8.7329617, -6.8202181, -8.7201958, -6.8711462, -1.4677656, 1.4815370
8: -4.7142663, -2.3699038, -4.6485844, -2.4286067, -1.9612412, 1.9864752
9: -12.1778069, -9.7346182, -12.0749903, -9.7448025, -1.5310309, 1.5323789

Time for backsubstitution: 14.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 5733
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4625

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8258640, upper bound: 0.8355145
time: 4.79 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8258640, upper bound: 0.8355175
time: 5.68 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 25.60 seconds
IS_A2_A1_B1, status: Status.VERIFIED, split count: 3, time: 25.60
Output dim: 2, lower bound: -0.8290402, upper bound: 0.8290404
IS_A2_A1_B2, status: Status.VERIFIED, split count: 3, time: 25.60
Output dim: 2, lower bound: -0.8290402, upper bound: 0.8290380
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 25.60
Output dim: 2, lower bound: -0.8258640, upper bound: 0.8355145
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 25.60
Output dim: 2, lower bound: -0.8258640, upper bound: 0.8355175

## BFS IS instance: IS_A2_A2_B1

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

Time for backsubstitution: 14.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 5733
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 927

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8258450, upper bound: 0.8315054
time: 7.31 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8258450, upper bound: 0.8354766
time: 7.23 seconds

## BFS IS instance: IS_A2_A2_B2

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

Time for backsubstitution: 15.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 5733
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 927

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8258450, upper bound: 0.8315092
time: 6.01 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8258450, upper bound: 0.8354810
time: 6.08 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 27.38 seconds
IS_A2_A2_B1_B1, status: Status.VERIFIED, split count: 4, time: 27.38
Output dim: 2, lower bound: -0.8258450, upper bound: 0.8315054
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 27.38
Output dim: 2, lower bound: -0.8258450, upper bound: 0.8354766
IS_A2_A2_B2_B1, status: Status.VERIFIED, split count: 4, time: 27.38
Output dim: 2, lower bound: -0.8258450, upper bound: 0.8315092
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 27.38
Output dim: 2, lower bound: -0.8258450, upper bound: 0.8354810

## BFS IS instance: IS_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -10.8866825, -8.2206821, -10.8091211, -8.2324104, -2.1212811, 2.0871119
1: -3.3762746, -0.7959619, -3.3391781, -0.9065304, -2.1965442, 2.1995916
2: 1.5121660, 3.6583605, 1.6559459, 3.6252816, -1.8491960, 1.8433666
3: -7.3229775, -5.2210131, -7.2581635, -5.2407041, -1.9271331, 1.8830032
4: -2.3919730, -0.3174796, -2.3504572, -0.4264522, -1.5301058, 1.5167849
5: -4.6397600, -2.6701581, -4.6192684, -2.7543788, -1.7778969, 1.7723646
6: -4.7547035, -2.0704308, -4.7066488, -2.2050276, -2.1308312, 2.1277473
7: -8.7329617, -6.8214922, -8.7201328, -6.8740072, -1.4385586, 1.4718314
8: -4.7139325, -2.3699126, -4.6400323, -2.4291372, -1.9374509, 1.9769816
9: -12.1772671, -9.7346344, -12.0747070, -9.7458649, -1.5268934, 1.5204310

Time for backsubstitution: 15.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5733
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5733

## Relational analysis of IS_A2_A2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8193601, upper bound: 0.8354794
time: 13.73 seconds

## Relational analysis of IS_A2_A2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8193601, upper bound: 0.8354776
time: 5.88 seconds

## BFS IS instance: IS_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -10.8896751, -8.2206678, -10.8349199, -8.2258806, -2.1327691, 2.1265664
1: -3.3763187, -0.7948623, -3.3603618, -0.8296483, -2.2250338, 2.2202516
2: 1.5120273, 3.6585860, 1.5636326, 3.6523421, -1.8767362, 1.8704734
3: -7.3235064, -5.2208352, -7.3161769, -5.2255673, -1.9424911, 1.9077206
4: -2.3939750, -0.3174436, -2.3792233, -0.4216137, -1.5304878, 1.5344732
5: -4.6403503, -2.6699915, -4.6340818, -2.6942508, -1.7910099, 1.7872229
6: -4.7553759, -2.0704281, -4.7340188, -2.1549973, -2.1583247, 2.1599607
7: -8.7329617, -6.8189297, -8.7316723, -6.8540049, -1.4599254, 1.4790390
8: -4.7146053, -2.3699107, -4.6648550, -2.3736377, -1.9892340, 1.9932833
9: -12.1783552, -9.7346029, -12.0853195, -9.7414665, -1.5344903, 1.5399399

Time for backsubstitution: 14.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5733
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5733

## Relational analysis of IS_A2_A2_B2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8193601, upper bound: 0.8354783
time: 12.74 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8193601, upper bound: 0.8354791
time: 5.72 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 33.66 seconds
IS_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 33.66
Output dim: 2, lower bound: -0.8193601, upper bound: 0.8354794
IS_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 33.66
Output dim: 2, lower bound: -0.8193601, upper bound: 0.8354776
IS_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 33.66
Output dim: 2, lower bound: -0.8193601, upper bound: 0.8354783
IS_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 33.66
Output dim: 2, lower bound: -0.8193601, upper bound: 0.8354791

## BFS IS instance: IS_A2_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -10.8866825, -8.2206821, -10.8084745, -8.2390385, -2.1146936, 2.0863461
1: -3.3762746, -0.7959619, -3.3329046, -0.9067824, -2.1889272, 2.1921463
2: 1.5121660, 3.6583605, 1.6565157, 3.6167743, -1.8405142, 1.8387351
3: -7.3229775, -5.2210131, -7.2571974, -5.2426596, -1.9208345, 1.8802652
4: -2.3919730, -0.3174796, -2.3337803, -0.4268193, -1.5288123, 1.4999449
5: -4.6397600, -2.6701581, -4.6163435, -2.7548745, -1.7556043, 1.7668109
6: -4.7547035, -2.0704308, -4.6937361, -2.2057414, -2.1298223, 2.1140032
7: -8.7329617, -6.8214922, -8.7166748, -6.8750753, -1.4342926, 1.4666815
8: -4.7139325, -2.3699126, -4.6383028, -2.4350250, -1.9288502, 1.9473138
9: -12.1772671, -9.7346344, -12.0737734, -9.7603798, -1.5126741, 1.5111725

Time for backsubstitution: 14.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4632

## Relational analysis of IS_A2_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8193493, upper bound: 0.8343485
time: 5.78 seconds

## Relational analysis of IS_A2_A2_B1_B2_B1_A2

### Relational analysis result of IS_A2_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8193583, upper bound: 0.8354747
time: 14.15 seconds

## BFS IS instance: IS_A2_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -10.8866825, -8.2206821, -10.8637457, -8.2271624, -2.1234760, 2.1131225
1: -3.3762746, -0.7959619, -3.3551610, -0.8717880, -2.2003708, 2.2138615
2: 1.5121660, 3.6583605, 1.6044031, 3.6315556, -1.8557782, 1.8527789
3: -7.3229775, -5.2210131, -7.2652631, -5.2359872, -1.9295468, 1.9045701
4: -2.3919730, -0.3174796, -2.3649106, -0.3222680, -1.5407057, 1.5302701
5: -4.6397600, -2.6701581, -4.6255398, -2.7302177, -1.7813578, 1.7804568
6: -4.7547035, -2.0704308, -4.7281008, -2.1205020, -2.1552610, 2.1447592
7: -8.7329617, -6.8214922, -8.7214088, -6.8390546, -1.4459770, 1.4752187
8: -4.7139325, -2.3699126, -4.6898074, -2.4254210, -1.9233747, 1.9821837
9: -12.1772671, -9.7346344, -12.1676540, -9.7390270, -1.5300279, 1.5241132

Time for backsubstitution: 14.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4632

## Relational analysis of IS_A2_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8193493, upper bound: 0.8343495
time: 5.19 seconds

## Relational analysis of IS_A2_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8193583, upper bound: 0.8354757
time: 5.85 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -10.8896751, -8.2206678, -10.8342657, -8.2325153, -2.1261826, 2.1257825
1: -3.3763187, -0.7948623, -3.3540893, -0.8299017, -2.2174273, 2.2128167
2: 1.5120273, 3.6585860, 1.5642052, 3.6438279, -1.8680465, 1.8691189
3: -7.3235064, -5.2208352, -7.3152032, -5.2275257, -1.9221740, 1.9049745
4: -2.3939750, -0.3174436, -2.3626246, -0.4219871, -1.5291828, 1.5176787
5: -4.6403503, -2.6699915, -4.6311531, -2.6947579, -1.7791996, 1.7816579
6: -4.7553759, -2.0704281, -4.7211189, -2.1557121, -2.1573167, 2.1462464
7: -8.7329617, -6.8189297, -8.7282143, -6.8550768, -1.4556677, 1.4738890
8: -4.7146053, -2.3699107, -4.6630812, -2.3795266, -1.9806428, 1.9636238
9: -12.1783552, -9.7346029, -12.0843792, -9.7559814, -1.5202789, 1.5306892

Time for backsubstitution: 14.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4632

## Relational analysis of IS_A2_A2_B2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8193493, upper bound: 0.8343502
time: 5.30 seconds

## Relational analysis of IS_A2_A2_B2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8193583, upper bound: 0.8354767
time: 12.78 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -10.8896751, -8.2206678, -10.8896751, -8.2206717, -2.1377034, 2.1404212
1: -3.3763187, -0.7948623, -3.3763158, -0.7948618, -2.2288437, 2.2275982
2: 1.5120273, 3.6585860, 1.5120302, 3.6585872, -1.8833175, 1.8773460
3: -7.3235064, -5.2208352, -7.3234940, -5.2208357, -1.9449215, 1.9295421
4: -2.3939750, -0.3174436, -2.3939738, -0.3174472, -1.5484879, 1.5497949
5: -4.6403503, -2.6699915, -4.6403446, -2.6699924, -1.7944379, 1.7953477
6: -4.7553759, -2.0704281, -4.7553740, -2.0704331, -2.1691494, 2.1769614
7: -8.7329617, -6.8189297, -8.7329626, -6.8189392, -1.4671774, 1.4823952
8: -4.7146053, -2.3699107, -4.7146058, -2.3699193, -1.9751644, 1.9983373
9: -12.1783552, -9.7346029, -12.1783552, -9.7346039, -1.5424681, 1.5435476

Time for backsubstitution: 15.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 5841
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4632

## Relational analysis of IS_A2_A2_B2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8193493, upper bound: 0.8343506
time: 5.23 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8193583, upper bound: 0.8354774
time: 6.23 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 26.69 seconds
IS_A2_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 26.69
Output dim: 2, lower bound: -0.8193493, upper bound: 0.8343485
IS_A2_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 26.69
Output dim: 2, lower bound: -0.8193583, upper bound: 0.8354747
IS_A2_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 26.69
Output dim: 2, lower bound: -0.8193493, upper bound: 0.8343495
IS_A2_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 26.69
Output dim: 2, lower bound: -0.8193583, upper bound: 0.8354757
IS_A2_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 26.69
Output dim: 2, lower bound: -0.8193493, upper bound: 0.8343502
IS_A2_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 26.69
Output dim: 2, lower bound: -0.8193583, upper bound: 0.8354767
IS_A2_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 26.69
Output dim: 2, lower bound: -0.8193493, upper bound: 0.8343506
IS_A2_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 26.69
Output dim: 2, lower bound: -0.8193583, upper bound: 0.8354774

## BFS IS instance: IS_A2_A2_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -10.8812523, -8.2212400, -10.8073921, -8.2392263, -2.1090965, 2.0850053
1: -3.3743391, -0.7980849, -3.3320377, -0.9069912, -2.1866279, 2.1889825
2: 1.5204434, 3.6558864, 1.6577079, 3.6153502, -1.8305545, 1.8353286
3: -7.3194089, -5.2240009, -7.2556639, -5.2433929, -1.9165697, 1.8757749
4: -2.3892426, -0.3206687, -2.3325005, -0.4273496, -1.5255631, 1.4951749
5: -4.6346297, -2.6714976, -4.6155300, -2.7555380, -1.7494621, 1.7646492
6: -4.7528963, -2.0712190, -4.6933537, -2.2058740, -2.1277847, 2.1128078
7: -8.7293739, -6.8270416, -8.7145481, -6.8756981, -1.4302576, 1.4584943
8: -4.7103305, -2.3717399, -4.6376610, -2.4359627, -1.9243379, 1.9448671
9: -12.1694393, -9.7437096, -12.0691051, -9.7621918, -1.5044434, 1.4984164

Time for backsubstitution: 15.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6111

## Relational analysis of IS_A2_A2_B1_B2_B1_A1_A1

### Relational analysis result of IS_A2_A2_B1_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8179424, upper bound: 0.8267267
time: 5.07 seconds

## Relational analysis of IS_A2_A2_B1_B2_B1_A1_A2

### Relational analysis result of IS_A2_A2_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8193423, upper bound: 0.8343437
time: 5.45 seconds

## BFS IS instance: IS_A2_A2_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -10.8866806, -8.2206831, -10.8084736, -8.2390404, -2.1125908, 2.0861325
1: -3.3762743, -0.7959615, -3.3329034, -0.9067841, -2.1873465, 2.1909275
2: 1.5121677, 3.6583569, 1.6565173, 3.6167731, -1.8366532, 1.8378055
3: -7.3229747, -5.2210159, -7.2571950, -5.2426586, -1.9198751, 1.8802636
4: -2.3919709, -0.3174801, -2.3337784, -0.4268193, -1.5284076, 1.4985018
5: -4.6397600, -2.6701598, -4.6163430, -2.7548749, -1.7559090, 1.7651927
6: -4.7547035, -2.0704308, -4.6937361, -2.2057419, -2.1297436, 2.1135573
7: -8.7329569, -6.8214912, -8.7166719, -6.8750763, -1.4302194, 1.4635772
8: -4.7139316, -2.3699155, -4.6383014, -2.4350257, -1.9288497, 1.9465971
9: -12.1772594, -9.7346382, -12.0737686, -9.7603798, -1.5070939, 1.5111682

Time for backsubstitution: 14.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 5841
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6111

## Relational analysis of IS_A2_A2_B1_B2_B1_A2_A1

### Relational analysis result of IS_A2_A2_B1_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8179499, upper bound: 0.8278140
time: 5.30 seconds

## Relational analysis of IS_A2_A2_B1_B2_B1_A2_A2

### Relational analysis result of IS_A2_A2_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8193517, upper bound: 0.8354707
time: 5.48 seconds

## BFS IS instance: IS_A2_A2_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -10.8812523, -8.2212400, -10.8626661, -8.2273512, -2.1179495, 2.1117811
1: -3.3743391, -0.7980849, -3.3542948, -0.8719964, -2.1980534, 2.2106962
2: 1.5204434, 3.6558864, 1.6055924, 3.6301339, -1.8458204, 1.8493726
3: -7.3194089, -5.2240009, -7.2637515, -5.2367210, -1.9252825, 1.9001064
4: -2.3892426, -0.3206687, -2.3636301, -0.3227992, -1.5374479, 1.5258400
5: -4.6346297, -2.6714976, -4.6247387, -2.7308815, -1.7752094, 1.7782702
6: -4.7528963, -2.0712190, -4.7277069, -2.1206353, -2.1532240, 2.1435511
7: -8.7293739, -6.8270416, -8.7192822, -6.8396778, -1.4419236, 1.4670317
8: -4.7103305, -2.3717399, -4.6891627, -2.4263573, -1.9188595, 1.9797187
9: -12.1694393, -9.7437096, -12.1629829, -9.7408438, -1.5218129, 1.5113609

Time for backsubstitution: 14.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6111

## Relational analysis of IS_A2_A2_B1_B2_B2_A1_A1

### Relational analysis result of IS_A2_A2_B1_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8179424, upper bound: 0.8267269
time: 6.27 seconds

## Relational analysis of IS_A2_A2_B1_B2_B2_A1_A2

### Relational analysis result of IS_A2_A2_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8202284, upper bound: 0.8343447
time: 5.88 seconds

## BFS IS instance: IS_A2_A2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -10.8866806, -8.2206831, -10.8637466, -8.2271624, -2.1234760, 2.1129088
1: -3.3762743, -0.7959615, -3.3551590, -0.8717895, -2.1987901, 2.2126431
2: 1.5121677, 3.6583569, 1.6044037, 3.6315541, -1.8519173, 1.8518496
3: -7.3229747, -5.2210159, -7.2652617, -5.2359877, -1.9285879, 1.9045682
4: -2.3919709, -0.3174801, -2.3649092, -0.3222694, -1.5394204, 1.5302682
5: -4.6397600, -2.6701598, -4.6255398, -2.7302179, -1.7793479, 1.7788105
6: -4.7547035, -2.0704308, -4.7280998, -2.1205020, -2.1551819, 2.1443014
7: -8.7329569, -6.8214912, -8.7214069, -6.8390555, -1.4418862, 1.4721143
8: -4.7139316, -2.3699155, -4.6898055, -2.4254212, -1.9233718, 1.9814677
9: -12.1772594, -9.7346382, -12.1676493, -9.7390308, -1.5268471, 1.5241079

Time for backsubstitution: 15.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 5841
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6111

## Relational analysis of IS_A2_A2_B1_B2_B2_A2_A1

### Relational analysis result of IS_A2_A2_B1_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8187969, upper bound: 0.8278137
time: 6.04 seconds

## Relational analysis of IS_A2_A2_B1_B2_B2_A2_A2

### Relational analysis result of IS_A2_A2_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8202301, upper bound: 0.8354691
time: 7.52 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -10.8842421, -8.2212248, -10.8331833, -8.2327070, -2.1205826, 2.1244154
1: -3.3743796, -0.7969872, -3.3532214, -0.8301120, -2.2151289, 2.2096531
2: 1.5203041, 3.6561129, 1.5654044, 3.6424048, -1.8580918, 1.8657115
3: -7.3199439, -5.2238207, -7.3136845, -5.2282605, -1.9179177, 1.9005153
4: -2.3912454, -0.3206332, -2.3613486, -0.4225240, -1.5259147, 1.5129085
5: -4.6352186, -2.6713309, -4.6303282, -2.6954165, -1.7730608, 1.7794843
6: -4.7535658, -2.0712154, -4.7207136, -2.1558461, -2.1552763, 2.1450417
7: -8.7293758, -6.8244834, -8.7260876, -6.8556900, -1.4516168, 1.4657010
8: -4.7110023, -2.3717370, -4.6624289, -2.3804638, -1.9761300, 1.9611764
9: -12.1705265, -9.7436752, -12.0797110, -9.7578030, -1.5120423, 1.5179360

Time for backsubstitution: 14.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6111

## Relational analysis of IS_A2_A2_B2_B2_B1_A1_B1

### Relational analysis result of IS_A2_A2_B2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8116385, upper bound: 0.8330293
time: 6.18 seconds

## Relational analysis of IS_A2_A2_B2_B2_B1_A1_B2

### Relational analysis result of IS_A2_A2_B2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8193422, upper bound: 0.8343487
time: 5.64 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -10.8896751, -8.2206669, -10.8342667, -8.2325172, -2.1240773, 2.1255679
1: -3.3763161, -0.7948633, -3.3540888, -0.8299034, -2.2158465, 2.2115970
2: 1.5120277, 3.6585839, 1.5642065, 3.6438260, -1.8641846, 1.8658051
3: -7.3235059, -5.2208366, -7.3151994, -5.2275271, -1.9221621, 1.9049721
4: -2.3939726, -0.3174431, -2.3626237, -0.4219880, -1.5287778, 1.5162356
5: -4.6403494, -2.6699932, -4.6311522, -2.6947577, -1.7795033, 1.7800274
6: -4.7553735, -2.0704284, -4.7211185, -2.1557126, -2.1572385, 2.1457715
7: -8.7329578, -6.8189311, -8.7282133, -6.8550777, -1.4515729, 1.4707828
8: -4.7146044, -2.3699121, -4.6630821, -2.3795261, -1.9806409, 1.9629066
9: -12.1783466, -9.7346039, -12.0843735, -9.7559814, -1.5146923, 1.5306854

Time for backsubstitution: 15.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6111

## Relational analysis of IS_A2_A2_B2_B2_B1_A2_A1

### Relational analysis result of IS_A2_A2_B2_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8179499, upper bound: 0.8278182
time: 6.76 seconds

## Relational analysis of IS_A2_A2_B2_B2_B1_A2_A2

### Relational analysis result of IS_A2_A2_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8193521, upper bound: 0.8354723
time: 5.57 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -10.8842421, -8.2212248, -10.8885918, -8.2208605, -2.1321020, 2.1389656
1: -3.3743796, -0.7969872, -3.3754506, -0.7950699, -2.2265291, 2.2244768
2: 1.5203041, 3.6561129, 1.5132271, 3.6571629, -1.8733625, 1.8738911
3: -7.3199439, -5.2238207, -7.3219991, -5.2215695, -1.9406652, 1.9251084
4: -2.3912454, -0.3206332, -2.3926983, -0.3179805, -1.5452089, 1.5450230
5: -4.6352186, -2.6713309, -4.6395292, -2.6706524, -1.7882919, 1.7931840
6: -4.7535658, -2.0712154, -4.7549596, -2.0705667, -2.1671085, 2.1757450
7: -8.7293758, -6.8244834, -8.7308359, -6.8195524, -1.4631090, 1.4742073
8: -4.7110023, -2.3717370, -4.7139511, -2.3708551, -1.9706492, 1.9958725
9: -12.1705265, -9.7436752, -12.1736870, -9.7364254, -1.5342557, 1.5308001

Time for backsubstitution: 15.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6111

## Relational analysis of IS_A2_A2_B2_B2_B2_A1_A1

### Relational analysis result of IS_A2_A2_B2_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8179423, upper bound: 0.8267292
time: 5.57 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2_A1_A2

### Relational analysis result of IS_A2_A2_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8202288, upper bound: 0.8343477
time: 7.53 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -10.8896751, -8.2206669, -10.8896723, -8.2206707, -2.1355977, 2.1387815
1: -3.3763161, -0.7948633, -3.3763146, -0.7948632, -2.2272635, 2.2275972
2: 1.5120277, 3.6585839, 1.5120313, 3.6585851, -1.8794560, 1.8739922
3: -7.3235059, -5.2208366, -7.3234925, -5.2208362, -1.9449091, 1.9295402
4: -2.3939726, -0.3174431, -2.3939734, -0.3174467, -1.5471976, 1.5483515
5: -4.6403494, -2.6699932, -4.6403446, -2.6699929, -1.7924266, 1.7936893
6: -4.7553735, -2.0704284, -4.7553725, -2.0704331, -2.1686659, 2.1764755
7: -8.7329578, -6.8189311, -8.7329597, -6.8189392, -1.4630659, 1.4792888
8: -4.7146044, -2.3699121, -4.7146044, -2.3699198, -1.9751625, 1.9976208
9: -12.1783466, -9.7346039, -12.1783504, -9.7346039, -1.5392866, 1.5406671

Time for backsubstitution: 14.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6111

## Relational analysis of IS_A2_A2_B2_B2_B2_A2_A1

### Relational analysis result of IS_A2_A2_B2_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8179499, upper bound: 0.8278167
time: 5.63 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2_A2_A2

### Relational analysis result of IS_A2_A2_B2_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8202302, upper bound: 0.8250154
time: 8.24 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 29.02 seconds
IS_A2_A2_B1_B2_B1_A1_A1, status: Status.VERIFIED, split count: 7, time: 29.02
Output dim: 2, lower bound: -0.8179424, upper bound: 0.8267267
IS_A2_A2_B1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 29.02
Output dim: 2, lower bound: -0.8193423, upper bound: 0.8343437
IS_A2_A2_B1_B2_B1_A2_A1, status: Status.VERIFIED, split count: 7, time: 29.02
Output dim: 2, lower bound: -0.8179499, upper bound: 0.8278140
IS_A2_A2_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 29.02
Output dim: 2, lower bound: -0.8193517, upper bound: 0.8354707
IS_A2_A2_B1_B2_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 29.02
Output dim: 2, lower bound: -0.8179424, upper bound: 0.8267269
IS_A2_A2_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 29.02
Output dim: 2, lower bound: -0.8202284, upper bound: 0.8343447
IS_A2_A2_B1_B2_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 29.02
Output dim: 2, lower bound: -0.8187969, upper bound: 0.8278137
IS_A2_A2_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 29.02
Output dim: 2, lower bound: -0.8202301, upper bound: 0.8354691
IS_A2_A2_B2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 29.02
Output dim: 2, lower bound: -0.8116385, upper bound: 0.8330293
IS_A2_A2_B2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 29.02
Output dim: 2, lower bound: -0.8193422, upper bound: 0.8343487
IS_A2_A2_B2_B2_B1_A2_A1, status: Status.VERIFIED, split count: 7, time: 29.02
Output dim: 2, lower bound: -0.8179499, upper bound: 0.8278182
IS_A2_A2_B2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 29.02
Output dim: 2, lower bound: -0.8193521, upper bound: 0.8354723
IS_A2_A2_B2_B2_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 29.02
Output dim: 2, lower bound: -0.8179423, upper bound: 0.8267292
IS_A2_A2_B2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 29.02
Output dim: 2, lower bound: -0.8202288, upper bound: 0.8343477
IS_A2_A2_B2_B2_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 29.02
Output dim: 2, lower bound: -0.8179499, upper bound: 0.8278167
IS_A2_A2_B2_B2_B2_A2_A2, status: Status.VERIFIED, split count: 7, time: 29.02
Output dim: 2, lower bound: -0.8202302, upper bound: 0.8250154

## BFS IS instance: IS_A2_A2_B1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -10.8812523, -8.2212410, -10.8073921, -8.2392263, -2.1121297, 2.0850048
1: -3.3743384, -0.7980865, -3.3320377, -0.9069912, -2.1981425, 2.1889818
2: 1.5204443, 3.6558843, 1.6577079, 3.6153502, -1.8279538, 1.8012781
3: -7.3194094, -5.2240019, -7.2556639, -5.2433929, -1.9316320, 1.8757739
4: -2.3892426, -0.3206697, -2.3325005, -0.4273496, -1.5255622, 1.4834442
5: -4.6346288, -2.6714981, -4.6155300, -2.7555380, -1.7449231, 1.7638297
6: -4.7528963, -2.0712180, -4.6933537, -2.2058740, -2.1291318, 2.1128073
7: -8.7293730, -6.8270426, -8.7145481, -6.8756981, -1.4178901, 1.4561259
8: -4.7103314, -2.3717394, -4.6376610, -2.4359627, -1.9217548, 1.9448676
9: -12.1694355, -9.7437096, -12.0691051, -9.7621918, -1.4440885, 1.4984157

Time for backsubstitution: 15.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 5841
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 6111

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 904

## Relational analysis of IS_A2_A2_B1_B2_B1_A1_A2_B1

### Relational analysis result of IS_A2_A2_B1_B2_B1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8185134, upper bound: 0.8298489
time: 5.65 seconds

## Relational analysis of IS_A2_A2_B1_B2_B1_A1_A2_B2

### Relational analysis result of IS_A2_A2_B1_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8193388, upper bound: 0.8343405
time: 5.17 seconds

## BFS IS instance: IS_A2_A2_B1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -10.8866796, -8.2206841, -10.8084736, -8.2390404, -2.1156254, 2.0861316
1: -3.3762732, -0.7959623, -3.3329034, -0.9067841, -2.1988626, 2.1909280
2: 1.5121684, 3.6583548, 1.6565173, 3.6167731, -1.8340549, 1.8037543
3: -7.3229761, -5.2210159, -7.2571950, -5.2426586, -1.9349384, 1.8802621
4: -2.3919699, -0.3174829, -2.3337784, -0.4268193, -1.5284071, 1.4867713
5: -4.6397591, -2.6701603, -4.6163430, -2.7548749, -1.7513571, 1.7643726
6: -4.7547016, -2.0704324, -4.6937361, -2.2057419, -2.1310902, 2.1135564
7: -8.7329559, -6.8214931, -8.7166719, -6.8750763, -1.4178751, 1.4612094
8: -4.7139325, -2.3699150, -4.6383014, -2.4350257, -1.9262600, 1.9465971
9: -12.1772575, -9.7346382, -12.0737686, -9.7603798, -1.4467385, 1.5111678

Time for backsubstitution: 15.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 5841
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 6111

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5841

## Relational analysis of IS_A2_A2_B1_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 904

## Relational analysis of IS_A2_A2_B1_B2_B1_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1_B2_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8185485, upper bound: 0.8310020
time: 5.55 seconds

## Relational analysis of IS_A2_A2_B1_B2_B1_A2_A2_B2

### Relational analysis result of IS_A2_A2_B1_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8193486, upper bound: 0.8354678
time: 7.11 seconds

## BFS IS instance: IS_A2_A2_B1_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -10.8812523, -8.2212410, -10.8626661, -8.2273512, -2.1222539, 2.1117802
1: -3.3743384, -0.7980865, -3.3542948, -0.8719964, -2.2095771, 2.2106962
2: 1.5204443, 3.6558843, 1.6055924, 3.6301339, -1.8432198, 1.8153350
3: -7.3194094, -5.2240019, -7.2637515, -5.2367210, -1.9403443, 1.9001060
4: -2.3892426, -0.3206697, -2.3636301, -0.3227992, -1.5366306, 1.5141319
5: -4.6346288, -2.6714981, -4.6247387, -2.7308815, -1.7703300, 1.7774508
6: -4.7528963, -2.0712180, -4.7277069, -2.1206353, -2.1545696, 2.1435506
7: -8.7293730, -6.8270426, -8.7192822, -6.8396778, -1.4295945, 1.4646636
8: -4.7103314, -2.3717394, -4.6891627, -2.4263573, -1.9162741, 1.9797196
9: -12.1694355, -9.7437096, -12.1629829, -9.7408438, -1.4614835, 1.5113602

Time for backsubstitution: 15.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 5841
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 6111

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 904

## Relational analysis of IS_A2_A2_B1_B2_B2_A1_A2_B1

### Relational analysis result of IS_A2_A2_B1_B2_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8194108, upper bound: 0.8298492
time: 6.09 seconds

## Relational analysis of IS_A2_A2_B1_B2_B2_A1_A2_B2

### Relational analysis result of IS_A2_A2_B1_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8202245, upper bound: 0.8343414
time: 6.19 seconds

## BFS IS instance: IS_A2_A2_B1_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -10.8866796, -8.2206841, -10.8637466, -8.2271624, -2.1271157, 2.1129084
1: -3.3762732, -0.7959623, -3.3551590, -0.8717895, -2.2103138, 2.2126431
2: 1.5121684, 3.6583548, 1.6044037, 3.6315541, -1.8493195, 1.8178124
3: -7.3229761, -5.2210159, -7.2652617, -5.2359877, -1.9436512, 1.9045677
4: -2.3919699, -0.3174829, -2.3649092, -0.3222694, -1.5386021, 1.5185595
5: -4.6397591, -2.6701603, -4.6255398, -2.7302179, -1.7744551, 1.7779906
6: -4.7547016, -2.0704324, -4.7280998, -2.1205020, -2.1565289, 2.1443000
7: -8.7329559, -6.8214931, -8.7214069, -6.8390555, -1.4295805, 1.4697460
8: -4.7139325, -2.3699150, -4.6898055, -2.4254212, -1.9207802, 1.9814668
9: -12.1772575, -9.7346382, -12.1676493, -9.7390308, -1.4665160, 1.5241079

Time for backsubstitution: 15.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5841
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 5818
type: A, layer: 1, pos: 5841
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 5818
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 510
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 6111

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5841

## Relational analysis of IS_A2_A2_B1_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 904

## Relational analysis of IS_A2_A2_B1_B2_B2_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1_B2_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8194390, upper bound: 0.8310001
time: 6.07 seconds

## Relational analysis of IS_A2_A2_B1_B2_B2_A2_A2_B2

### Relational analysis result of IS_A2_A2_B1_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8202267, upper bound: 0.8354653
time: 7.25 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -10.8818483, -8.2315369, -10.8233080, -8.2556114, -2.0954251, 2.1019387
1: -3.3611307, -0.8025692, -3.3331378, -0.8450801, -2.1856837, 2.1706271
2: 1.5228469, 3.6310773, 1.5788027, 3.5890136, -1.8012648, 1.8044915
3: -7.3171716, -5.2450027, -7.2967100, -5.2548723, -1.8759170, 1.8283195
4: -2.3748360, -0.3280888, -2.3355980, -0.4555311, -1.4789304, 1.4604654
5: -4.6217442, -2.6727681, -4.6081066, -2.7022429, -1.7472301, 1.7510061
6: -4.7393570, -2.0768657, -4.6954999, -2.1749377, -2.1255550, 2.1048465
7: -8.7208691, -6.8308444, -8.6855345, -6.8707438, -1.4112556, 1.4134405
8: -4.7039061, -2.3751011, -4.6493564, -2.3966022, -1.9509687, 1.9419305
9: -12.1465607, -9.7541943, -11.9899807, -9.7794542, -1.4145169, 1.4230490

Time for backsubstitution: 15.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 5841
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 5841
type: A, layer: 1, pos: 5818
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 5818
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 510
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 548

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 904

## Relational analysis of IS_A2_A2_B2_B2_B1_A1_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8108277, upper bound: 0.8285630
time: 5.86 seconds

## Relational analysis of IS_A2_A2_B2_B2_B1_A1_B1_B2

### Relational analysis result of IS_A2_A2_B2_B2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8116342, upper bound: 0.8330260
time: 5.49 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -10.8838730, -8.2214251, -10.8331814, -8.2327070, -2.1201668, 2.1284919
1: -3.3695955, -0.7970749, -3.3532214, -0.8301122, -2.2109957, 2.2202055
2: 1.5208374, 3.6549191, 1.5654043, 3.6424024, -1.8227181, 1.8618276
3: -7.3197222, -5.2243524, -7.3136845, -5.2282600, -1.9168377, 1.9178629
4: -2.3872921, -0.3207359, -2.3613486, -0.4225254, -1.5094273, 1.5120082
5: -4.6346364, -2.6717005, -4.6303277, -2.6954179, -1.7723794, 1.7738686
6: -4.7493744, -2.0715542, -4.7207127, -2.1558454, -2.1507144, 2.1457875
7: -8.7292175, -6.8256259, -8.7260866, -6.8556895, -1.4488435, 1.4521731
8: -4.7106209, -2.3726282, -4.6624289, -2.3804636, -1.9758177, 1.9575036
9: -12.1699791, -9.7455750, -12.0797071, -9.7578030, -1.5070961, 1.4548144

Time for backsubstitution: 15.00 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=1.8296241760253906
rel_dist={2: [-0.835577596891917, 0.8355774930246449]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2414.63 seconds
