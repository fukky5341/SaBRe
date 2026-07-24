## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 2.00905310091
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175)
1: (-15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554)
2: (-9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.5983925, 4.5983925)
3: (-7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804)
4: (-12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999)
5: (-6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669)
6: (-13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.5662365, 4.5662365)
7: (-10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956)
8: (7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923)
9: (-7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9519506, 3.9519506)

## BASE Result
execution time: IAR + LP analysis = 12.97 + 35.05 = 48.02 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -2.6198275, upper bound: 2.6198270


# Binary Search by BASE starts (time budget: 3551.98 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=3.14231538772583
rel_dist={8: [-2.00908813677729, 2.009087543266727]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.8899760246276855
rel_dist={8: [-1.5721964040429413, 1.5721963328311173]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=2.9740891456604004
rel_dist={8: [-1.725812598105506, 1.7258119688107083]}

## Binary search (step 3) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=5, k_high=5, k_mid=5, eps_mid=0.0195312, abs_max=3.0582022666931152
rel_dist={8: [-1.8721584799452113, 1.8721582345714651]}

## Binary Search Result
Binary search time: 208.52 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01953125


# Individual Split (IS_dual_ind) starts
Time budget: 3343.46 seconds

## Binary search (step 0) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6195
type: A, layer: 1, pos: 4555
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6195

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3480088, upper bound: 2.3247599
time: 5.30 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3480088, upper bound: 2.3480098
time: 6.68 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 12.11 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 12.11
Output dim: 8, lower bound: -2.3480088, upper bound: 2.3247599
IS_A2, status: Status.UNKNOWN, split count: 1, time: 12.11
Output dim: 8, lower bound: -2.3480088, upper bound: 2.3480098

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -6.1879435, -1.7816978, -6.2108626, -1.7584810, -4.4294624, 4.4291649
1: -15.2639427, -10.7425175, -15.3055611, -10.7053556, -4.5585871, 4.5630436
2: -9.0594845, -4.6674333, -9.1430454, -4.5636950, -4.4491224, 4.3801212
3: -7.5704598, -3.5940862, -7.6099777, -3.5488539, -4.0216060, 4.0158916
4: -12.2075682, -7.3990154, -12.2783413, -7.3609490, -4.8466191, 4.8793259
5: -5.9838700, -2.2321615, -6.0305119, -2.1928334, -3.7910366, 3.7983503
6: -13.7705059, -9.3012829, -13.8124371, -9.2557640, -4.2112408, 4.2019963
7: -10.2260075, -5.8782048, -10.2669849, -5.8650084, -4.3609991, 4.3887801
8: 7.8683748, 11.0728474, 7.8145905, 11.1099663, -3.2415915, 3.2582569
9: -7.1385541, -3.2466414, -7.1733198, -3.2267346, -3.8815360, 3.7883682

Time for backsubstitution: 12.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6195
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6195

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3247561, upper bound: 2.3247565
time: 6.03 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3247562, upper bound: 2.3247559
time: 6.36 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -6.2133999, -1.7566977, -6.2134099, -1.7566910, -4.4567089, 4.4567122
1: -15.3121786, -10.7039528, -15.3121958, -10.7039452, -4.6082335, 4.6082430
2: -9.1586056, -4.5603261, -9.1586962, -4.5603161, -4.5075827, 4.5025911
3: -7.6162882, -3.5472469, -7.6163149, -3.5472395, -4.0690489, 4.0690680
4: -12.2817307, -7.3541937, -12.2817440, -7.3541484, -4.9275823, 4.9275503
5: -6.0326066, -2.1862900, -6.0326171, -2.1862571, -3.8463495, 3.8463271
6: -13.8142948, -9.2481070, -13.8143044, -9.2480755, -4.2637520, 4.2391386
7: -10.2709599, -5.8640842, -10.2709761, -5.8640795, -4.4068804, 4.4068918
8: 7.8114176, 11.1164827, 7.8114066, 11.1164970, -3.3050795, 3.3050761
9: -7.1753492, -3.2234240, -7.1753635, -3.2234123, -3.9247780, 3.9364986

Time for backsubstitution: 12.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6195
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6195

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3247562, upper bound: 2.3480091
time: 8.62 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3247561, upper bound: 2.3480105
time: 6.53 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 27.62 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 27.62
Output dim: 8, lower bound: -2.3247561, upper bound: 2.3247565
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 27.62
Output dim: 8, lower bound: -2.3247562, upper bound: 2.3247559
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 27.62
Output dim: 8, lower bound: -2.3247562, upper bound: 2.3480091
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 27.62
Output dim: 8, lower bound: -2.3247561, upper bound: 2.3480105

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -6.1879435, -1.7816978, -6.1879435, -1.7816978, -4.4062457, 4.4062457
1: -15.2639427, -10.7425175, -15.2639427, -10.7425175, -4.5214252, 4.5214252
2: -9.0594845, -4.6674333, -9.0594845, -4.6674333, -4.2938309, 4.2938309
3: -7.5704598, -3.5940862, -7.5704598, -3.5940862, -3.9763737, 3.9763737
4: -12.2075682, -7.3990154, -12.2075682, -7.3990154, -4.8085527, 4.8085527
5: -5.9838700, -2.2321615, -5.9838700, -2.2321615, -3.7517085, 3.7517085
6: -13.7705059, -9.3012829, -13.7705059, -9.3012829, -4.1584978, 4.1584978
7: -10.2260075, -5.8782048, -10.2260075, -5.8782048, -4.3478026, 4.3478026
8: 7.8683748, 11.0728474, 7.8683748, 11.0728474, -3.2044725, 3.2044725
9: -7.1385541, -3.2466414, -7.1385541, -3.2466414, -3.7495627, 3.7495632

Time for backsubstitution: 13.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4555
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 4555

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3113489, upper bound: 2.3242806
time: 5.19 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3247346, upper bound: 2.3247390
time: 4.94 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -6.1879435, -1.7816978, -6.2133999, -1.7566977, -4.4312458, 4.4317021
1: -15.2639427, -10.7425175, -15.3121786, -10.7039528, -4.5599899, 4.5696611
2: -9.0594845, -4.6674333, -9.1586056, -4.5603261, -4.4027576, 4.3962708
3: -7.5704598, -3.5940862, -7.6162882, -3.5472469, -4.0232129, 4.0222020
4: -12.2075682, -7.3990154, -12.2817307, -7.3541937, -4.8533745, 4.8827152
5: -5.9838700, -2.2321615, -6.0326066, -2.1862900, -3.7975800, 3.8004451
6: -13.7705059, -9.3012829, -13.8142948, -9.2481070, -4.2190809, 4.2040505
7: -10.2260075, -5.8782048, -10.2709599, -5.8640842, -4.3619232, 4.3927550
8: 7.8683748, 11.0728474, 7.8114176, 11.1164827, -3.2481079, 3.2614298
9: -7.1385541, -3.2466414, -7.1753492, -3.2234240, -3.8744645, 3.7894068

Time for backsubstitution: 12.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4555
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 4555

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3113491, upper bound: 2.3242812
time: 9.45 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3247346, upper bound: 2.3247403
time: 5.98 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -6.2133999, -1.7566977, -6.1879435, -1.7816978, -4.4317021, 4.4312458
1: -15.3121786, -10.7039528, -15.2639427, -10.7425175, -4.5696611, 4.5599899
2: -9.1586056, -4.5603261, -9.0594845, -4.6674333, -4.3962708, 4.4027581
3: -7.6162882, -3.5472469, -7.5704598, -3.5940862, -4.0222020, 4.0232129
4: -12.2817307, -7.3541937, -12.2075682, -7.3990154, -4.8827152, 4.8533745
5: -6.0326066, -2.1862900, -5.9838700, -2.2321615, -3.8004451, 3.7975800
6: -13.8142948, -9.2481070, -13.7705059, -9.3012829, -4.2040510, 4.2190809
7: -10.2709599, -5.8640842, -10.2260075, -5.8782048, -4.3927550, 4.3619232
8: 7.8114176, 11.1164827, 7.8683748, 11.0728474, -3.2614298, 3.2481079
9: -7.1753492, -3.2234240, -7.1385541, -3.2466414, -3.7894063, 3.8744645

Time for backsubstitution: 12.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4555
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 4555

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3113490, upper bound: 2.3475835
time: 6.68 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3247347, upper bound: 2.3479871
time: 5.61 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -6.2133999, -1.7566977, -6.2133999, -1.7566977, -4.4567022, 4.4567022
1: -15.3121786, -10.7039528, -15.3121786, -10.7039528, -4.6082258, 4.6082258
2: -9.1586056, -4.5603261, -9.1586056, -4.5603261, -4.4910755, 4.4910765
3: -7.6162882, -3.5472469, -7.6162882, -3.5472469, -4.0690413, 4.0690413
4: -12.2817307, -7.3541937, -12.2817307, -7.3541937, -4.9275370, 4.9275370
5: -6.0326066, -2.1862900, -6.0326066, -2.1862900, -3.8463166, 3.8463166
6: -13.8142948, -9.2481070, -13.8142948, -9.2481070, -4.2391291, 4.2391291
7: -10.2709599, -5.8640842, -10.2709599, -5.8640842, -4.4068756, 4.4068756
8: 7.8114176, 11.1164827, 7.8114176, 11.1164827, -3.3050652, 3.3050652
9: -7.1753492, -3.2234240, -7.1753492, -3.2234240, -3.9364758, 3.9364758

Time for backsubstitution: 12.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4555
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 4555

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3113490, upper bound: 2.3475840
time: 13.67 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3247346, upper bound: 2.3479893
time: 8.93 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 35.08 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 35.08
Output dim: 8, lower bound: -2.3113489, upper bound: 2.3242806
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 35.08
Output dim: 8, lower bound: -2.3247346, upper bound: 2.3247390
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 35.08
Output dim: 8, lower bound: -2.3113491, upper bound: 2.3242812
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 35.08
Output dim: 8, lower bound: -2.3247346, upper bound: 2.3247403
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 35.08
Output dim: 8, lower bound: -2.3113490, upper bound: 2.3475835
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 35.08
Output dim: 8, lower bound: -2.3247347, upper bound: 2.3479871
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 35.08
Output dim: 8, lower bound: -2.3113490, upper bound: 2.3475840
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 35.08
Output dim: 8, lower bound: -2.3247346, upper bound: 2.3479893

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -6.1277590, -1.8190880, -6.1763220, -1.7838597, -4.3438993, 4.3572340
1: -15.2049656, -10.7705097, -15.2527761, -10.7444725, -4.4604931, 4.4822664
2: -9.0476599, -4.7045918, -9.0585632, -4.6742620, -4.2726870, 4.2492442
3: -7.5520787, -3.6056707, -7.5683756, -3.5959489, -3.9561298, 3.9627049
4: -12.1676674, -7.4248271, -12.2036152, -7.4041157, -4.7635517, 4.7787881
5: -5.9627142, -2.2491808, -5.9812770, -2.2350807, -3.7276335, 3.7320962
6: -13.7271786, -9.3266354, -13.7620440, -9.3030853, -4.1122141, 4.1248541
7: -10.2019348, -5.9000587, -10.2236328, -5.8823423, -4.3195925, 4.3235741
8: 7.8975544, 11.0509777, 7.8739462, 11.0711040, -3.1735497, 3.1770315
9: -7.1151929, -3.2647555, -7.1345129, -3.2495778, -3.7180271, 3.7273922

Time for backsubstitution: 13.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 4555

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3113516, upper bound: 2.3113509
time: 5.07 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3113516, upper bound: 2.3242809
time: 5.00 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -6.1878967, -1.7817087, -6.1879435, -1.7816978, -4.4061990, 4.4062347
1: -15.2638922, -10.7425289, -15.2639427, -10.7425175, -4.5213747, 4.5214138
2: -9.0594788, -4.6674643, -9.0594845, -4.6674333, -4.2938271, 4.2861614
3: -7.5704508, -3.5940967, -7.5704598, -3.5940862, -3.9763646, 3.9763632
4: -12.2075539, -7.3990283, -12.2075682, -7.3990154, -4.8085384, 4.8085399
5: -5.9838586, -2.2321703, -5.9838700, -2.2321615, -3.7516971, 3.7516997
6: -13.7704639, -9.3012924, -13.7705059, -9.3012829, -4.1566648, 4.1584926
7: -10.2259960, -5.8782253, -10.2260075, -5.8782048, -4.3477912, 4.3477821
8: 7.8684001, 11.0728378, 7.8683748, 11.0728474, -3.1974869, 3.2044630
9: -7.1385384, -3.2466509, -7.1385541, -3.2466414, -3.7596569, 3.7495403

Time for backsubstitution: 12.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 4555

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3242814, upper bound: 2.3113519
time: 5.42 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3242813, upper bound: 2.3113518
time: 4.72 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -6.1277590, -1.8190880, -6.2017784, -1.7588434, -4.3689156, 4.3826904
1: -15.2049656, -10.7705097, -15.3009768, -10.7059031, -4.4990625, 4.5304670
2: -9.0476599, -4.7045918, -9.1576881, -4.5671849, -4.3778257, 4.3516846
3: -7.5520787, -3.6056707, -7.6141872, -3.5491095, -4.0029693, 4.0085163
4: -12.1676674, -7.4248271, -12.2777996, -7.3592987, -4.8083687, 4.8529725
5: -5.9627142, -2.2491808, -6.0300236, -2.1892128, -3.7735014, 3.7808428
6: -13.7271786, -9.3266354, -13.8058233, -9.2499361, -4.1725264, 4.1680775
7: -10.2019348, -5.9000587, -10.2685480, -5.8682184, -4.3337164, 4.3684893
8: 7.8975544, 11.0509777, 7.8170147, 11.1147404, -3.2171860, 3.2339630
9: -7.1151929, -3.2647555, -7.1713085, -3.2263904, -3.8419657, 3.7672372

Time for backsubstitution: 12.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 4555

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3346081, upper bound: 2.3113515
time: 5.19 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3346082, upper bound: 2.3242813
time: 5.13 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6.1878967, -1.7817087, -6.2133999, -1.7566977, -4.4311991, 4.4316912
1: -15.2638922, -10.7425289, -15.3121786, -10.7039528, -4.5599394, 4.5696497
2: -9.0594788, -4.6674643, -9.1586056, -4.5603261, -4.3985243, 4.3886008
3: -7.5704508, -3.5940967, -7.6162882, -3.5472469, -4.0232038, 4.0221915
4: -12.2075539, -7.3990283, -12.2817307, -7.3541937, -4.8533602, 4.8827024
5: -5.9838586, -2.2321703, -6.0326066, -2.1862900, -3.7975686, 3.8004363
6: -13.7704639, -9.3012924, -13.8142948, -9.2481070, -4.2173872, 4.1989026
7: -10.2259960, -5.8782253, -10.2709599, -5.8640842, -4.3619118, 4.3927345
8: 7.8684001, 11.0728378, 7.8114176, 11.1164827, -3.2432299, 3.2614202
9: -7.1385384, -3.2466509, -7.1753492, -3.2234240, -3.8838658, 3.7893839

Time for backsubstitution: 12.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 4555

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3475834, upper bound: 2.3113501
time: 17.32 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3475836, upper bound: 2.3247391
time: 5.25 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.1533055, -1.7940526, -6.1763220, -1.7838597, -4.3694458, 4.3822694
1: -15.2530642, -10.7319374, -15.2527761, -10.7444725, -4.5085917, 4.5208387
2: -9.1467924, -4.5976739, -9.0585632, -4.6742620, -4.3751383, 4.3583412
3: -7.5978928, -3.5588391, -7.5683756, -3.5959489, -4.0019436, 4.0095367
4: -12.2417517, -7.3800268, -12.2036152, -7.4041157, -4.8376360, 4.8235884
5: -6.0114632, -2.2032671, -5.9812770, -2.2350807, -3.7763824, 3.7780099
6: -13.7709227, -9.2735767, -13.7620440, -9.3030853, -4.1577339, 4.1848993
7: -10.2466421, -5.8859348, -10.2236328, -5.8823423, -4.3642998, 4.3376980
8: 7.8407826, 11.0946245, 7.8739462, 11.0711040, -3.2303214, 3.2206783
9: -7.1516862, -3.2416601, -7.1345129, -3.2495778, -3.7577076, 3.8517675

Time for backsubstitution: 12.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4555

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3113511, upper bound: 2.3346077
time: 5.99 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3113511, upper bound: 2.3475831
time: 6.32 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6.2133527, -1.7567067, -6.1879435, -1.7816978, -4.4316549, 4.4312367
1: -15.3121233, -10.7039614, -15.2639427, -10.7425175, -4.5696058, 4.5599813
2: -9.1586027, -4.5603609, -9.0594845, -4.6674333, -4.3962679, 4.3948393
3: -7.6162786, -3.5472536, -7.5704598, -3.5940862, -4.0221925, 4.0232062
4: -12.2817135, -7.3542094, -12.2075682, -7.3990154, -4.8826981, 4.8533587
5: -6.0325971, -2.1862960, -5.9838700, -2.2321615, -3.8004355, 3.7975740
6: -13.8142548, -9.2481155, -13.7705059, -9.3012829, -4.2022190, 4.2190733
7: -10.2709513, -5.8641047, -10.2260075, -5.8782048, -4.3927464, 4.3619027
8: 7.8114424, 11.1164742, 7.8683748, 11.0728474, -3.2614050, 3.2480993
9: -7.1753325, -3.2234309, -7.1385541, -3.2466414, -3.7994995, 3.8744440

Time for backsubstitution: 13.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 4555

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3242810, upper bound: 2.3346088
time: 5.07 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3242811, upper bound: 2.3479870
time: 4.46 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.1533055, -1.7940526, -6.2017784, -1.7588434, -4.3944621, 4.4077258
1: -15.2530642, -10.7319374, -15.3009768, -10.7059031, -4.5471611, 4.5690393
2: -9.1467924, -4.5976739, -9.1576881, -4.5671849, -4.4661493, 4.4466558
3: -7.5978928, -3.5588391, -7.6141872, -3.5491095, -4.0487833, 4.0553484
4: -12.2417517, -7.3800268, -12.2777996, -7.3592987, -4.8824530, 4.8977728
5: -6.0114632, -2.2032671, -6.0300236, -2.1892128, -3.8222504, 3.8267565
6: -13.7709227, -9.2735767, -13.8058233, -9.2499361, -4.1925364, 4.2049417
7: -10.2466421, -5.8859348, -10.2685480, -5.8682184, -4.3784237, 4.3826132
8: 7.8407826, 11.0946245, 7.8170147, 11.1147404, -3.2739577, 3.2776098
9: -7.1516862, -3.2416601, -7.1713085, -3.2263904, -3.9039812, 3.9137630

Time for backsubstitution: 13.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 4555

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3113476, upper bound: 2.3346081
time: 5.34 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3113476, upper bound: 2.3475846
time: 4.80 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6.2133527, -1.7567067, -6.2133999, -1.7566977, -4.4566550, 4.4566932
1: -15.3121233, -10.7039614, -15.3121786, -10.7039528, -4.6081705, 4.6082172
2: -9.1586027, -4.5603609, -9.1586056, -4.5603261, -4.4868412, 4.4831567
3: -7.6162786, -3.5472536, -7.6162882, -3.5472469, -4.0690317, 4.0690346
4: -12.2817135, -7.3542094, -12.2817307, -7.3541937, -4.9275198, 4.9275212
5: -6.0325971, -2.1862960, -6.0326066, -2.1862900, -3.8463070, 3.8463106
6: -13.8142548, -9.2481155, -13.8142948, -9.2481070, -4.2374325, 4.2391210
7: -10.2709513, -5.8641047, -10.2709599, -5.8640842, -4.4068670, 4.4068551
8: 7.8114424, 11.1164742, 7.8114176, 11.1164827, -3.3050404, 3.3050566
9: -7.1753325, -3.2234309, -7.1753492, -3.2234240, -3.9458752, 3.9364548

Time for backsubstitution: 13.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 4555

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3242775, upper bound: 2.3346096
time: 5.43 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3242775, upper bound: 2.3346096
time: 4.57 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 23.27 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.27
Output dim: 8, lower bound: -2.3113516, upper bound: 2.3113509
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.27
Output dim: 8, lower bound: -2.3113516, upper bound: 2.3242809
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.27
Output dim: 8, lower bound: -2.3242814, upper bound: 2.3113519
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.27
Output dim: 8, lower bound: -2.3242813, upper bound: 2.3113518
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.27
Output dim: 8, lower bound: -2.3346081, upper bound: 2.3113515
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.27
Output dim: 8, lower bound: -2.3346082, upper bound: 2.3242813
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.27
Output dim: 8, lower bound: -2.3475834, upper bound: 2.3113501
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.27
Output dim: 8, lower bound: -2.3475836, upper bound: 2.3247391
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.27
Output dim: 8, lower bound: -2.3113511, upper bound: 2.3346077
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.27
Output dim: 8, lower bound: -2.3113511, upper bound: 2.3475831
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.27
Output dim: 8, lower bound: -2.3242810, upper bound: 2.3346088
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.27
Output dim: 8, lower bound: -2.3242811, upper bound: 2.3479870
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.27
Output dim: 8, lower bound: -2.3113476, upper bound: 2.3346081
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.27
Output dim: 8, lower bound: -2.3113476, upper bound: 2.3475846
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.27
Output dim: 8, lower bound: -2.3242775, upper bound: 2.3346096
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.27
Output dim: 8, lower bound: -2.3242775, upper bound: 2.3346096

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.1277590, -1.8190880, -6.1277590, -1.8190880, -4.3086710, 4.3086710
1: -15.2049656, -10.7705097, -15.2049656, -10.7705097, -4.4344559, 4.4344559
2: -9.0476599, -4.7045918, -9.0476599, -4.7045918, -4.2374182, 4.2374177
3: -7.5520787, -3.6056707, -7.5520787, -3.6056707, -3.9464080, 3.9464080
4: -12.1676674, -7.4248271, -12.1676674, -7.4248271, -4.7428403, 4.7428403
5: -5.9627142, -2.2491808, -5.9627142, -2.2491808, -3.7135334, 3.7135334
6: -13.7271786, -9.3266354, -13.7271786, -9.3266354, -4.0889320, 4.0889325
7: -10.2019348, -5.9000587, -10.2019348, -5.9000587, -4.3018761, 4.3018761
8: 7.8975544, 11.0509777, 7.8975544, 11.0509777, -3.1534233, 3.1534233
9: -7.1151929, -3.2647555, -7.1151929, -3.2647555, -3.7043309, 3.7043309

Time for backsubstitution: 12.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3113498, upper bound: 2.3087111
time: 5.80 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3113498, upper bound: 2.3113487
time: 4.87 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.1277590, -1.8190880, -6.1878967, -1.7817087, -4.3460503, 4.3688087
1: -15.2049656, -10.7705097, -15.2638922, -10.7425289, -4.4624367, 4.4933825
2: -9.0476599, -4.7045918, -9.0594788, -4.6674643, -4.2808390, 4.2503676
3: -7.5520787, -3.6056707, -7.5704508, -3.5940967, -3.9579821, 3.9647801
4: -12.1676674, -7.4248271, -12.2075539, -7.3990283, -4.7686391, 4.7827268
5: -5.9627142, -2.2491808, -5.9838586, -2.2321703, -3.7305439, 3.7346778
6: -13.7271786, -9.3266354, -13.7704639, -9.3012924, -4.1138239, 4.1335592
7: -10.2019348, -5.9000587, -10.2259960, -5.8782253, -4.3237095, 4.3259373
8: 7.8975544, 11.0509777, 7.8684001, 11.0728378, -3.1752834, 3.1825776
9: -7.1151929, -3.2647555, -7.1385384, -3.2466509, -3.7224159, 3.7294149

Time for backsubstitution: 12.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3113498, upper bound: 2.3216871
time: 5.37 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3113496, upper bound: 2.3242786
time: 5.05 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6.1878967, -1.7817087, -6.1277590, -1.8190880, -4.3688087, 4.3460503
1: -15.2638922, -10.7425289, -15.2049656, -10.7705097, -4.4933825, 4.4624367
2: -9.0594788, -4.6674643, -9.0476599, -4.7045918, -4.2503672, 4.2808390
3: -7.5704508, -3.5940967, -7.5520787, -3.6056707, -3.9647801, 3.9579821
4: -12.2075539, -7.3990283, -12.1676674, -7.4248271, -4.7827268, 4.7686391
5: -5.9838586, -2.2321703, -5.9627142, -2.2491808, -3.7346778, 3.7305439
6: -13.7704639, -9.3012924, -13.7271786, -9.3266354, -4.1335592, 4.1138239
7: -10.2259960, -5.8782253, -10.2019348, -5.9000587, -4.3259373, 4.3237095
8: 7.8684001, 11.0728378, 7.8975544, 11.0509777, -3.1825776, 3.1752834
9: -7.1385384, -3.2466509, -7.1151929, -3.2647555, -3.7294149, 3.7224159

Time for backsubstitution: 12.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3242776, upper bound: 2.3087110
time: 5.37 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3242776, upper bound: 2.3113486
time: 4.81 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6.1878967, -1.7817087, -6.1878967, -1.7817087, -4.4061880, 4.4061880
1: -15.2638922, -10.7425289, -15.2638922, -10.7425289, -4.5213633, 4.5213633
2: -9.0594788, -4.6674643, -9.0594788, -4.6674643, -4.2861567, 4.2861567
3: -7.5704508, -3.5940967, -7.5704508, -3.5940967, -3.9763541, 3.9763541
4: -12.2075539, -7.3990283, -12.2075539, -7.3990283, -4.8085256, 4.8085256
5: -5.9838586, -2.2321703, -5.9838586, -2.2321703, -3.7516882, 3.7516882
6: -13.7704639, -9.3012924, -13.7704639, -9.3012924, -4.1566591, 4.1566591
7: -10.2259960, -5.8782253, -10.2259960, -5.8782253, -4.3477707, 4.3477707
8: 7.8684001, 11.0728378, 7.8684001, 11.0728378, -3.1974812, 3.1974807
9: -7.1385384, -3.2466509, -7.1385384, -3.2466509, -3.7596350, 3.7596350

Time for backsubstitution: 12.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3242783, upper bound: 2.3087110
time: 4.51 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3242782, upper bound: 2.3138763
time: 4.65 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -6.1277590, -1.8190880, -6.1533055, -1.7940526, -4.3337064, 4.3342175
1: -15.2049656, -10.7705097, -15.2530642, -10.7319374, -4.4730282, 4.4825544
2: -9.0476599, -4.7045918, -9.1467924, -4.5976739, -4.3464556, 4.3398695
3: -7.5520787, -3.6056707, -7.5978928, -3.5588391, -3.9932396, 3.9922221
4: -12.1676674, -7.4248271, -12.2417517, -7.3800268, -4.7876406, 4.8169246
5: -5.9627142, -2.2491808, -6.0114632, -2.2032671, -3.7594471, 3.7622824
6: -13.7271786, -9.3266354, -13.7709227, -9.2735767, -4.1488371, 4.1344552
7: -10.2019348, -5.9000587, -10.2466421, -5.8859348, -4.3160000, 4.3465834
8: 7.8975544, 11.0509777, 7.8407826, 11.0946245, -3.1970701, 3.2101951
9: -7.1151929, -3.2647555, -7.1516862, -3.2416601, -3.8279057, 3.7440114

Time for backsubstitution: 13.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3346064, upper bound: 2.3087106
time: 5.04 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3346064, upper bound: 2.3113482
time: 4.95 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -6.1277590, -1.8190880, -6.2133527, -1.7567067, -4.3710523, 4.3942647
1: -15.2049656, -10.7705097, -15.3121233, -10.7039614, -4.5010042, 4.5416136
2: -9.0476599, -4.7045918, -9.1586027, -4.5603609, -4.3824682, 4.3528090
3: -7.5520787, -3.6056707, -7.6162786, -3.5472536, -4.0048251, 4.0106077
4: -12.1676674, -7.4248271, -12.2817135, -7.3542094, -4.8134580, 4.8568864
5: -5.9627142, -2.2491808, -6.0325971, -2.1862960, -3.7764182, 3.7834163
6: -13.7271786, -9.3266354, -13.8142548, -9.2481155, -4.1742382, 4.1702213
7: -10.2019348, -5.9000587, -10.2709513, -5.8641047, -4.3378301, 4.3708925
8: 7.8975544, 11.0509777, 7.8114424, 11.1164742, -3.2189198, 3.2395353
9: -7.1151929, -3.2647555, -7.1753325, -3.2234309, -3.8463335, 3.7693181

Time for backsubstitution: 13.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3346065, upper bound: 2.3216867
time: 5.12 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3346064, upper bound: 2.3242780
time: 5.32 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6.1878967, -1.7817087, -6.1533055, -1.7940526, -4.3938441, 4.3715968
1: -15.2638922, -10.7425289, -15.2530642, -10.7319374, -4.5319548, 4.5105352
2: -9.0594788, -4.6674643, -9.1467924, -4.5976739, -4.3552256, 4.3821220
3: -7.5704508, -3.5940967, -7.5978928, -3.5588391, -4.0116119, 4.0037961
4: -12.2075539, -7.3990283, -12.2417517, -7.3800268, -4.8275270, 4.8427234
5: -5.9838586, -2.2321703, -6.0114632, -2.2032671, -3.7805915, 3.7792928
6: -13.7704639, -9.3012924, -13.7709227, -9.2735767, -4.1936255, 4.1540794
7: -10.2259960, -5.8782253, -10.2466421, -5.8859348, -4.3400612, 4.3684168
8: 7.8684001, 11.0728378, 7.8407826, 11.0946245, -3.2262244, 3.2320552
9: -7.1385384, -3.2466509, -7.1516862, -3.2416601, -3.8544846, 3.7620964

Time for backsubstitution: 13.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3475797, upper bound: 2.3087104
time: 4.85 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3475797, upper bound: 2.3113479
time: 4.75 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6.1878967, -1.7817087, -6.2133527, -1.7567067, -4.4311900, 4.4316440
1: -15.2638922, -10.7425289, -15.3121233, -10.7039614, -4.5599308, 4.5695944
2: -9.0594788, -4.6674643, -9.1586027, -4.5603609, -4.3947821, 4.3885975
3: -7.5704508, -3.5940967, -7.6162786, -3.5472536, -4.0231972, 4.0221820
4: -12.2075539, -7.3990283, -12.2817135, -7.3542094, -4.8533444, 4.8826852
5: -5.9838586, -2.2321703, -6.0325971, -2.1862960, -3.7975626, 3.8004267
6: -13.7704639, -9.3012924, -13.8142548, -9.2481155, -4.2173805, 4.1980548
7: -10.2259960, -5.8782253, -10.2709513, -5.8641047, -4.3618913, 4.3927259
8: 7.8684001, 11.0728378, 7.8114424, 11.1164742, -3.2432222, 3.2613955
9: -7.1385384, -3.2466509, -7.1753325, -3.2234309, -3.8838453, 3.7994766

Time for backsubstitution: 13.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3475803, upper bound: 2.3087100
time: 13.83 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3475804, upper bound: 2.3113474
time: 14.77 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.1533055, -1.7940526, -6.1277590, -1.8190880, -4.3342175, 4.3337064
1: -15.2530642, -10.7319374, -15.2049656, -10.7705097, -4.4825544, 4.4730282
2: -9.1467924, -4.5976739, -9.0476599, -4.7045918, -4.3398695, 4.3464556
3: -7.5978928, -3.5588391, -7.5520787, -3.6056707, -3.9922221, 3.9932396
4: -12.2417517, -7.3800268, -12.1676674, -7.4248271, -4.8169246, 4.7876406
5: -6.0114632, -2.2032671, -5.9627142, -2.2491808, -3.7622824, 3.7594471
6: -13.7709227, -9.2735767, -13.7271786, -9.3266354, -4.1344557, 4.1488371
7: -10.2466421, -5.8859348, -10.2019348, -5.9000587, -4.3465834, 4.3160000
8: 7.8407826, 11.0946245, 7.8975544, 11.0509777, -3.2101951, 3.1970701
9: -7.1516862, -3.2416601, -7.1151929, -3.2647555, -3.7440114, 3.8279061

Time for backsubstitution: 13.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3113495, upper bound: 2.3319967
time: 5.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3113493, upper bound: 2.3346056
time: 5.44 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.1533055, -1.7940526, -6.1878967, -1.7817087, -4.3715968, 4.3938441
1: -15.2530642, -10.7319374, -15.2638922, -10.7425289, -4.5105352, 4.5319548
2: -9.1467924, -4.5976739, -9.0594788, -4.6674643, -4.3821220, 4.3552256
3: -7.5978928, -3.5588391, -7.5704508, -3.5940967, -4.0037961, 4.0116119
4: -12.2417517, -7.3800268, -12.2075539, -7.3990283, -4.8427234, 4.8275270
5: -6.0114632, -2.2032671, -5.9838586, -2.2321703, -3.7792928, 3.7805915
6: -13.7709227, -9.2735767, -13.7704639, -9.3012924, -4.1540785, 4.1936259
7: -10.2466421, -5.8859348, -10.2259960, -5.8782253, -4.3684168, 4.3400612
8: 7.8407826, 11.0946245, 7.8684001, 11.0728378, -3.2320552, 3.2262244
9: -7.1516862, -3.2416601, -7.1385384, -3.2466509, -3.7620964, 3.8544846

Time for backsubstitution: 13.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3113493, upper bound: 2.3449762
time: 5.34 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3113494, upper bound: 2.3475808
time: 5.13 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6.2133527, -1.7567067, -6.1277590, -1.8190880, -4.3942647, 4.3710523
1: -15.3121233, -10.7039614, -15.2049656, -10.7705097, -4.5416136, 4.5010042
2: -9.1586027, -4.5603609, -9.0476599, -4.7045918, -4.3528090, 4.3824677
3: -7.6162786, -3.5472536, -7.5520787, -3.6056707, -4.0106077, 4.0048251
4: -12.2817135, -7.3542094, -12.1676674, -7.4248271, -4.8568864, 4.8134580
5: -6.0325971, -2.1862960, -5.9627142, -2.2491808, -3.7834163, 3.7764182
6: -13.8142548, -9.2481155, -13.7271786, -9.3266354, -4.1702213, 4.1742387
7: -10.2709513, -5.8641047, -10.2019348, -5.9000587, -4.3708925, 4.3378301
8: 7.8114424, 11.1164742, 7.8975544, 11.0509777, -3.2395353, 3.2189198
9: -7.1753325, -3.2234309, -7.1151929, -3.2647555, -3.7693186, 3.8463340

Time for backsubstitution: 12.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3242772, upper bound: 2.3319967
time: 4.69 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3242773, upper bound: 2.3346055
time: 5.22 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6.2133527, -1.7567067, -6.1878967, -1.7817087, -4.4316440, 4.4311900
1: -15.3121233, -10.7039614, -15.2638922, -10.7425289, -4.5695944, 4.5599308
2: -9.1586027, -4.5603609, -9.0594788, -4.6674643, -4.3885984, 4.3947825
3: -7.6162786, -3.5472536, -7.5704508, -3.5940967, -4.0221820, 4.0231972
4: -12.2817135, -7.3542094, -12.2075539, -7.3990283, -4.8826852, 4.8533444
5: -6.0325971, -2.1862960, -5.9838586, -2.2321703, -3.8004267, 3.7975626
6: -13.8142548, -9.2481155, -13.7704639, -9.3012924, -4.1980553, 4.2173805
7: -10.2709513, -5.8641047, -10.2259960, -5.8782253, -4.3927259, 4.3618913
8: 7.8114424, 11.1164742, 7.8684001, 11.0728378, -3.2613955, 3.2432222
9: -7.1753325, -3.2234309, -7.1385384, -3.2466509, -3.7994776, 3.8838453

Time for backsubstitution: 13.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3242780, upper bound: 2.3344740
time: 4.60 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3242779, upper bound: 2.3346067
time: 5.66 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -6.1533055, -1.7940526, -6.1533055, -1.7940526, -4.3592529, 4.3592529
1: -15.2530642, -10.7319374, -15.2530642, -10.7319374, -4.5211267, 4.5211267
2: -9.1467924, -4.5976739, -9.1467924, -4.5976739, -4.4347792, 4.4347787
3: -7.5978928, -3.5588391, -7.5978928, -3.5588391, -4.0390539, 4.0390539
4: -12.2417517, -7.3800268, -12.2417517, -7.3800268, -4.8617249, 4.8617249
5: -6.0114632, -2.2032671, -6.0114632, -2.2032671, -3.8081961, 3.8081961
6: -13.7709227, -9.2735767, -13.7709227, -9.2735767, -4.1688442, 4.1688447
7: -10.2466421, -5.8859348, -10.2466421, -5.8859348, -4.3607073, 4.3607073
8: 7.8407826, 11.0946245, 7.8407826, 11.0946245, -3.2538419, 3.2538419
9: -7.1516862, -3.2416601, -7.1516862, -3.2416601, -3.8899074, 3.8899069

Time for backsubstitution: 12.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3113459, upper bound: 2.3319980
time: 5.17 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3113459, upper bound: 2.3346066
time: 4.77 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -6.1533055, -1.7940526, -6.2133527, -1.7567067, -4.3965988, 4.4193001
1: -15.2530642, -10.7319374, -15.3121233, -10.7039614, -4.5491028, 4.5801859
2: -9.1467924, -4.5976739, -9.1586027, -4.5603609, -4.4707909, 4.4435430
3: -7.5978928, -3.5588391, -7.6162786, -3.5472536, -4.0506392, 4.0574398
4: -12.2417517, -7.3800268, -12.2817135, -7.3542094, -4.8875422, 4.9016867
5: -6.0114632, -2.2032671, -6.0325971, -2.1862960, -3.8251672, 3.8293300
6: -13.7709227, -9.2735767, -13.8142548, -9.2481155, -4.1942482, 4.2136745
7: -10.2466421, -5.8859348, -10.2709513, -5.8641047, -4.3825374, 4.3850164
8: 7.8407826, 11.0946245, 7.8114424, 11.1164742, -3.2756915, 3.2831821
9: -7.1516862, -3.2416601, -7.1753325, -3.2234309, -3.9083261, 3.9165668

Time for backsubstitution: 12.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3113458, upper bound: 2.3449768
time: 4.98 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3113459, upper bound: 2.3475818
time: 4.66 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6.2133527, -1.7567067, -6.1533055, -1.7940526, -4.4193001, 4.3965988
1: -15.3121233, -10.7039614, -15.2530642, -10.7319374, -4.5801859, 4.5491028
2: -9.1586027, -4.5603609, -9.1467924, -4.5976739, -4.4435434, 4.4707913
3: -7.6162786, -3.5472536, -7.5978928, -3.5588391, -4.0574398, 4.0506392
4: -12.2817135, -7.3542094, -12.2417517, -7.3800268, -4.9016867, 4.8875422
5: -6.0325971, -2.1862960, -6.0114632, -2.2032671, -3.8293300, 3.8251672
6: -13.8142548, -9.2481155, -13.7709227, -9.2735767, -4.2136755, 4.1942482
7: -10.2709513, -5.8641047, -10.2466421, -5.8859348, -4.3850164, 4.3825374
8: 7.8114424, 11.1164742, 7.8407826, 11.0946245, -3.2831821, 3.2756915
9: -7.1753325, -3.2234309, -7.1516862, -3.2416601, -3.9165673, 3.9083261

Time for backsubstitution: 12.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3242737, upper bound: 2.3319974
time: 5.07 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3242738, upper bound: 2.3346063
time: 4.73 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6.2133527, -1.7567067, -6.2133527, -1.7567067, -4.4566460, 4.4566460
1: -15.3121233, -10.7039614, -15.3121233, -10.7039614, -4.6081619, 4.6081619
2: -9.1586027, -4.5603609, -9.1586027, -4.5603609, -4.4831009, 4.4830995
3: -7.6162786, -3.5472536, -7.6162786, -3.5472536, -4.0690250, 4.0690250
4: -12.2817135, -7.3542094, -12.2817135, -7.3542094, -4.9275041, 4.9275041
5: -6.0325971, -2.1862960, -6.0325971, -2.1862960, -3.8463011, 3.8463011
6: -13.8142548, -9.2481155, -13.8142548, -9.2481155, -4.2374258, 4.2374258
7: -10.2709513, -5.8641047, -10.2709513, -5.8641047, -4.4068465, 4.4068465
8: 7.8114424, 11.1164742, 7.8114424, 11.1164742, -3.3050318, 3.3050318
9: -7.1753325, -3.2234309, -7.1753325, -3.2234309, -3.9458542, 3.9458547

Time for backsubstitution: 13.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3242743, upper bound: 2.3344751
time: 4.60 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3242743, upper bound: 2.3346057
time: 4.75 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 22.63 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.63
Output dim: 8, lower bound: -2.3113498, upper bound: 2.3087111
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.63
Output dim: 8, lower bound: -2.3113498, upper bound: 2.3113487
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.63
Output dim: 8, lower bound: -2.3113498, upper bound: 2.3216871
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.63
Output dim: 8, lower bound: -2.3113496, upper bound: 2.3242786
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.63
Output dim: 8, lower bound: -2.3242776, upper bound: 2.3087110
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.63
Output dim: 8, lower bound: -2.3242776, upper bound: 2.3113486
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.63
Output dim: 8, lower bound: -2.3242783, upper bound: 2.3087110
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.63
Output dim: 8, lower bound: -2.3242782, upper bound: 2.3138763
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.63
Output dim: 8, lower bound: -2.3346064, upper bound: 2.3087106
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.63
Output dim: 8, lower bound: -2.3346064, upper bound: 2.3113482
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.63
Output dim: 8, lower bound: -2.3346065, upper bound: 2.3216867
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.63
Output dim: 8, lower bound: -2.3346064, upper bound: 2.3242780
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.63
Output dim: 8, lower bound: -2.3475797, upper bound: 2.3087104
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.63
Output dim: 8, lower bound: -2.3475797, upper bound: 2.3113479
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.63
Output dim: 8, lower bound: -2.3475803, upper bound: 2.3087100
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.63
Output dim: 8, lower bound: -2.3475804, upper bound: 2.3113474
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.63
Output dim: 8, lower bound: -2.3113495, upper bound: 2.3319967
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.63
Output dim: 8, lower bound: -2.3113493, upper bound: 2.3346056
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.63
Output dim: 8, lower bound: -2.3113493, upper bound: 2.3449762
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.63
Output dim: 8, lower bound: -2.3113494, upper bound: 2.3475808
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.63
Output dim: 8, lower bound: -2.3242772, upper bound: 2.3319967
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.63
Output dim: 8, lower bound: -2.3242773, upper bound: 2.3346055
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.63
Output dim: 8, lower bound: -2.3242780, upper bound: 2.3344740
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.63
Output dim: 8, lower bound: -2.3242779, upper bound: 2.3346067
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.63
Output dim: 8, lower bound: -2.3113459, upper bound: 2.3319980
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.63
Output dim: 8, lower bound: -2.3113459, upper bound: 2.3346066
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.63
Output dim: 8, lower bound: -2.3113458, upper bound: 2.3449768
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.63
Output dim: 8, lower bound: -2.3113459, upper bound: 2.3475818
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.63
Output dim: 8, lower bound: -2.3242737, upper bound: 2.3319974
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.63
Output dim: 8, lower bound: -2.3242738, upper bound: 2.3346063
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.63
Output dim: 8, lower bound: -2.3242743, upper bound: 2.3344751
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.63
Output dim: 8, lower bound: -2.3242743, upper bound: 2.3346057
Binary search (step 0): status=Status.UNKNOWN, k_low=6, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=3.3050923347473145
rel_dist={8: [-2.3480243628211177, 2.348024799945545]}

## Binary search (step 1) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6195
type: A, layer: 1, pos: 4555
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6195

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1376539, upper bound: 2.1186598
time: 5.61 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1376537, upper bound: 2.1376536
time: 7.42 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 13.17 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 13.17
Output dim: 8, lower bound: -2.1376539, upper bound: 2.1186598
IS_A2, status: Status.UNKNOWN, split count: 1, time: 13.17
Output dim: 8, lower bound: -2.1376537, upper bound: 2.1376536

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -6.1879435, -1.7816978, -6.2091494, -1.7597084, -4.4279652, 4.4256988
1: -15.2639427, -10.7425175, -15.3010950, -10.7063179, -4.4406815, 4.5300636
2: -9.0594845, -4.6674333, -9.1324720, -4.5660038, -4.2481718, 4.1667862
3: -7.5704598, -3.5940862, -7.6057081, -3.5499494, -4.0205107, 4.0116220
4: -12.2075682, -7.3990154, -12.2760239, -7.3655386, -4.8420296, 4.8770084
5: -5.9838700, -2.2321615, -6.0290804, -2.1972632, -3.6643038, 3.6720529
6: -13.7705059, -9.3012829, -13.8111706, -9.2609310, -3.9703932, 3.9632015
7: -10.2260075, -5.8782048, -10.2642660, -5.8656473, -4.3603601, 4.3860612
8: 7.8683748, 11.0728474, 7.8167686, 11.1055670, -3.0677624, 3.1755977
9: -7.1385541, -3.2466414, -7.1719365, -3.2289586, -3.7048750, 3.5830765

Time for backsubstitution: 13.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6195
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6195

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1186570, upper bound: 2.1186579
time: 5.04 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1186570, upper bound: 2.1186570
time: 13.44 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -6.2133999, -1.7566977, -6.2134089, -1.7566910, -4.4535160, 4.4567113
1: -15.3121786, -10.7039528, -15.3121939, -10.7039461, -4.5886164, 4.6050925
2: -9.1586056, -4.5603261, -9.1586838, -4.5603180, -4.3024845, 4.3011184
3: -7.6162882, -3.5472469, -7.6163106, -3.5472407, -4.0477238, 4.0690637
4: -12.2817307, -7.3541937, -12.2817440, -7.3541565, -4.9275742, 4.9275503
5: -6.0326066, -2.1862900, -6.0326147, -2.1862621, -3.7216377, 3.7146559
6: -13.8142948, -9.2481070, -13.8143024, -9.2480803, -4.0281916, 4.0003037
7: -10.2709599, -5.8640842, -10.2709703, -5.8640804, -4.4068794, 4.4068861
8: 7.8114176, 11.1164827, 7.8114080, 11.1164961, -3.2178316, 3.2062731
9: -7.1753492, -3.2234240, -7.1753602, -3.2234144, -3.7479601, 3.7588367

Time for backsubstitution: 13.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6195
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6195

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1186571, upper bound: 2.1376547
time: 4.85 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1186570, upper bound: 2.1376537
time: 8.33 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 26.47 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 26.47
Output dim: 8, lower bound: -2.1186570, upper bound: 2.1186579
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 26.47
Output dim: 8, lower bound: -2.1186570, upper bound: 2.1186570
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 26.47
Output dim: 8, lower bound: -2.1186571, upper bound: 2.1376547
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 26.47
Output dim: 8, lower bound: -2.1186570, upper bound: 2.1376537

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -6.1879435, -1.7816978, -6.1879435, -1.7816978, -4.3907785, 4.3907785
1: -15.2639427, -10.7425175, -15.2639427, -10.7425175, -4.3803043, 4.3803043
2: -9.0594845, -4.6674333, -9.0594845, -4.6674333, -4.0914392, 4.0914388
3: -7.5704598, -3.5940862, -7.5704598, -3.5940862, -3.9763737, 3.9763737
4: -12.2075682, -7.3990154, -12.2075682, -7.3990154, -4.8085527, 4.8085527
5: -5.9838700, -2.2321615, -5.9838700, -2.2321615, -3.6292849, 3.6292849
6: -13.7705059, -9.3012829, -13.7705059, -9.3012829, -3.9211063, 3.9211059
7: -10.2260075, -5.8782048, -10.2260075, -5.8782048, -4.3478026, 4.3478026
8: 7.8683748, 11.0728474, 7.8683748, 11.0728474, -3.0335011, 3.0335011
9: -7.1385541, -3.2466414, -7.1385541, -3.2466414, -3.5458817, 3.5458822

Time for backsubstitution: 12.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4555
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 4555

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1068661, upper bound: 2.1170295
time: 5.24 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1186414, upper bound: 2.1186451
time: 5.38 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -6.1879435, -1.7816978, -6.2133999, -1.7566977, -4.4144726, 4.4173069
1: -15.2639427, -10.7425175, -15.3121786, -10.7039528, -4.4223204, 4.5398531
2: -9.0594845, -4.6674333, -9.1586056, -4.5603261, -4.2012863, 4.1841302
3: -7.5704598, -3.5940862, -7.6162882, -3.5472469, -4.0232129, 4.0222020
4: -12.2075682, -7.3990154, -12.2817307, -7.3541937, -4.8533745, 4.8827152
5: -5.9838700, -2.2321615, -6.0326066, -2.1862900, -3.6754255, 3.6753063
6: -13.7705059, -9.3012829, -13.8142948, -9.2481070, -3.9835253, 3.9666591
7: -10.2260075, -5.8782048, -10.2709599, -5.8640842, -4.3619232, 4.3927550
8: 7.8683748, 11.0728474, 7.8114176, 11.1164827, -3.0792441, 3.1708853
9: -7.1385541, -3.2466414, -7.1753492, -3.2234240, -3.6997266, 3.5857258

Time for backsubstitution: 12.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4555
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 4555

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1068660, upper bound: 2.1170293
time: 5.33 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1186415, upper bound: 2.1186438
time: 9.41 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -6.2133999, -1.7566977, -6.1879435, -1.7816978, -4.4173069, 4.4144726
1: -15.3121786, -10.7039528, -15.2639427, -10.7425175, -4.5398531, 4.4223204
2: -9.1586056, -4.5603261, -9.0594845, -4.6674333, -4.1841307, 4.2012873
3: -7.6162882, -3.5472469, -7.5704598, -3.5940862, -4.0222020, 4.0232129
4: -12.2817307, -7.3541937, -12.2075682, -7.3990154, -4.8827152, 4.8533745
5: -6.0326066, -2.1862900, -5.9838700, -2.2321615, -3.6753073, 3.6754255
6: -13.8142948, -9.2481070, -13.7705059, -9.3012829, -3.9666595, 3.9835253
7: -10.2709599, -5.8640842, -10.2260075, -5.8782048, -4.3927550, 4.3619232
8: 7.8114176, 11.1164827, 7.8683748, 11.0728474, -3.1708856, 3.0792437
9: -7.1753492, -3.2234240, -7.1385541, -3.2466414, -3.5857253, 3.6997266

Time for backsubstitution: 12.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4555
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 4555

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1068661, upper bound: 2.1361209
time: 5.63 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1186415, upper bound: 2.1376382
time: 5.31 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -6.2133999, -1.7566977, -6.2133999, -1.7566977, -4.4567022, 4.4567022
1: -15.3121786, -10.7039528, -15.3121786, -10.7039528, -4.6050577, 4.6050572
2: -9.1586056, -4.5603261, -9.1586056, -4.5603261, -4.2837563, 4.2837563
3: -7.6162882, -3.5472469, -7.6162882, -3.5472469, -4.0477161, 4.0477161
4: -12.2817307, -7.3541937, -12.2817307, -7.3541937, -4.9275370, 4.9275370
5: -6.0326066, -2.1862900, -6.0326066, -2.1862900, -3.7146492, 3.7146487
6: -13.8142948, -9.2481070, -13.8142948, -9.2481070, -4.0002966, 4.0002966
7: -10.2709599, -5.8640842, -10.2709599, -5.8640842, -4.4068756, 4.4068756
8: 7.8114176, 11.1164827, 7.8114176, 11.1164827, -3.2062640, 3.2062640
9: -7.1753492, -3.2234240, -7.1753492, -3.2234240, -3.7588186, 3.7588186

Time for backsubstitution: 13.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4555
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 4555

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1068661, upper bound: 2.1361217
time: 5.38 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1186414, upper bound: 2.1376377
time: 7.73 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 26.38 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 26.38
Output dim: 8, lower bound: -2.1068661, upper bound: 2.1170295
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.38
Output dim: 8, lower bound: -2.1186414, upper bound: 2.1186451
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.38
Output dim: 8, lower bound: -2.1068660, upper bound: 2.1170293
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.38
Output dim: 8, lower bound: -2.1186415, upper bound: 2.1186438
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 26.38
Output dim: 8, lower bound: -2.1068661, upper bound: 2.1361209
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.38
Output dim: 8, lower bound: -2.1186415, upper bound: 2.1376382
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.38
Output dim: 8, lower bound: -2.1068661, upper bound: 2.1361217
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.38
Output dim: 8, lower bound: -2.1186414, upper bound: 2.1376377

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -6.1277590, -1.8190880, -6.1710334, -1.7848673, -4.3294296, 4.3393154
1: -15.2049656, -10.7705097, -15.2476873, -10.7453871, -4.3170738, 4.3330112
2: -9.0476599, -4.7045918, -9.0581493, -4.6773758, -4.0665379, 4.0463338
3: -7.5520787, -3.6056707, -7.5674262, -3.5967965, -3.9552822, 3.9617555
4: -12.1676674, -7.4248271, -12.2017918, -7.4064312, -4.7612362, 4.7769647
5: -5.9627142, -2.2491808, -5.9800730, -2.2364025, -3.5970640, 3.6053224
6: -13.7271786, -9.3266354, -13.7581863, -9.3039322, -3.8740692, 3.8833718
7: -10.2019348, -5.9000587, -10.2225571, -5.8842268, -4.3177080, 4.3224983
8: 7.8975544, 11.0509777, 7.8764830, 11.0702953, -3.0013828, 3.0030208
9: -7.1151929, -3.2647555, -7.1326494, -3.2509146, -3.5123444, 3.5218697

Time for backsubstitution: 12.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 4555

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1068677, upper bound: 2.1068684
time: 4.77 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1068677, upper bound: 2.1170301
time: 5.04 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -6.1878967, -1.7817087, -6.1879401, -1.7817001, -4.3879900, 4.3907700
1: -15.2638922, -10.7425289, -15.2639389, -10.7425194, -4.3801460, 4.3842912
2: -9.0594788, -4.6674643, -9.0594826, -4.6674337, -4.0914307, 4.0827498
3: -7.5704508, -3.5940967, -7.5704584, -3.5940869, -3.9763639, 3.9763618
4: -12.2075539, -7.3990283, -12.2075672, -7.3990169, -4.8085370, 4.8085389
5: -5.9838586, -2.2321703, -5.9838696, -2.2321639, -3.6449080, 3.6288338
6: -13.7704639, -9.3012924, -13.7705021, -9.3012848, -3.9190407, 3.9210968
7: -10.2259960, -5.8782253, -10.2260056, -5.8782067, -4.3477893, 4.3477802
8: 7.8684001, 11.0728378, 7.8683767, 11.0728455, -3.0239210, 3.0334911
9: -7.1385384, -3.2466509, -7.1385541, -3.2466421, -3.5551682, 3.5456028

Time for backsubstitution: 12.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 4555

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1170294, upper bound: 2.1068683
time: 5.36 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1170293, upper bound: 2.1068679
time: 6.36 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -6.1277590, -1.8190880, -6.1964846, -1.7598429, -4.3531475, 4.3658371
1: -15.2049656, -10.7705097, -15.2958775, -10.7068195, -4.3590937, 4.4935994
2: -9.0476599, -4.7045918, -9.1572723, -4.5703087, -4.1721549, 4.1393046
3: -7.5520787, -3.6056707, -7.6132312, -3.5499575, -4.0021210, 4.0075607
4: -12.1676674, -7.4248271, -12.2759933, -7.3616180, -4.8060493, 4.8511662
5: -5.9627142, -2.2491808, -6.0288191, -2.1905346, -3.6432190, 3.6512876
6: -13.7271786, -9.3266354, -13.8019657, -9.2507925, -3.9361687, 3.9236021
7: -10.2019348, -5.9000587, -10.2674522, -5.8701043, -4.3318305, 4.3673935
8: 7.8975544, 11.0509777, 7.8195705, 11.1139326, -3.0471258, 3.1342237
9: -7.1151929, -3.2647555, -7.1694450, -3.2277384, -3.6652365, 3.5616870

Time for backsubstitution: 12.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 4555

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1259578, upper bound: 2.1068669
time: 6.86 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1259577, upper bound: 2.1170286
time: 6.93 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6.1878967, -1.7817087, -6.2133927, -1.7566986, -4.4116907, 4.4172993
1: -15.2638922, -10.7425289, -15.3121719, -10.7039547, -4.4221582, 4.5433683
2: -9.0594788, -4.6674643, -9.1586075, -4.5603285, -4.1970510, 4.1754074
3: -7.5704508, -3.5940967, -7.6162868, -3.5472484, -4.0232024, 4.0221901
4: -12.2075539, -7.3990283, -12.2817268, -7.3541946, -4.8533592, 4.8826985
5: -5.9838586, -2.2321703, -6.0326061, -2.1862910, -3.6910477, 3.6748524
6: -13.7704639, -9.3012924, -13.8142910, -9.2481089, -3.9816179, 3.9595151
7: -10.2259960, -5.8782253, -10.2709599, -5.8640852, -4.3619108, 4.3927345
8: 7.8684001, 11.0728378, 7.8114195, 11.1164818, -3.0696621, 3.1609793
9: -7.1385384, -3.2466509, -7.1753507, -3.2234247, -3.7083755, 3.5854464

Time for backsubstitution: 12.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 4555

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1361204, upper bound: 2.1068670
time: 14.31 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1361204, upper bound: 2.1068677
time: 5.73 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.1533055, -1.7940526, -6.1710334, -1.7848673, -4.3560133, 4.3630447
1: -15.2530642, -10.7319374, -15.2476873, -10.7453871, -4.4771547, 4.3750353
2: -9.1467924, -4.5976739, -9.0581493, -4.6773758, -4.1550226, 4.1563582
3: -7.5978928, -3.5588391, -7.5674262, -3.5967965, -4.0010962, 4.0085869
4: -12.2417517, -7.3800268, -12.2017918, -7.4064312, -4.8353205, 4.8217649
5: -6.0114632, -2.2032671, -5.9800730, -2.2364025, -3.6430359, 3.6514812
6: -13.7709227, -9.2735767, -13.7581863, -9.3039322, -3.9195890, 3.9452448
7: -10.2466421, -5.8859348, -10.2225571, -5.8842268, -4.3624153, 4.3366222
8: 7.8407826, 11.0946245, 7.8764830, 11.0702953, -3.1383877, 3.0473514
9: -7.1516862, -3.2416601, -7.1326494, -3.2509146, -3.5520258, 3.6750741

Time for backsubstitution: 12.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4555

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1068674, upper bound: 2.1259582
time: 8.56 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1068674, upper bound: 2.1361210
time: 9.60 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6.2133527, -1.7567067, -6.1879401, -1.7817001, -4.4145193, 4.4144621
1: -15.3121233, -10.7039614, -15.2639389, -10.7425194, -4.5397024, 4.4263062
2: -9.1586027, -4.5603609, -9.0594826, -4.6674337, -4.1799402, 4.1923051
3: -7.6162786, -3.5472536, -7.5704584, -3.5940869, -4.0221920, 4.0232048
4: -12.2817135, -7.3542094, -12.2075672, -7.3990169, -4.8826966, 4.8533578
5: -6.0325971, -2.1862960, -5.9838696, -2.2321639, -3.6886549, 3.6749716
6: -13.8142548, -9.2481155, -13.7705021, -9.3012848, -3.9645929, 3.9835143
7: -10.2709513, -5.8641047, -10.2260056, -5.8782067, -4.3927445, 4.3619008
8: 7.8114424, 11.1164742, 7.8683767, 11.0728455, -3.1603618, 3.0739861
9: -7.1753325, -3.2234309, -7.1385541, -3.2466421, -3.5950108, 3.6994686

Time for backsubstitution: 12.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4555

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1170291, upper bound: 2.1259584
time: 5.39 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1170290, upper bound: 2.1376373
time: 7.55 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.1533055, -1.7940526, -6.1964846, -1.7598429, -4.3934627, 4.4024320
1: -15.2530642, -10.7319374, -15.2958775, -10.7068195, -4.5423613, 4.5511589
2: -9.1467924, -4.5976739, -9.1572723, -4.5703087, -4.2546287, 4.2388229
3: -7.5978928, -3.5588391, -7.6132312, -3.5499575, -4.0278196, 4.0320163
4: -12.2417517, -7.3800268, -12.2759933, -7.3616180, -4.8801336, 4.8959665
5: -6.0114632, -2.2032671, -6.0288191, -2.1905346, -3.6823807, 3.6906581
6: -13.7709227, -9.2735767, -13.8019657, -9.2507925, -3.9529009, 3.9620051
7: -10.2466421, -5.8859348, -10.2674522, -5.8701043, -4.3765378, 4.3815174
8: 7.8407826, 11.0946245, 7.8195705, 11.1139326, -3.1737108, 3.1741333
9: -7.1516862, -3.2416601, -7.1694450, -3.2277384, -3.7243452, 3.7341385

Time for backsubstitution: 12.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 4555

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1068649, upper bound: 2.1259594
time: 6.09 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1068649, upper bound: 2.1361221
time: 5.77 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6.2133527, -1.7567067, -6.2133927, -1.7566986, -4.4566541, 4.4566860
1: -15.3121233, -10.7039614, -15.3121719, -10.7039547, -4.6042418, 4.5959563
2: -9.1586027, -4.5603609, -9.1586075, -4.5603285, -4.2795191, 4.2747731
3: -7.6162786, -3.5472536, -7.6162868, -3.5472484, -4.0477066, 4.0465097
4: -12.2817135, -7.3542094, -12.2817268, -7.3541946, -4.9275188, 4.9275174
5: -6.0325971, -2.1862960, -6.0326061, -2.1862910, -3.7303209, 3.7141962
6: -13.8142548, -9.2481155, -13.8142910, -9.2481089, -3.9983864, 4.0002828
7: -10.2709513, -5.8641047, -10.2709599, -5.8640852, -4.4068661, 4.4068551
8: 7.8114424, 11.1164742, 7.8114195, 11.1164818, -3.1963587, 3.2008865
9: -7.1753325, -3.2234309, -7.1753507, -3.2234247, -3.7674665, 3.7585588

Time for backsubstitution: 12.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 4555

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1170267, upper bound: 2.1259592
time: 6.43 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1170267, upper bound: 2.1259590
time: 9.28 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 28.21 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.21
Output dim: 8, lower bound: -2.1068677, upper bound: 2.1068684
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.21
Output dim: 8, lower bound: -2.1068677, upper bound: 2.1170301
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.21
Output dim: 8, lower bound: -2.1170294, upper bound: 2.1068683
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.21
Output dim: 8, lower bound: -2.1170293, upper bound: 2.1068679
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.21
Output dim: 8, lower bound: -2.1259578, upper bound: 2.1068669
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.21
Output dim: 8, lower bound: -2.1259577, upper bound: 2.1170286
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.21
Output dim: 8, lower bound: -2.1361204, upper bound: 2.1068670
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.21
Output dim: 8, lower bound: -2.1361204, upper bound: 2.1068677
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.21
Output dim: 8, lower bound: -2.1068674, upper bound: 2.1259582
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.21
Output dim: 8, lower bound: -2.1068674, upper bound: 2.1361210
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.21
Output dim: 8, lower bound: -2.1170291, upper bound: 2.1259584
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.21
Output dim: 8, lower bound: -2.1170290, upper bound: 2.1376373
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.21
Output dim: 8, lower bound: -2.1068649, upper bound: 2.1259594
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.21
Output dim: 8, lower bound: -2.1068649, upper bound: 2.1361221
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.21
Output dim: 8, lower bound: -2.1170267, upper bound: 2.1259592
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.21
Output dim: 8, lower bound: -2.1170267, upper bound: 2.1259590

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.1277590, -1.8190880, -6.1277590, -1.8190880, -4.2970085, 4.2970085
1: -15.2049656, -10.7705097, -15.2049656, -10.7705097, -4.2901001, 4.2900996
2: -9.0476599, -4.7045918, -9.0476599, -4.7045918, -4.0349989, 4.0349989
3: -7.5520787, -3.6056707, -7.5520787, -3.6056707, -3.9464080, 3.9464080
4: -12.1676674, -7.4248271, -12.1676674, -7.4248271, -4.7428403, 4.7428403
5: -5.9627142, -2.2491808, -5.9627142, -2.2491808, -3.5824118, 3.5824118
6: -13.7271786, -9.3266354, -13.7271786, -9.3266354, -3.8514385, 3.8514385
7: -10.2019348, -5.9000587, -10.2019348, -5.9000587, -4.3018761, 4.3018761
8: 7.8975544, 11.0509777, 7.8975544, 11.0509777, -2.9817109, 2.9817119
9: -7.1151929, -3.2647555, -7.1151929, -3.2647555, -3.5006576, 3.5006576

Time for backsubstitution: 13.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1068661, upper bound: 2.1046363
time: 9.45 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1068661, upper bound: 2.1068656
time: 5.00 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.1277590, -1.8190880, -6.1878967, -1.7817087, -4.3316727, 4.3560543
1: -15.2049656, -10.7705097, -15.2638922, -10.7425289, -4.3184175, 4.3505554
2: -9.0476599, -4.7045918, -9.0594788, -4.6674643, -4.0784206, 4.0479755
3: -7.5520787, -3.6056707, -7.5704508, -3.5940967, -3.9579821, 3.9647801
4: -12.1676674, -7.4248271, -12.2075539, -7.3990283, -4.7686391, 4.7827268
5: -5.9627142, -2.2491808, -5.9838586, -2.2321703, -3.6026402, 3.6036434
6: -13.7271786, -9.3266354, -13.7704639, -9.3012924, -3.8764324, 3.8960652
7: -10.2019348, -5.9000587, -10.2259960, -5.8782253, -4.3237095, 4.3259373
8: 7.8975544, 11.0509777, 7.8684001, 11.0728378, -3.0040150, 3.0111628
9: -7.1151929, -3.2647555, -7.1385384, -3.2466509, -3.5187426, 3.5257339

Time for backsubstitution: 12.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1068662, upper bound: 2.1147980
time: 8.09 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1068661, upper bound: 2.1170273
time: 5.29 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6.1878967, -1.7817087, -6.1277590, -1.8190880, -4.3560543, 4.3316727
1: -15.2638922, -10.7425289, -15.2049656, -10.7705097, -4.3505554, 4.3184180
2: -9.0594788, -4.6674643, -9.0476599, -4.7045918, -4.0479755, 4.0784206
3: -7.5704508, -3.5940967, -7.5520787, -3.6056707, -3.9647801, 3.9579821
4: -12.2075539, -7.3990283, -12.1676674, -7.4248271, -4.7827268, 4.7686391
5: -5.9838586, -2.2321703, -5.9627142, -2.2491808, -3.6036425, 3.6026402
6: -13.7704639, -9.3012924, -13.7271786, -9.3266354, -3.8960657, 3.8764324
7: -10.2259960, -5.8782253, -10.2019348, -5.9000587, -4.3259373, 4.3237095
8: 7.8684001, 11.0728378, 7.8975544, 11.0509777, -3.0111632, 3.0040145
9: -7.1385384, -3.2466509, -7.1151929, -3.2647555, -3.5257339, 3.5187426

Time for backsubstitution: 12.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1170259, upper bound: 2.1046357
time: 9.53 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1170260, upper bound: 2.1068655
time: 7.38 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6.1878967, -1.7817087, -6.1878967, -1.7817087, -4.3879833, 4.3879843
1: -15.2638922, -10.7425289, -15.2638922, -10.7425289, -4.3842449, 4.3842454
2: -9.0594788, -4.6674643, -9.0594788, -4.6674643, -4.0827446, 4.0827451
3: -7.5704508, -3.5940967, -7.5704508, -3.5940967, -3.9763541, 3.9763541
4: -12.2075539, -7.3990283, -12.2075539, -7.3990283, -4.8085256, 4.8085256
5: -5.9838586, -2.2321703, -5.9838586, -2.2321703, -3.6448879, 3.6448884
6: -13.7704639, -9.3012924, -13.7704639, -9.3012924, -3.9190350, 3.9190350
7: -10.2259960, -5.8782253, -10.2259960, -5.8782253, -4.3477707, 4.3477707
8: 7.8684001, 11.0728378, 7.8684001, 11.0728378, -3.0239143, 3.0239139
9: -7.1385384, -3.2466509, -7.1385384, -3.2466509, -3.5551481, 3.5551476

Time for backsubstitution: 12.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1170266, upper bound: 2.1046367
time: 6.42 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1170266, upper bound: 2.1068646
time: 5.79 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -6.1277590, -1.8190880, -6.1533055, -1.7940526, -4.3207397, 4.3235922
1: -15.2049656, -10.7705097, -15.2530642, -10.7319374, -4.3321238, 4.4508500
2: -9.0476599, -4.7045918, -9.1467924, -4.5976739, -4.1449566, 4.1279488
3: -7.5520787, -3.6056707, -7.5978928, -3.5588391, -3.9932396, 3.9922221
4: -12.1676674, -7.4248271, -12.2417517, -7.3800268, -4.7876406, 4.8169246
5: -5.9627142, -2.2491808, -6.0114632, -2.2032671, -3.6285706, 3.6283455
6: -13.7271786, -9.3266354, -13.7709227, -9.2735767, -3.9131804, 3.8969612
7: -10.2019348, -5.9000587, -10.2466421, -5.8859348, -4.3160000, 4.3465834
8: 7.8975544, 11.0509777, 7.8407826, 11.0946245, -3.0274706, 3.1187921
9: -7.1151929, -3.2647555, -7.1516862, -3.2416601, -3.6531763, 3.5403385

Time for backsubstitution: 12.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1259561, upper bound: 2.1046351
time: 7.26 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1259573, upper bound: 2.1068642
time: 7.25 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -6.1277590, -1.8190880, -6.2133527, -1.7567067, -4.3553696, 4.3825836
1: -15.2049656, -10.7705097, -15.3121233, -10.7039614, -4.3604307, 4.5108347
2: -9.0476599, -4.7045918, -9.1586027, -4.5603609, -4.1789055, 4.1367426
3: -7.5520787, -3.6056707, -7.6162786, -3.5472536, -4.0048251, 4.0106077
4: -12.1676674, -7.4248271, -12.2817135, -7.3542094, -4.8134580, 4.8568864
5: -5.9627142, -2.2491808, -6.0325971, -2.1862960, -3.6487799, 3.6495919
6: -13.7271786, -9.3266354, -13.8142548, -9.2481155, -3.9386826, 3.9267206
7: -10.2019348, -5.9000587, -10.2709513, -5.8641047, -4.3378301, 4.3708925
8: 7.8975544, 11.0509777, 7.8114424, 11.1164742, -3.0444746, 3.1355014
9: -7.1151929, -3.2647555, -7.1753325, -3.2234309, -3.6716042, 3.5656371

Time for backsubstitution: 12.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1259561, upper bound: 2.1147978
time: 13.88 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1259562, upper bound: 2.1170260
time: 6.52 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6.1878967, -1.7817087, -6.1533055, -1.7940526, -4.3797846, 4.3582563
1: -15.2638922, -10.7425289, -15.2530642, -10.7319374, -4.3877344, 4.4785948
2: -9.0594788, -4.6674643, -9.1467924, -4.5976739, -4.1537552, 4.1617494
3: -7.5704508, -3.5940967, -7.5978928, -3.5588391, -4.0116119, 4.0037961
4: -12.2075539, -7.3990283, -12.2417517, -7.3800268, -4.8275270, 4.8427234
5: -5.9838586, -2.2321703, -6.0114632, -2.2032671, -3.6498013, 3.6486249
6: -13.7704639, -9.3012924, -13.7709227, -9.2735767, -3.9579687, 3.9146934
7: -10.2259960, -5.8782253, -10.2466421, -5.8859348, -4.3400612, 4.3684168
8: 7.8684001, 11.0728378, 7.8407826, 11.0946245, -3.0486016, 3.1310861
9: -7.1385384, -3.2466509, -7.1516862, -3.2416601, -3.6797466, 3.5584230

Time for backsubstitution: 12.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1361176, upper bound: 2.1046360
time: 5.13 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1361180, upper bound: 2.1068649
time: 6.31 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6.1878967, -1.7817087, -6.2133527, -1.7567067, -4.4116879, 4.4145126
1: -15.2638922, -10.7425289, -15.3121233, -10.7039614, -4.4262600, 4.5433207
2: -9.0594788, -4.6674643, -9.1586027, -4.5603609, -4.1922541, 4.1752872
3: -7.5704508, -3.5940967, -7.6162786, -3.5472536, -4.0231972, 4.0221820
4: -12.2075539, -7.3990283, -12.2817135, -7.3542094, -4.8533444, 4.8826852
5: -5.9838586, -2.2321703, -6.0325971, -2.1862960, -3.6910267, 3.6886487
6: -13.7704639, -9.3012924, -13.8142548, -9.2481155, -3.9816113, 3.9584312
7: -10.2259960, -5.8782253, -10.2709513, -5.8641047, -4.3618913, 4.3927259
8: 7.8684001, 11.0728378, 7.8114424, 11.1164742, -3.0688643, 3.1556857
9: -7.1385384, -3.2466509, -7.1753325, -3.2234309, -3.7083569, 3.5949898

Time for backsubstitution: 13.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1361180, upper bound: 2.1084527
time: 10.30 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1361180, upper bound: 2.1068642
time: 5.31 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.1533055, -1.7940526, -6.1277590, -1.8190880, -4.3235912, 4.3207397
1: -15.2530642, -10.7319374, -15.2049656, -10.7705097, -4.4508495, 4.3321238
2: -9.1467924, -4.5976739, -9.0476599, -4.7045918, -4.1279483, 4.1449571
3: -7.5978928, -3.5588391, -7.5520787, -3.6056707, -3.9922221, 3.9932396
4: -12.2417517, -7.3800268, -12.1676674, -7.4248271, -4.8169246, 4.7876406
5: -6.0114632, -2.2032671, -5.9627142, -2.2491808, -3.6283455, 3.6285706
6: -13.7709227, -9.2735767, -13.7271786, -9.3266354, -3.8969612, 3.9131804
7: -10.2466421, -5.8859348, -10.2019348, -5.9000587, -4.3465834, 4.3160000
8: 7.8407826, 11.0946245, 7.8975544, 11.0509777, -3.1187921, 3.0274701
9: -7.1516862, -3.2416601, -7.1151929, -3.2647555, -3.5403380, 3.6531768

Time for backsubstitution: 13.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1068657, upper bound: 2.1237713
time: 7.47 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1068658, upper bound: 2.1259555
time: 7.51 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.1533055, -1.7940526, -6.1878967, -1.7817087, -4.3582563, 4.3797846
1: -15.2530642, -10.7319374, -15.2638922, -10.7425289, -4.4785948, 4.3877339
2: -9.1467924, -4.5976739, -9.0594788, -4.6674643, -4.1617489, 4.1537547
3: -7.5978928, -3.5588391, -7.5704508, -3.5940967, -4.0037961, 4.0116119
4: -12.2417517, -7.3800268, -12.2075539, -7.3990283, -4.8427234, 4.8275270
5: -6.0114632, -2.2032671, -5.9838586, -2.2321703, -3.6486244, 3.6498022
6: -13.7709227, -9.2735767, -13.7704639, -9.3012924, -3.9146929, 3.9579692
7: -10.2466421, -5.8859348, -10.2259960, -5.8782253, -4.3684168, 4.3400612
8: 7.8407826, 11.0946245, 7.8684001, 11.0728378, -3.1310863, 3.0486016
9: -7.1516862, -3.2416601, -7.1385384, -3.2466509, -3.5584230, 3.6797466

Time for backsubstitution: 13.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1068657, upper bound: 2.1339273
time: 7.16 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1068658, upper bound: 2.1361183
time: 7.65 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6.2133527, -1.7567067, -6.1277590, -1.8190880, -4.3825846, 4.3553696
1: -15.3121233, -10.7039614, -15.2049656, -10.7705097, -4.5108337, 4.3604307
2: -9.1586027, -4.5603609, -9.0476599, -4.7045918, -4.1367426, 4.1789055
3: -7.6162786, -3.5472536, -7.5520787, -3.6056707, -4.0106077, 4.0048251
4: -12.2817135, -7.3542094, -12.1676674, -7.4248271, -4.8568864, 4.8134580
5: -6.0325971, -2.1862960, -5.9627142, -2.2491808, -3.6495914, 3.6487799
6: -13.8142548, -9.2481155, -13.7271786, -9.3266354, -3.9267206, 3.9386835
7: -10.2709513, -5.8641047, -10.2019348, -5.9000587, -4.3708925, 4.3378301
8: 7.8114424, 11.1164742, 7.8975544, 11.0509777, -3.1355014, 3.0444744
9: -7.1753325, -3.2234309, -7.1151929, -3.2647555, -3.5656376, 3.6716046

Time for backsubstitution: 13.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1170256, upper bound: 2.1237700
time: 16.37 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1170256, upper bound: 2.1259553
time: 6.31 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6.2133527, -1.7567067, -6.1878967, -1.7817087, -4.4145126, 4.4116879
1: -15.3121233, -10.7039614, -15.2638922, -10.7425289, -4.5433207, 4.4262605
2: -9.1586027, -4.5603609, -9.0594788, -4.6674643, -4.1752872, 4.1922545
3: -7.6162786, -3.5472536, -7.5704508, -3.5940967, -4.0221820, 4.0231972
4: -12.2817135, -7.3542094, -12.2075539, -7.3990283, -4.8826852, 4.8533444
5: -6.0325971, -2.1862960, -5.9838586, -2.2321703, -3.6886482, 3.6910272
6: -13.8142548, -9.2481155, -13.7704639, -9.3012924, -3.9584312, 3.9816108
7: -10.2709513, -5.8641047, -10.2259960, -5.8782253, -4.3927259, 4.3618913
8: 7.8114424, 11.1164742, 7.8684001, 11.0728378, -3.1556854, 3.0688639
9: -7.1753325, -3.2234309, -7.1385384, -3.2466509, -3.5949888, 3.7083559

Time for backsubstitution: 12.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1170263, upper bound: 2.1237712
time: 5.94 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1170263, upper bound: 2.1259550
time: 12.98 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -6.1533055, -1.7940526, -6.1533055, -1.7940526, -4.3592529, 4.3592529
1: -15.2530642, -10.7319374, -15.2530642, -10.7319374, -4.5160265, 4.5160265
2: -9.1467924, -4.5976739, -9.1467924, -4.5976739, -4.2274313, 4.2274308
3: -7.5978928, -3.5588391, -7.5978928, -3.5588391, -4.0178337, 4.0178337
4: -12.2417517, -7.3800268, -12.2417517, -7.3800268, -4.8617249, 4.8617249
5: -6.0114632, -2.2032671, -6.0114632, -2.2032671, -3.6677132, 3.6677132
6: -13.7709227, -9.2735767, -13.7709227, -9.2735767, -3.9299107, 3.9299107
7: -10.2466421, -5.8859348, -10.2466421, -5.8859348, -4.3607073, 4.3607073
8: 7.8407826, 11.0946245, 7.8407826, 11.0946245, -3.1541042, 3.1541033
9: -7.1516862, -3.2416601, -7.1516862, -3.2416601, -3.7122588, 3.7122588

Time for backsubstitution: 12.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1068634, upper bound: 2.1237723
time: 5.77 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1068634, upper bound: 2.1259558
time: 5.49 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 23.99 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.99
Output dim: 8, lower bound: -2.1068661, upper bound: 2.1046363
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.99
Output dim: 8, lower bound: -2.1068661, upper bound: 2.1068656
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.99
Output dim: 8, lower bound: -2.1068662, upper bound: 2.1147980
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.99
Output dim: 8, lower bound: -2.1068661, upper bound: 2.1170273
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.99
Output dim: 8, lower bound: -2.1170259, upper bound: 2.1046357
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.99
Output dim: 8, lower bound: -2.1170260, upper bound: 2.1068655
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.99
Output dim: 8, lower bound: -2.1170266, upper bound: 2.1046367
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.99
Output dim: 8, lower bound: -2.1170266, upper bound: 2.1068646
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.99
Output dim: 8, lower bound: -2.1259561, upper bound: 2.1046351
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.99
Output dim: 8, lower bound: -2.1259573, upper bound: 2.1068642
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.99
Output dim: 8, lower bound: -2.1259561, upper bound: 2.1147978
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.99
Output dim: 8, lower bound: -2.1259562, upper bound: 2.1170260
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.99
Output dim: 8, lower bound: -2.1361176, upper bound: 2.1046360
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.99
Output dim: 8, lower bound: -2.1361180, upper bound: 2.1068649
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.99
Output dim: 8, lower bound: -2.1361180, upper bound: 2.1084527
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.99
Output dim: 8, lower bound: -2.1361180, upper bound: 2.1068642
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.99
Output dim: 8, lower bound: -2.1068657, upper bound: 2.1237713
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.99
Output dim: 8, lower bound: -2.1068658, upper bound: 2.1259555
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.99
Output dim: 8, lower bound: -2.1068657, upper bound: 2.1339273
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.99
Output dim: 8, lower bound: -2.1068658, upper bound: 2.1361183
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.99
Output dim: 8, lower bound: -2.1170256, upper bound: 2.1237700
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.99
Output dim: 8, lower bound: -2.1170256, upper bound: 2.1259553
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.99
Output dim: 8, lower bound: -2.1170263, upper bound: 2.1237712
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.99
Output dim: 8, lower bound: -2.1170263, upper bound: 2.1259550
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.99
Output dim: 8, lower bound: -2.1068634, upper bound: 2.1237723
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.99
Output dim: 8, lower bound: -2.1068634, upper bound: 2.1259558
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.99
Output dim: 8, lower bound: -2.1068649, upper bound: 2.1361221
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.99
Output dim: 8, lower bound: -2.1170267, upper bound: 2.1259592
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.99
Output dim: 8, lower bound: -2.1170267, upper bound: 2.1259590
Binary search (step 1): status=Status.UNKNOWN, k_low=6, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=3.226428508758545
rel_dist={8: [-2.137668074409355, 2.1376683881402094]}

## Binary search (step 2) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6195
type: A, layer: 1, pos: 4555
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6195

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.0089420, upper bound: 1.9922138
time: 6.00 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.0090751, upper bound: 2.0090750
time: 10.12 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 16.25 seconds
IS_A1, status: Status.VERIFIED, split count: 1, time: 16.25
Output dim: 8, lower bound: -2.0089420, upper bound: 1.9922138
IS_A2, status: Status.UNKNOWN, split count: 1, time: 16.25
Output dim: 8, lower bound: -2.0090751, upper bound: 2.0090750

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -6.2133999, -1.7566977, -6.2134047, -1.7566948, -4.3524456, 4.3856573
1: -15.3121786, -10.7039528, -15.3121920, -10.7039490, -4.4694004, 4.4849429
2: -9.1586056, -4.5603261, -9.1586761, -4.5603189, -4.1999350, 4.2003827
3: -7.6162882, -3.5472469, -7.6163082, -3.5472400, -3.9469023, 3.9834385
4: -12.2817307, -7.3541937, -12.2817430, -7.3541608, -4.9275699, 4.9275494
5: -6.0326066, -2.1862900, -6.0326147, -2.1862636, -3.6336031, 3.6262121
6: -13.8142948, -9.2481070, -13.8143024, -9.2480831, -3.9104118, 3.8808870
7: -10.2709599, -5.8640842, -10.2709703, -5.8640814, -4.4068785, 4.4068861
8: 7.8114176, 11.1164827, 7.8114080, 11.1164942, -3.1325798, 3.1209745
9: -7.1753492, -3.2234240, -7.1753573, -3.2234144, -3.6593885, 3.6700058

Time for backsubstitution: 13.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6195
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6195

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.9922117, upper bound: 2.0089420
time: 7.60 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9922117, upper bound: 2.0090741
time: 8.11 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 28.91 seconds
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 28.91
Output dim: 8, lower bound: -1.9922117, upper bound: 2.0089420
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 28.91
Output dim: 8, lower bound: -1.9922117, upper bound: 2.0090741

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -6.2133999, -1.7566977, -6.2133999, -1.7566977, -4.3856344, 4.3856344
1: -15.3121786, -10.7039528, -15.3121786, -10.7039528, -4.4849286, 4.4849286
2: -9.1586056, -4.5603261, -9.1586056, -4.5603261, -4.1800957, 4.1800957
3: -7.6162882, -3.5472469, -7.6162882, -3.5472469, -3.9468966, 3.9468966
4: -12.2817307, -7.3541937, -12.2817307, -7.3541937, -4.9275370, 4.9275370
5: -6.0326066, -2.1862900, -6.0326066, -2.1862900, -3.6262064, 3.6262059
6: -13.8142948, -9.2481070, -13.8142948, -9.2481070, -3.8808804, 3.8808799
7: -10.2709599, -5.8640842, -10.2709599, -5.8640842, -4.4068756, 4.4068756
8: 7.8114176, 11.1164827, 7.8114176, 11.1164827, -3.1209674, 3.1209669
9: -7.1753492, -3.2234240, -7.1753492, -3.2234240, -3.6699905, 3.6699905

Time for backsubstitution: 13.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4555
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 4555

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.9816170, upper bound: 2.0071872
time: 8.89 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9921989, upper bound: 2.0090616
time: 11.26 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 33.38 seconds
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 33.38
Output dim: 8, lower bound: -1.9816170, upper bound: 2.0071872
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 33.38
Output dim: 8, lower bound: -1.9921989, upper bound: 2.0090616

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6.2133527, -1.7567067, -6.2133880, -1.7567015, -4.3826895, 4.3856192
1: -15.3121233, -10.7039614, -15.3121691, -10.7039528, -4.4816904, 4.4749427
2: -9.1586027, -4.5603609, -9.1586065, -4.5603333, -4.1758585, 4.1705775
3: -7.6162786, -3.5472536, -7.6162863, -3.5472472, -3.9468870, 3.9456205
4: -12.2817135, -7.3542094, -12.2817249, -7.3541985, -4.9275150, 4.9275155
5: -6.0325971, -2.1862960, -6.0326042, -2.1862922, -3.6412001, 3.6252341
6: -13.8142548, -9.2481155, -13.8142891, -9.2481098, -3.8788624, 3.8808641
7: -10.2709513, -5.8641047, -10.2709589, -5.8640852, -4.4068661, 4.4068542
8: 7.8114424, 11.1164742, 7.8114214, 11.1164799, -3.1104784, 3.1144476
9: -7.1753325, -3.2234309, -7.1753469, -3.2234247, -3.6782589, 3.6694427

Time for backsubstitution: 13.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 4555

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.9905065, upper bound: 1.9984887
time: 6.70 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9905066, upper bound: 2.0090629
time: 9.22 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 29.15 seconds
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 29.15
Output dim: 8, lower bound: -1.9905065, upper bound: 1.9984887
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.15
Output dim: 8, lower bound: -1.9905066, upper bound: 2.0090629

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6.2133527, -1.7567067, -6.2133527, -1.7567067, -4.3826847, 4.3826847
1: -15.3121233, -10.7039614, -15.3121233, -10.7039614, -4.4749279, 4.4749284
2: -9.1586027, -4.5603609, -9.1586027, -4.5603609, -4.1705341, 4.1705337
3: -7.6162786, -3.5472536, -7.6162786, -3.5472536, -3.9456148, 3.9456148
4: -12.2817135, -7.3542094, -12.2817135, -7.3542094, -4.9275041, 4.9275041
5: -6.0325971, -2.1862960, -6.0325971, -2.1862960, -3.6411819, 3.6411824
6: -13.8142548, -9.2481155, -13.8142548, -9.2481155, -3.8788548, 3.8788548
7: -10.2709513, -5.8641047, -10.2709513, -5.8641047, -4.4068465, 4.4068465
8: 7.8114424, 11.1164742, 7.8114424, 11.1164742, -3.1085691, 3.1085694
9: -7.1753325, -3.2234309, -7.1753325, -3.2234309, -3.6782427, 3.6782427

Time for backsubstitution: 12.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.9905040, upper bound: 2.0005314
time: 5.56 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.9905040, upper bound: 1.9984846
time: 5.59 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 23.63 seconds
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 23.63
Output dim: 8, lower bound: -1.9905040, upper bound: 2.0005314
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 23.63
Output dim: 8, lower bound: -1.9905040, upper bound: 1.9984846
Binary search (step 2): status=Status.VERIFIED, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=3.14231538772583
rel_dist={8: [-2.00908813677729, 2.009087543266727]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0234375
execution time: 1796.22 seconds
